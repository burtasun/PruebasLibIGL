#include "AmbientOcclusionExpanded.h"

#include <igl/avg_edge_length.h>
#include <igl/per_vertex_normals.h>
#include <igl/readOFF.h>
#include <igl/embree/ambient_occlusion.h>
#include <igl/opengl/glfw/Viewer.h>
#include <iostream>

#define M_PI 3.14159265358979323846
#define rnd_norm static_cast<double>(rand()) / static_cast<double>(RAND_MAX)//0 - 1
#define RAD2DEG(rad) rad*180/ M_PI
#define DEG2RAD(deg) deg*M_PI/180

// Mesh
Eigen::MatrixXd V;
Eigen::MatrixXi F;

Eigen::VectorXd AO;//ratio no occlud
Eigen::MatrixXd VO;//vector no occlud

Eigen::MatrixXd V_face; //baricentre triangles
Eigen::VectorXd AO_face;
Eigen::MatrixXd VO_face;
Eigen::MatrixXf AO_face_sect;
Eigen::MatrixXf VO_face_sect;

// It allows to change the degree of the field when a number is pressed
bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
	static unsigned int sector = 0;
	using namespace Eigen;
	using namespace std;
	const RowVector3d color(0.9, 0.85, 0.9);
	switch (key)
	{
	case '1':
		// Show the mesh without the ambient occlusion factor
		viewer.data().set_colors(color);
		break;
	case '2':
	{
		// Show the mesh with the ambient occlusion factor
		MatrixXd C = color.replicate(V.rows(), 1);
		for (unsigned i = 0; i < C.rows(); ++i)
			C.row(i) *= AO(i);//std::min<double>(AO(i)+0.2,1);
		viewer.data().set_colors(C);
		break;
	}
	case '3':
	{
		MatrixXd C = color.replicate(V_face.rows(), 1);
		for (unsigned i = 0; i < C.rows(); ++i)
			C.row(i) *= AO_face(i);//std::min<double>(AO(i)+0.2,1);
		viewer.data().set_colors(C);
		break;
	}
	case '-':
		if (++sector >= (AO_face_sect.cols()))
			sector = 0;
	case '4':
	{
		MatrixXd C = color.replicate(V_face.rows(), 1);
		for (unsigned i = 0; i < C.rows(); ++i)
			C.row(i) *= AO_face_sect(i, sector);
		viewer.data().set_colors(C);
		break;
	}
	case '.':
		viewer.core().lighting_factor += 0.1;
		break;
	case ',':
		viewer.core().lighting_factor -= 0.1;
		break;
	default: break;
	}
	viewer.core().lighting_factor =
		std::min(std::max(viewer.core().lighting_factor, 0.f), 1.f);

	return false;
}


int main(int argc, char *argv[])
{
	using namespace std;
	using namespace Eigen;
	cout <<
		"Press 1 to turn off Ambient Occlusion" << endl <<
		"Press 2 to turn on Ambient Occlusion per vertex" << endl <<
		"Press 3 to turn on Ambient Occlusion per face" << endl <<
		"Press 4 to turn on Ambient Occlusion per face sectorized" << endl <<
		"Press - to turn on Ambient Occlusion per face sectorized / change orientation seed" << endl <<
		"Press . to turn up lighting" << endl <<
		"Press , to turn down lighting" << endl;

	bool res = igl::readOFF("Recursos/fertility.off", V, F);
	cout << "V.rows()\t" << V.rows() << endl << "F.rows()\t" << F.rows() << endl;

	//Per vertex
	MatrixXd N;
	igl::per_vertex_normals(V, F, N);
	AmbientOcclusionExpanded::ambient_occlusion_expanded_embree(V, F, V, N, 100, AO, VO);
	AO = 1.0 - AO.array();

	//Per faces
	MatrixXd PFN;
	igl::per_face_normals(V, F, PFN);
	AmbientOcclusionExpanded::face_baricentre(V_face, V, F);
	AmbientOcclusionExpanded::ambient_occlusion_expanded_embree(V_face, F, V_face, PFN, 100, AO_face, VO_face);
	AO_face = 1.0 - AO_face.array();

	//Per faces sectorized
	//	Rejection sampling
	int n_zenith = 40;//Z 0 - 360 subdiv
	int m_azimuth = 10;//X 0 - 90 subdiv
	int n_samples_sector = 5;
	int tot_samples = n_samples_sector * n_zenith * m_azimuth;

	//		normal sector samples
	MatrixXf samples_sector_normal(n_samples_sector, 3);
	double incr_azimuth = M_PI / (2 * n_zenith);//Delta phi sector
	for (int i = 0; i < n_samples_sector; i++) {
		float theta = 2 * M_PI * rnd_norm;//Zenith // Z
		float phi = //Azimuth // X
			incr_azimuth *
			(rnd_norm * 2 - 1); //(-1 , 1)

		samples_sector_normal.row(i) <<
			sin(phi) * cos(theta),
			sin(phi) * sin(theta),
			cos(phi);
	}
	//		roto_translation sectors
	MatrixXf samples_sectors(n_samples_sector, 3 * (n_zenith * (m_azimuth - 1) + 1));
	samples_sectors.block(0, 0, n_samples_sector, 3) = samples_sector_normal;
	int sect_act = 0;
	for (int i = 1; i < m_azimuth; i++) {
		float phi = i * (M_PI / 2) / m_azimuth;
		for (int j = 0; j < n_zenith; j++) {
			float theta = j * (2*M_PI) / n_zenith;
			Matrix3f rot_sector = Isometry3f(
				AngleAxisf(theta, Eigen::Vector3f::UnitZ())*
				AngleAxisf(phi, Eigen::Vector3f::UnitX()) 				
			).rotation();
			cout << "phi\t" << RAD2DEG(phi) << "\t\t theta\t" << RAD2DEG(theta) << theta << endl;
			cout << "rot_sector\n" << rot_sector.format(Eigen::IOFormat(4, 0, ", ", "\n", "[", "]")) << endl << endl;
			++sect_act;
			for (int k = 0; k < n_samples_sector; k++)
				samples_sectors.block(k, 3 * sect_act, 1, 3) = (rot_sector * samples_sector_normal.row(k).transpose()).transpose();
		}
	}

	MatrixXd U_iso;
	AmbientOcclusionExpanded::iso_parameters(U_iso, V, F);
	AmbientOcclusionExpanded::ambient_occlusion_expanded_embree_sector(V_face, F, V_face, PFN, U_iso, samples_sectors, AO_face_sect, VO_face_sect);
	AO_face_sect = 1.0 - AO_face_sect.array();


	//Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
	//std::cout << VO.format(CleanFmt);
	//std::cout << endl << AO.format(CleanFmt) << endl;
	// Show mesh
	igl::opengl::glfw::Viewer viewer;
	viewer.data().set_mesh(V, F);
	viewer.callback_key_down = &key_down;
	key_down(viewer, '4', 0);
	viewer.data().show_lines = false;
	viewer.core().lighting_factor = 0.0f;
	viewer.launch();
}
