#include "AmbientOcclusionExpanded.h"

#include <igl/avg_edge_length.h>
#include <igl/per_vertex_normals.h>
#include <igl/readOFF.h>
#include <igl/embree/ambient_occlusion.h>
#include <igl/opengl/glfw/Viewer.h>
#include <iostream>


// Mesh
Eigen::MatrixXd V;
Eigen::MatrixXi F;

Eigen::VectorXd AO;//ratio no occlud
Eigen::MatrixXd VO;//vector no occlud


Eigen::MatrixXd V_face; //baricentre triangles
Eigen::VectorXd AO_face;
Eigen::MatrixXd VO_face;

// It allows to change the degree of the field when a number is pressed
bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
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


bool face_baricentre(Eigen::MatrixXd &F_baricentres, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) {
	int Frows = F.rows();
	F_baricentres.resize(Frows, 3);
#pragma omp parallel for if (Frows>10000)
	for (int i = 0; i < Frows; i++)
		F_baricentres.row(i) = ((V.row(F(i, 0)) + V.row(F(i, 1)) + V.row(F(i, 2))) / 3).normalized();
	return true;
}

int main(int argc, char *argv[])
{
	using namespace std;
	using namespace Eigen;
	cout <<
		"Press 1 to turn off Ambient Occlusion" << endl <<
		"Press 2 to turn on Ambient Occlusion per vertex" << endl <<
		"Press 3 to turn on Ambient Occlusion per face" << endl <<
		"Press . to turn up lighting" << endl <<
		"Press , to turn down lighting" << endl;

	// Load a mesh in OFF format
	bool res = igl::readOFF("Recursos/fertility.off", V, F);
	cout << "V.rows()\t" << V.rows() << endl << "F.rows()\t" << F.rows() << endl;
	MatrixXd N;
	igl::per_vertex_normals(V, F, N);
	AmbientOcclusionExpanded::ambient_occlusion_expanded_embree(V, F, V, N, 100, AO, VO);
	AO = 1.0 - AO.array();

	MatrixXd PFN;
	igl::per_face_normals(V, F, PFN);
	face_baricentre(V_face, V, F);
	AmbientOcclusionExpanded::ambient_occlusion_expanded_embree(V_face, F, V_face, PFN, 100, AO_face, VO_face);
	AO_face = 1.0 - AO_face.array();

	//Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
	//std::cout << VO.format(CleanFmt);
	//std::cout << endl << AO.format(CleanFmt) << endl;
	// Show mesh
	igl::opengl::glfw::Viewer viewer;
	viewer.data().set_mesh(V, F);
	viewer.callback_key_down = &key_down;
	key_down(viewer, '2', 0);
	viewer.data().show_lines = false;
	viewer.core().lighting_factor = 0.0f;
	viewer.launch();
}
