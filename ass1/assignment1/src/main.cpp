#include <iostream>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
/*** insert any libigl headers here ***/

using namespace std;
using Viewer = igl::opengl::glfw::Viewer;

// Vertex array, #V x3
Eigen::MatrixXd V;
// Face array, #F x3
Eigen::MatrixXi F;
// Per-face normal array, #F x3
Eigen::MatrixXd FN;
// Per-vertex normal array, #V x3
Eigen::MatrixXd VN;
// Per-corner normal array, (3#F) x3
Eigen::MatrixXd CN;
// Vectors of indices for adjacency relations
std::vector<std::vector<int> > VF, VFi, VV;
// Integer vector of component IDs per face, #F x1
Eigen::VectorXi cid;
// Per-face color array, #F x3
Eigen::MatrixXd component_colors_per_face;
//flags calcs
bool vfcalc,vvcalc;

void vertex_face_adjacency(
    const Eigen::Matrix<double, -1, 3>& V,
    const Eigen::Matrix<int, -1, 3>& F,
    std::vector<std::vector<int>>& faces_in_vertex/*[vertex][faces]*/)
{
    faces_in_vertex.resize(V.rows());
    for (int i = 0; i < F.rows(); i++) {
        for (int j = 0; j < 3; j++)
            faces_in_vertex[F(i, j)].push_back(i);
    }
}
//Adjacency Vertex-to-Vertex
//	Generates a list neighbour vertex
template<bool remove_duplicate>
void vertex_vertex_adjacency(
    const Eigen::Matrix<double, -1, 3>& V,
    const Eigen::Matrix<int, -1, 3>& F,
    std::vector<std::vector<int>>& vertex_neighbours/*[vertex][vertex]*/)
{
    vertex_neighbours.resize(V.rows());
    for (int i = 0; i < F.rows(); i++) {
        //0 12
        vertex_neighbours[F(i, 0)].push_back(F(i, 1));
        vertex_neighbours[F(i, 0)].push_back(F(i, 2));
        //1 02
        vertex_neighbours[F(i, 1)].push_back(F(i, 0));
        vertex_neighbours[F(i, 1)].push_back(F(i, 2));
        //2 01
        vertex_neighbours[F(i, 2)].push_back(F(i, 0));
        vertex_neighbours[F(i, 2)].push_back(F(i, 1));
    }
    if (remove_duplicate) {
        for (auto& vec : vertex_neighbours) {
            sort(vec.begin(), vec.end());
            vec.erase(unique(vec.begin(), vec.end()), vec.end());
        }
    }
}
void showidx(vector<vector<int>>& idx) {
    for (int i = 0; i < idx.size(); i++) {
        cout << '\n' << i << '\t';
        for (auto& f : idx[i])
            cout << f << '\t';
    }
}

bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers) {
    if (key == '1') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        // Add your code for computing vertex to face relations here;
        // store in VF,VFi.
        if(!vfcalc)
            vertex_face_adjacency(V, F, VF);
        showidx(VF);
    }

    if (key == '2') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        // Add your code for computing vertex to vertex relations here:
        // store in VV.
        if(!vvcalc)
            vertex_vertex_adjacency<true>(V, F, VV);
        showidx(VV);
    }

    if (key == '3') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        FN.setZero(F.rows(),3);
        // Add your code for computing per-face normals here: store in FN.

        // Set the viewer normals.
        viewer.data().set_normals(FN);
    }

    if (key == '4') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        // Add your code for computing per-vertex normals here: store in VN.

        // Set the viewer normals.
    }

    if (key == '5') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        // Add your code for computing per-corner normals here: store in CN.

        //Set the viewer normals
    }

    if (key == '6') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        component_colors_per_face.setZero(F.rows(),3);
        // Add your code for computing per-face connected components here:
        // store the component labels in cid.

        // Compute colors for the faces based on components, storing them in
        // component_colors_per_face.

        // Set the viewer colors
        viewer.data().set_colors(component_colors_per_face);
    }

    if (key == '7') {
        Eigen::MatrixXd Vout=V;
        Eigen::MatrixXi Fout=F;
        // Add your code for sqrt(3) subdivision here.

        // Set up the viewer to display the new mesh
        V = Vout; F = Fout;
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
    }

    return true;
}

bool load_mesh(Viewer& viewer,string filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
    if (!igl::readOFF(filename, V, F))
        return false;
  viewer.data().clear();
  viewer.data().set_mesh(V,F);
  viewer.data().compute_normals();
  viewer.core.align_camera_center(V, F);
  return true;
}

int main(int argc, char *argv[]) {
    vfcalc = vvcalc = false;
    // Show the mesh
    Viewer viewer;
    viewer.callback_key_down = callback_key_down;
    
    std::string filename;
    if (argc == 2) {
        filename = std::string(argv[1]);
    }
    else {
        filename = std::string("../data/plane.off");
    }
    if (!load_mesh(viewer, filename, V, F))
        return EXIT_FAILURE;

    callback_key_down(viewer, '0', 0);

    viewer.launch();
}
