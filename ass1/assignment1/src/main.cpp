#include <iostream>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
/*** insert any libigl headers here ***/

#define M_PI 3.14159265359 
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
// Face area
vector<double> FArea;
//flags calcs
bool vfcalc, vvcalc, pfncalc;
//others
double degree_thresh;

void vertex_face_adjacency(
    const Eigen::Matrix<double, -1, 3>& V,
    const Eigen::Matrix<int, -1, 3>& F,
    std::vector<std::vector<int>>& faces_in_vertex/*[vertex][faces]*/)
{
    if (vfcalc) return;
    faces_in_vertex.resize(V.rows());
    for (int i = 0; i < F.rows(); i++) {
        for (int j = 0; j < 3; j++)
            faces_in_vertex[F(i, j)].push_back(i);
    }
    vfcalc = true;
}
//Adjacency Vertex-to-Vertex
//	Generates a list neighbour vertex
template<bool remove_duplicate>
void vertex_vertex_adjacency(
    const Eigen::Matrix<double, -1, 3>& V,
    const Eigen::Matrix<int, -1, 3>& F,
    std::vector<std::vector<int>>& vertex_neighbours/*[vertex][vertex]*/)
{
    if (vvcalc)return;
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
    vvcalc = true;
}
void showidx(vector<vector<int>>& idx) {
    for (int i = 0; i < idx.size(); i++) {
        cout << '\n' << i << '\t';
        for (auto& f : idx[i])
            cout << f << '\t';
    }
}

/*per_face_normals*/
void per_face_normals(
    const Eigen::Matrix<double, -1, 3>& V,
    const Eigen::Matrix<int, -1, 3>& F,
    Eigen::Matrix<double, -1, -1>& PFN)
{
    if (pfncalc) return;
    PFN.resize(F.rows(), 3);
    for (int i = 0; i < F.rows(); i++)
        PFN.row(i) =
            (V.row(F(i, 1)) - V.row(F(i, 0))).cross(/*v1*/
                (V.row(F(i, 2)) - V.row(F(i, 0)))/*v2*/
            ).normalized();
    pfncalc = true;
}
void face_areas(
    const Eigen::Matrix<double, -1, 3>& V,
    const Eigen::Matrix<int, -1, 3>& F,
    vector<double>& FArea
    )
{
    FArea.resize(F.rows());
    for (int i = 0; i < F.rows(); i++)
        FArea[i] =
            (V.row(F(i, 1)) - V.row(F(i, 0))).norm() *
            (V.row(F(i, 2)) - V.row(F(i, 0))).norm() / 2.0;
}
/*per_vertex_normals*/
void per_vertex_normals(
    const Eigen::Matrix<double, -1, 3>& V,
    const vector<vector<int>>& VF,
    const Eigen::Matrix<double, -1, -1> PFN,
    const vector<double> &FArea,
    Eigen::Matrix<double, -1, -1>& VN)
{
    VN.resize(V.rows(), 3);
    for (int i = 0; i < V.rows(); i++) {
        VN.row(i) = Eigen::RowVector3d::Zero();
        for (int j = 0; j < VF[i].size(); j++) {
            auto& f = VF[i][j];
            //cout << j << '\t' << f << '\t' << PFN.row(f) << endl;;
            VN.row(i) += FArea[f] * PFN.row(f);
        }
        VN.row(i).normalize();
        //cout << "norm\t" << VN.row(i) << endl << endl;
    }
}

/*per_corner_normals*/
void per_corner_normals(
    const Eigen::Matrix<double, -1, 3>& V,
    const Eigen::Matrix<int, -1, 3>& F,
    const vector<vector<int>>& VF,
    const Eigen::Matrix<double, -1, -1>& PFN,
    const double degree_thresh,
    Eigen::Matrix<double, -1, -1>& PCN
) {
    const auto m = PFN.rows();
    double thresh_cos_rad = cos(degree_thresh * M_PI / 180.0);
    PCN.setZero(m, 3);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < 3; j++) {
            auto& vi = F(i, j);
            for (int k = 0; k < VF[vi].size(); k++) {
                if (PFN.row(VF[vi][k]).dot(PFN.row(i)) >= thresh_cos_rad)
                    PCN.row(i) += PFN.row(VF[vi][k]);
            }
        }
        PCN.row(i).normalize();
    }
}
void AA_Cube(const double dim,
    Eigen::Matrix<double, -1, -1>& V,
    Eigen::Matrix<int, -1, -1>& F)
{
    V.resize(8, 3);
    F.resize(12, 3);
    double d = dim / 2;
    V.row(0) = Eigen::RowVector3d(-d, -d, d);
    V.row(1) = Eigen::RowVector3d(d, -d, d);
    V.row(2) = Eigen::RowVector3d(d, -d, -d);
    V.row(3) = Eigen::RowVector3d(-d, -d, -d);
    V.row(4) = Eigen::RowVector3d(-d, d, d);
    V.row(5) = Eigen::RowVector3d(d, d, d);
    V.row(6) = Eigen::RowVector3d(-d, d, -d);
    V.row(7) = Eigen::RowVector3d(d, d, -d);

    F.row(0) = Eigen::RowVector3i(0, 3, 2);
    F.row(1) = Eigen::RowVector3i(0, 2, 1);
    F.row(2) = Eigen::RowVector3i(2, 7, 1);
    F.row(3) = Eigen::RowVector3i(1, 7, 5);
    F.row(4) = Eigen::RowVector3i(2, 3, 6);
    F.row(5) = Eigen::RowVector3i(2, 6, 7);
    F.row(6) = Eigen::RowVector3i(0, 6, 3);
    F.row(7) = Eigen::RowVector3i(0, 4, 6);
    F.row(8) = Eigen::RowVector3i(4, 5, 7);
    F.row(9) = Eigen::RowVector3i(4, 7, 6);
    F.row(10) = Eigen::RowVector3i(4, 1, 5);
    F.row(11) = Eigen::RowVector3i(4, 0, 1);
}

bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers) {
    if (key == '1') {
        //viewer.data().clear();
        //viewer.data().set_mesh(V, F);
        // Add your code for computing vertex to face relations here;
        // store in VF,VFi.
        vertex_face_adjacency(V, F, VF);
        showidx(VF);
    }

    if (key == '2') {
        //viewer.data().clear();
        //viewer.data().set_mesh(V, F);
        // Add your code for computing vertex to vertex relations here:
        // store in VV.
        vertex_vertex_adjacency<true>(V, F, VV);
        showidx(VV);
    }

    if (key == '3') {
        //viewer.data().clear();
        //viewer.data().set_mesh(V, F);
        FN.setZero(F.rows(),3);
        // Add your code for computing per-face normals here: store in FN.
        per_face_normals(V, F, FN);
        cout << "FN10\t" << FN.row(10) << endl;
        // Set the viewer normals.
        viewer.data().set_normals(FN);
    }

    if (key == '4') {
        //viewer.data().clear();
        //viewer.data().set_mesh(V, F);
        // Add your code for computing per-vertex normals here: store in VN.
        face_areas(V, F, FArea);
        vertex_face_adjacency(V, F, VF);
        per_face_normals(V, F, FN);
        cout << "FN10\t" << FN.row(10) << endl;
        per_vertex_normals(V, VF, FN, FArea, VN);
        // Set the viewer normals.
        viewer.data().set_normals(VN);
    }

    if (key == '5') {
        //viewer.data().clear();
        //viewer.data().set_mesh(V, F);
        // Add your code for computing per-corner normals here: store in CN.
        vertex_face_adjacency(V, F, VF);
        per_face_normals(V, F, FN);
        per_corner_normals(V, F, VF, FN, degree_thresh, CN);
        //Set the viewer normals
        viewer.data().set_normals(CN);
    }

    if (key == '6') {
        //viewer.data().clear();
        //viewer.data().set_mesh(V, F);
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
    if (key == '8' || key == '9') {
        if (key == '8')
            degree_thresh+=10;
        else if (key == '9')
            degree_thresh-=10;
        cout << "degree_thresh " << degree_thresh << endl;
        callback_key_down(viewer, '5', modifiers);
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
    vfcalc = vvcalc = pfncalc = false;
    degree_thresh = 0.0;
    // Show the mesh
    Viewer viewer;
    viewer.callback_key_down = callback_key_down;
    std::string filename;
    if (argc == 2) {
        filename = std::string(argv[1]);
        if (!load_mesh(viewer, filename, V, F))
            return EXIT_FAILURE;
    }
    else {
        AA_Cube(1.0, V, F);
        cout << "V\n" << V << "\n\nF\n" << F << endl;
    }
    
    viewer.data().clear();
    viewer.data().set_mesh(V, F);

    callback_key_down(viewer, '3', 0);

    viewer.launch();
}
