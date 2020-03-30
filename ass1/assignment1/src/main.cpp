#include <iostream>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/jet.h>
/*** insert any libigl headers here ***/

# define M_PI 3.14159265358979323846
#define M_PI2 2*M_PI
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
    std::vector<std::vector<int>>& faces_in_vertex,/*[vertex][faces]*/
    std::vector<std::vector<int>>& vertexidx_faces_of_vertex/*[vertex][vertexidx_of_face]*/)
{
    if (vfcalc) return;
    faces_in_vertex.resize(V.rows());
    vertexidx_faces_of_vertex.resize(V.rows());
    for (int i = 0; i < F.rows(); i++) {
        for (int j = 0; j < 3; j++) {
            faces_in_vertex[F(i, j)].push_back(i);
            vertexidx_faces_of_vertex[F(i, j)].push_back(j);
        }
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
    //if (vvcalc)return;
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
    //vvcalc = true;
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
    //if (pfncalc) return;
    PFN.resize(F.rows(), 3);
    for (int i = 0; i < F.rows(); i++)
        PFN.row(i) =
            (V.row(F(i, 1)) - V.row(F(i, 0))).cross(/*v1*/
                (V.row(F(i, 2)) - V.row(F(i, 0)))/*v2*/
            ).normalized();
    //pfncalc = true;
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

void face_neighbours(
    const Eigen::Matrix<int, -1, 3>& F,
    const vector<vector<int>>& VF,
    vector<vector<int>>& Fnn
) {
    const auto m = F.rows();
    Fnn.resize(m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < F.cols(); j++) {
            for (int k = 0; k < VF[F(i, j)].size(); k++) {
                auto& f = VF[F(i, j)][k];
                if (f == i) continue;
                Fnn[i].push_back(f);
            }
        }
    }
    for (auto& fs : Fnn) {
        sort(fs.begin(), fs.end());
        fs.erase(unique(fs.begin(), fs.end()), fs.end());
    }
}

void face_barycenters(
    const Eigen::Matrix<double, -1, 3>& V,
    const Eigen::Matrix<int, -1, 3>& F,
    Eigen::Matrix<double, -1, 3>& V_m
) {
    const auto m = F.rows();
    V_m.setZero(m, 3);
    for (int i = 0; i < m; i++)
        V_m.row(i) = (V.row(F(i, 0)) + V.row(F(i, 1)) + V.row(F(i, 2))) / 3;
}

void barycenter_faces(
    const Eigen::Matrix<double, -1, 3>& V,
    const Eigen::Matrix<int, -1, 3>& F,
    Eigen::Matrix<int, -1, 3>& F_m/*F_m dim = 3 * F dim */
) {
    /*
        F       =>  0 - m-1      //      V          (n)
        F_m     =>  0 - 3m-1     //      V & V_m    (n+m)
    */
    const auto m = F.rows();
    const auto n = V.rows();
    F_m.setZero(3 * m, 3);
    for (int i = 0; i < m; i++) //012 => 01x 12x 20x // x = n + i        
        F_m.block<3, 3>(i * 3, 0) <<
            F(i, 0), F(i, 1), (n + i),
            F(i, 1), F(i, 2), (n + i),
            F(i, 2), F(i, 0), (n + i);
}

void averaged_positions(
    const Eigen::Matrix<double, -1, 3>& V,
    const vector<vector<int>>& VV,
    Eigen::Matrix<double, -1, 3>& P
) {
    const auto m = V.rows();
    P.setZero(m, 3);
    double an = 0.0;
    for (int i = 0; i < m; i++) {
        auto nv = VV[i].size();
        an = (4.0 - 2.0 * cos(M_PI2 / static_cast<double>(nv))) / 9.0;
        for (int j = 0; j < nv; j++)
            P.row(i) += V.row(VV[i][j]);
        P.row(i) = (1 - an) * V.row(i) + (an/nv) * P.row(i);
    }
}

/*Subdivision_sqrt3*/
void Subdivision_sqrt3(
    const Eigen::Matrix<double, -1, 3> &V,
    const Eigen::Matrix<int, -1, 3> &F,
    const vector<vector<int>> &VV,
    Eigen::Matrix<double,-1,3> &VSubdiv,
    Eigen::Matrix<int, -1, 3>& FSubdiv
) {
    Eigen::Matrix<double, -1, 3> V_m;
    face_barycenters(V, F, V_m);
    Eigen::Matrix<int, -1, 3> F_prim_prim;
    barycenter_faces(V, F, F_prim_prim);
    cout << "-------------F-------------\n" << F << "\n--------------------------\n--------------------------\n";
    cout << "-------------F_prim-------------\n" << F_prim_prim << "\n--------------------------\n--------------------------\n";
    Eigen::Matrix<double, -1, 3> P;
    averaged_positions(V, VV, P);    
    Eigen::Matrix<double, -1, 3> V_prim;//V=[P,V_m]
    V_prim.resize(P.rows() + V_m.rows(), 3);
    V_prim.block(0, 0, P.rows(), 3) = P;
    V_prim.block(P.rows(), 0, V_m.rows(), 3) = V_m;
    cout << "-------------V-------------\n" << V << "\n--------------------------\n--------------------------\n";
    cout << "-------------P-------------\n" << P << "\n--------------------------\n--------------------------\n";
    cout << "-------------V_m-------------\n" << V_m << "\n--------------------------\n--------------------------\n";
    cout << "-------------V_prim-------------\n" << V_prim << "\n--------------------------\n--------------------------\n";







    FSubdiv = F_prim_prim;
    VSubdiv = V_prim;
}

void ConnectedComponentsFaces(
    const Eigen::Matrix<int, -1, 3>& F,
    const vector<vector<int>>& Fnn,
    Eigen::VectorXi& Fid,
    vector<int>& cnt_id
) {
    const auto m = F.rows();
    Fid.setZero(m);
    vector<bool> checked(m, false);
    cnt_id.clear();
    vector<int> q;
    int id_curr = -1;

    for (int i = 0; i < m; i++) {
        if (checked[i]) continue;
        id_curr++;
        cnt_id.push_back(0);
        q.push_back(i);
        while(!q.empty()){
            auto f_curr = q.front(); q.erase(q.begin());
            if (checked[f_curr]) continue;
            for (int j = 0; j < Fnn[f_curr].size(); j++) {
                auto& fnn = Fnn[f_curr][j];
                if (checked[fnn]) continue;
                q.push_back(fnn);
            }
            checked[f_curr] = true;
            Fid[f_curr] = id_curr;
            cnt_id[id_curr]++;
        }
    }
    int cnt_tot = 0;
    for (auto& cnt : cnt_id)
        cnt_tot += cnt;
    assert(cnt_tot == m);
    assert(cnt_id.size() == id_curr);
}


bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers) {
    if (key == '1') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        // Add your code for computing vertex to face relations here;
        // store in VF,VFi.
        vertex_face_adjacency(V, F, VF, VFi);
        showidx(VF);
    }

    if (key == '2') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        // Add your code for computing vertex to vertex relations here:
        // store in VV.
        vertex_vertex_adjacency<true>(V, F, VV);
        showidx(VV);
    }

    if (key == '3') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        FN.setZero(F.rows(),3);
        // Add your code for computing per-face normals here: store in FN.
        per_face_normals(V, F, FN);
        cout << "FN10\t" << FN.row(10) << endl;
        // Set the viewer normals.
        viewer.data().set_normals(FN);
    }

    if (key == '4') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        // Add your code for computing per-vertex normals here: store in VN.
        face_areas(V, F, FArea);
        vertex_face_adjacency(V, F, VF, VFi);
        per_face_normals(V, F, FN);
        cout << "FN10\t" << FN.row(10) << endl;
        per_vertex_normals(V, VF, FN, FArea, VN);
        // Set the viewer normals.
        viewer.data().set_normals(VN);
    }

    if (key == '5') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        // Add your code for computing per-corner normals here: store in CN.
        vertex_face_adjacency(V, F, VF, VFi);
        per_face_normals(V, F, FN);
        per_corner_normals(V, F, VF, FN, degree_thresh, CN);
        //Set the viewer normals
        viewer.data().set_normals(CN);
    }

    if (key == '6') {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        component_colors_per_face.setZero(F.rows(),3);
        // Add your code for computing per-face connected components here:
        // store the component labels in cid.
        vector<vector<int>> Fnn;
        vertex_face_adjacency(V, F, VF, VFi);
        face_neighbours(F, VF, Fnn);
        Eigen::VectorXi Fid;
        vector<int> cnt_id;
        ConnectedComponentsFaces(F, Fnn, Fid, cnt_id);
        cout << "Cnt_id\t" << cnt_id.size() << endl;
        for (auto& cnt : cnt_id) cout << '\t' << cnt << endl;

        // Compute colors for the faces based on components, storing them in
        // component_colors_per_face.
        igl::jet(Fid, true, component_colors_per_face);
        // Set the viewer colors
        viewer.data().set_colors(component_colors_per_face);
    }

    if (key == '7') {
        // Add your code for sqrt(3) subdivision here.
        vertex_vertex_adjacency<true>(V, F, VV);
        Eigen::MatrixX3d V_subdiv; Eigen::MatrixX3i F_subdiv;
        Subdivision_sqrt3(V, F, VV, V_subdiv, F_subdiv);
        // Set up the viewer to display the new mesh
        V = V_subdiv; F = F_subdiv;
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
