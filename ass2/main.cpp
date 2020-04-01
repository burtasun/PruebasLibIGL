#include <array>

#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
/*** insert any necessary libigl headers here ***/
#include <igl/per_face_normals.h>
#include <igl/copyleft/marching_cubes.h>

#include <igl/jet.h>

using namespace std;
using Viewer = igl::opengl::glfw::Viewer;

// Input: imported points, #P x3
Eigen::MatrixXd P;

// Input: imported normals, #P x3
Eigen::MatrixXd N;

// Intermediate result: constrained points, #C x3
Eigen::MatrixXd constrained_points;

// Intermediate result: implicit function values at constrained points, #C x1
Eigen::VectorXd constrained_values;

// Parameter: degree of the polynomial
int polyDegree = 0;

// Parameter: Wendland weight function radius (make this relative to the size of the mesh)
double wendlandRadius = 0.1;

// Parameter: grid resolution
int resolution = 20;

// Intermediate result: grid points, at which the imlicit function will be evaluated, #G x3
Eigen::MatrixXd grid_points;

// Intermediate result: implicit function values at the grid points, #G x1
Eigen::VectorXd grid_values;

// Intermediate result: grid point colors, for display, #G x3
Eigen::MatrixXd grid_colors;

// Intermediate result: grid lines, for display, #L x6 (each row contains
// starting and ending point of line segment)
Eigen::MatrixXd grid_lines;

// Output: vertex array, #V x3
Eigen::MatrixXd V;

// Output: face array, #F x3
Eigen::MatrixXi F;

// Output: face normals of the reconstructed mesh, #F x3
Eigen::MatrixXd FN;

// Functions
void createGrid();
void evaluateImplicitFunc();
void getLines();
bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers);

// Creates a grid_points array for the simple sphere example. The points are
// stacked into a single matrix, ordered first in the x, then in the y and
// then in the z direction. If you find it necessary, replace this with your own
// function for creating the grid.
void createGrid() {
    grid_points.resize(0, 3);
    grid_colors.resize(0, 3);
    grid_lines. resize(0, 6);
    grid_values.resize(0);
    V. resize(0, 3);
    F. resize(0, 3);
    FN.resize(0, 3);

    // Grid bounds: axis-aligned bounding box
    Eigen::RowVector3d bb_min, bb_max;
    bb_min = P.colwise().minCoeff();
    bb_max = P.colwise().maxCoeff();

    // Bounding box dimensions
    Eigen::RowVector3d dim = bb_max - bb_min;

    // Grid spacing
    const double dx = dim[0] / (double)(resolution - 1);
    const double dy = dim[1] / (double)(resolution - 1);
    const double dz = dim[2] / (double)(resolution - 1);
    // 3D positions of the grid points -- see slides or marching_cubes.h for ordering
    grid_points.resize(resolution * resolution * resolution, 3);
    // Create each gridpoint
    for (unsigned int x = 0; x < resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                // Linear index of the point at (x,y,z)
                int index = x + resolution * (y + resolution * z);
                // 3D point at (x,y,z)
                grid_points.row(index) = bb_min + Eigen::RowVector3d(x * dx, y * dy, z * dz);
            }
        }
    }
}

// Function for explicitly evaluating the implicit function for a sphere of
// radius r centered at c : f(p) = ||p-c|| - r, where p = (x,y,z).
// This will NOT produce valid results for any mesh other than the given
// sphere.
// Replace this with your own function for evaluating the implicit function
// values at the grid points using MLS
void evaluateImplicitFunc() {
    // Sphere center
    auto bb_min = grid_points.colwise().minCoeff().eval();
    auto bb_max = grid_points.colwise().maxCoeff().eval();
    Eigen::RowVector3d center = 0.5 * (bb_min + bb_max);

    double radius = 0.5 * (bb_max - bb_min).minCoeff();

    // Scalar values of the grid points (the implicit function values)
    grid_values.resize(resolution * resolution * resolution);

    // Evaluate sphere's signed distance function at each gridpoint.
    for (unsigned int x = 0; x < resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                // Linear index of the point at (x,y,z)
                int index = x + resolution * (y + resolution * z);

                // Value at (x,y,z) = implicit function for the sphere
                grid_values[index] = (grid_points.row(index) - center).norm() - radius;
            }
        }
    }
}

// Code to display the grid lines given a grid structure of the given form.
// Assumes grid_points have been correctly assigned
// Replace with your own code for displaying lines if need be.
void getLines() {
    int nnodes = grid_points.rows();
    grid_lines.resize(3 * nnodes, 6);
    int numLines = 0;

    for (unsigned int x = 0; x<resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                int index = x + resolution * (y + resolution * z);
                if (x < resolution - 1) {
                    int index1 = (x + 1) + y * resolution + z * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
                if (y < resolution - 1) {
                    int index1 = x + (y + 1) * resolution + z * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
                if (z < resolution - 1) {
                    int index1 = x + y * resolution + (z + 1) * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
            }
        }
    }

    grid_lines.conservativeResize(numLines, Eigen::NoChange);
}

//[minxyz;maxxyz]
Eigen::Matrix<double,2,3> AABB(const Eigen::MatrixX3d& V) {
    Eigen::Matrix<double, 2, 3> BB;
    for (int i = 0; i < 3; i++) {
        BB(0, i) = V.block(0, i, V.rows(), 1).minCoeff();
        BB(1, i) = V.block(0, i, V.rows(), 1).maxCoeff();
    }
    return BB;
}


/*@TODO
    Query n closest
    Query n in radius
    */
struct grid3d {
public:
    vector<vector<int>> grid;
    double res;
    Eigen::RowVector3d min;//bb
    Eigen::RowVector3d max;//bb
    Eigen::RowVector3i bin;//nx ny nz
    inline bool inside(const Eigen::RowVector3d& p)const  {
        bool inside = true;
        for (int i = 0; i < 3; i++)
            inside &= (p[i] > min[i])& (p[i] < max[i]);
        return inside;
    }; 
    inline int atidx(int x, int y, int z) const {
        return x + y*bin[0] + z*bin[0]*bin[1];
    };
    const vector<int>& at(int i, int j, int k) const {
        return grid[atidx(i,j,k)];
    };
    inline Eigen::RowVector3i binpoint(const Eigen::RowVector3d& p) const {
        return Eigen::RowVector3i(((p - min) / res).cast<int>());
    }
    bool at(const Eigen::RowVector3d &p, vector<int>& binp) const {
        if (!inside(p))
            return false;
        auto pcomp = binpoint(p);
        binp = this->at(pcomp[0], pcomp[1], pcomp[2]);
        vector<int> v;
        return true;
    };
    //Assumes p inside grid
    void truncated_voxel(const Eigen::RowVector3d& p, const int voxel_dim, Eigen::RowVector3i& minvox, Eigen::RowVector3i& maxvox) const {
        auto binp = binpoint(p);
        for (int i = 0; i < 3; i++) {
            minvox[i] = ((binp[i] - voxel_dim) > 0) ? (binp[i] - voxel_dim) : 0;
            maxvox[i] = ((binp[i] + voxel_dim) < bin[i]) ? (binp[i] + voxel_dim) : bin[i] - 1;
        }
    }
};


/*@TODO
    Fine grained outside of grid search*/
template<bool aprox>
bool find_closest(
    const Eigen::MatrixX3d &P,
    const grid3d& grid,
    const Eigen::RowVector3d& p,
    int &idxclosest,
    double& distclosest
) {
    double mindist = numeric_limits<double>::max();

    const auto bruteforce_closest = [&](const Eigen::RowVector3d& pt)->int {
        int idx = -1;
        const auto m = P.rows();
        for (int i = 0; i < m; i++) {
            double dist = abs((pt - P.row(i)).norm());
            if (dist < mindist) {
                mindist = dist;//avoiding conditional jumps
                idx = i;
            }
        }
        return idx;
    };
    bool inside = grid.inside(p);
    if (!inside) {
        idxclosest = bruteforce_closest(p);
        distclosest = mindist;
        return idxclosest !=-1;
    }

    //Inside grid
    //returns closest idx, leaves unchanged otherwise
    const auto testbin = [&](const vector<int>& idx, const Eigen::RowVector3d& pt, int &id) {
        for (auto pijk : idx) {
            double dist(abs((P.row(pijk) - pt).norm()));
            if (dist < mindist) {
                mindist = dist;
                id = pijk;
            }
        }
    };
    
    if (aprox) {//assumes closest is in the same bin
        vector<int> binn;
        grid.at(p,binn);
        if (binn.size()) {
            testbin(binn, p, idxclosest);
            distclosest = mindist;
            return idxclosest != -1;
        }
    }
    //at least 1 neighbour in the vicinity to interrupt iteration by increasing bin voxel dimensions
    int idx = -1;
    int n = 1;//2n+1 voxel dimension
    Eigen::RowVector3i minvox, maxvox;
    while (n<10/*@todo define stop criteria not found*/) {
        grid.truncated_voxel(p, n, minvox, maxvox);
        for (int i = minvox[0]; i <= maxvox[0]; i++) {
            for (int j = minvox[1]; j <= maxvox[1]; j++) {
                for (int k = minvox[2]; k <= maxvox[2]; k++) {
                    if (grid.at(i, j, k).size())
                        testbin(grid.at(i, j, k), p, idx);
                }//k
            }//j
        }//i
        if (idx != -1)
            break;
        n++;
    }
    idxclosest = idx;
    distclosest = mindist;    
    return idxclosest != -1;
}


//find_n_radius
/*@TODO
    outside radius search*/
template<bool sorted = false>
bool find_n_radius(
    const Eigen::MatrixX3d& V,
    const grid3d& grid,
    const Eigen::RowVector3d& p,
    const double radius,
    vector<int>& idx
) {
    idx.clear();

    if (!grid.inside(p))//Not implemented
        return false;

    std::map<int, double> idx_dists;

    //coarse search using Manhattan distance
    //number of bins width at grid's resolution
    auto voxel_dim = static_cast<int>(ceil((2.0 * radius / grid.res)));
    auto voxeldimsq(voxel_dim * voxel_dim), voxeldimminsq((voxel_dim - 1) * (voxel_dim - 1));
    Eigen::RowVector3i minvox, maxvox;
    grid.truncated_voxel(p, voxel_dim, minvox, maxvox);
    Eigen::RowVector3i binp(grid.binpoint(p));
    for (int i = minvox[0]; i <= maxvox[0]; i++) {
        for (int j = minvox[1]; j <= maxvox[1]; j++) {
            for (int k = minvox[2]; k <= maxvox[2]; k++) {
                auto& bin(grid.at(i, j, k));
                if (!bin.size()) continue;
                
                int bindistsq = ((i - binp[0]) * (i - binp[0]) + (j - binp[1]) * (j - binp[1]) + (k - binp[2]) * (k - binp[2]));
                
                if (bindistsq > voxeldimsq)//outside of radius
                    continue;

                if (bindistsq < voxeldimminsq) {//inside radius
                    idx.insert(idx.end(), bin.begin(), bin.end());
                    continue;
                }
                for (auto id : bin) {
                //    if (sorted) {//compile time branch
                //        double dist = (V.row(id) - p).norm();
                //        if (dist <= radius)
                //            idx_dists[id] = dist;
                //    }
                //    else {
                        if ((V.row(id) - p).norm() < radius)//Discard real distance
                            idx.push_back(id);
                    }
                
            }//k
        }//j
    }//i
    if (sorted) {
        for (auto id : idx_dists)
            idx.push_back(id.first);
    }
}


//Builds a 3d grid with the vertex idx in each node
template<bool saveVgridIdx> //compile time branch
void spatial_indexer_3dgrid(
    const Eigen::MatrixX3d& V,
    const double factor,
    grid3d &grid,
    Eigen::MatrixX2i &VgridIdx
) {
    //determine the resolution
    auto bb = AABB(V);
    Eigen::RowVector3d bblength(bb.block<1, 3>(1, 0) - bb.block<1, 3>(0, 0));
    double resolution = bblength.minCoeff() / factor;
    grid.res = resolution;
    //Grid formatting
    //  inner => X / middle => Y / outter => Z
    for (int i = 0; i < 3; i++) {
        grid.bin[i] = std::ceil(bblength[i] / grid.res) + 1;
        grid.min[i] = bb(0, i);
        grid.max[i] = bb(1, i);
    }
    grid.grid.resize(grid.bin[0] * grid.bin[1] * grid.bin[2] );
    //for (auto& bin : grid.grid)
    //    bin.reserve(20);

    const double xmin = bb(0, 0), ymin = bb(0, 1), zmin = bb(0, 2);
    const int nx = grid.bin[0], ny = grid.bin[1], nz = grid.bin[2];

    const Eigen::RowVector3d min(bb.block<1, 3>(0, 0));
    Eigen::MatrixX3i Vidx = ((V - min.replicate(V.rows(), 1)) / grid.res).cast<int>();
    if (saveVgridIdx)
        VgridIdx.setZero(V.rows(),2);
    const auto m = V.rows();
    for (int i = 0; i < m; i++) {
        int idx = Vidx(i, 0) + Vidx(i, 1)*nx + Vidx(i, 2)*nx*ny ;
        grid.grid[idx].push_back(i);
        if (saveVgridIdx) {
            VgridIdx(i, 0) = idx;
            VgridIdx(i, 1) = grid.grid[idx].size() - 1;
        }
    }
#ifdef DEBUG
    cout << "print grid\n";
    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                cout << endl << x << ',' << y << ',' << z << '\t';
                for (auto elem : grid.grid[x + y*nx + z*nx*ny])
                    cout << elem << '\t';
            }
        }
    }
#endif // DEBUG
    //for (auto& bin : grid.grid)
    //    bin.shrink_to_fit();
}


void add_constraints(
    const Eigen::MatrixX3d& V,
    const Eigen::MatrixX3d& N,
    Eigen::MatrixX3d& Vconstr,
    Eigen::MatrixX3d& N_norm
) {
    assert(V.rows() == N.rows());

    const auto m = V.rows();
    

    N_norm = N.rowwise().normalized();

    auto bb = AABB(V);
    double Epsilon = 0.01 * abs((bb.block<1, 3>(0, 0) - bb.block<1, 3>(1, 0)).norm());//Epsilon 1/100th of bb diagonal

    //Epsilon fine tuning
    //  Assuming grid res > Epsilon => closest only in the same bin
    Eigen::MatrixX3d Vconstrloc;
    Vconstrloc.resize(m * 3, 3);

    double mindist = numeric_limits<double>::max();
    bool keepgoing = true;
    double average_points_per_bin = 5;
    grid3d grid;
    int factor_subdivision_grid = 2;
    while(true)
    {
        grid = grid3d();
        Eigen::MatrixX2i VgridIdx;
        spatial_indexer_3dgrid<false>(V, factor_subdivision_grid, grid, VgridIdx);
        int cnt_noempty = 0, cnt = 0;
        for (auto& bin : grid.grid) {
            if (bin.size()) { cnt_noempty++; cnt += bin.size(); }
        }
        if (((double)cnt / (double)cnt_noempty) < average_points_per_bin)
            break;
#ifdef DEBUG
        cout << "factor_subdivision_grid\t" << factor_subdivision_grid << "\t\tcnt_noempty\t" << cnt_noempty << "\t\tcnt\t" << cnt << endl;
#endif // DEBUG
        factor_subdivision_grid++;        
    }
    
    std::cout << "Epsilon " << Epsilon << endl;
    Vconstrloc.block(0, 0, m, 3) = V;
    const auto iterclosest = [&](int offset)
    {
        while (keepgoing)
        {
            Vconstrloc.block(m, 0, m, 3) = V + N_norm * Epsilon;
            Vconstrloc.block(2 * m, 0, m, 3) = V - N_norm * Epsilon;
            double distclosest; int idxclosest = -1;
            keepgoing = false;
            for (int i = 0; i < m; i++) {
                find_closest<false>(V, grid, Vconstrloc.row(offset + i), idxclosest, distclosest);
                if (idxclosest != i) {
                    Epsilon /= 2; keepgoing = true;
                    break;
                }
            }
        }
    };
    iterclosest(m);//+epsilon*N
    keepgoing = true;
    iterclosest(2 * m);//-epsilon*N
    Vconstr = std::move(Vconstrloc);
    std::cout << "EpsilonEnd " << Epsilon << endl;
    
}


bool callback_key_down(Viewer &viewer, unsigned char key, int modifiers) {
    if (key == '1') {
        // Show imported points
        viewer.data().clear();
        viewer.core.align_camera_center(P);
        viewer.data().point_size = 11;
        viewer.data().add_points(P, Eigen::RowVector3d(0,0,0));
    }

    if (key == '2') {
        // Add your code for computing auxiliary constraint points here
        Eigen::MatrixX3d Nnorm, Pconstr;
        add_constraints(P, N, Pconstr, Nnorm);
        // Add code for displaying all points, as above
        viewer.data().clear();
        viewer.core.align_camera_center(Pconstr);
        viewer.data().point_size = 11;

        Eigen::MatrixXd C_rgb; 
        C_rgb.resize(Pconstr.rows(), 3);
        C_rgb.block(0, 0, P.rows(), 3) = Eigen::RowVector3d(1, 0, 0).replicate(P.rows(), 1);
        C_rgb.block(P.rows(), 0, P.rows(), 3) = Eigen::RowVector3d(0, 1, 0).replicate(P.rows(), 1);
        C_rgb.block(P.rows()*2, 0, P.rows(), 3) = Eigen::RowVector3d(0, 0, 1).replicate(P.rows(), 1);
        viewer.data().add_points(Pconstr, C_rgb);
    }

    if (key == '3') {
        // Show grid points with colored nodes and connected with lines
        viewer.data().clear();
        viewer.core.align_camera_center(P);
        // Add code for creating a grid
        // Add your code for evaluating the implicit function at the grid points
        // Add code for displaying points and lines
        // You can use the following example:
        grid3d grid;
        {
            Eigen::MatrixX2i VgridIdx;
            spatial_indexer_3dgrid<false>(P, 4, grid, VgridIdx);
        }
        {
            const auto bruteforce_radius = [&](const Eigen::RowVector3d& pt, const double radius, vector<int> &idx) {
                const auto m = P.rows();
                for (int i = 0; i < m; i++) {
                    double dist = abs((pt - P.row(i)).norm());
                    if (dist < radius)
                        idx.push_back(i);
                }
            };
            /*DEBUG 3D GRID*/
            auto bb = AABB(P); 
            Eigen::RowVector3d dims(bb.row(1) - bb.row(0));
            
            constexpr int n = 100000;
            Eigen::Matrix<double, -1, 3> pts_test; pts_test.resize(n, 3);
            for (int i = 0; i < n; i++) {
                pts_test.row(i) = Eigen::RowVector3d((double)rand() / (double)RAND_MAX, (double)rand() / (double)RAND_MAX, (double)rand() / (double)RAND_MAX);
                pts_test.row(i) = (pts_test.row(i).array() * dims.array()).matrix() + bb.row(0);
            }
            cout << "ini\n";
            for (int i = 0; i < n; i++) {
                vector<int> idx;
                find_n_radius<false>(P, grid, Eigen::RowVector3d(pts_test.row(i)), 0.25, idx);
            }
            cout << "bruteforce\n";
            for (int i = 0; i < n; i++) {
                vector<int> idx;
                bruteforce_radius(Eigen::RowVector3d(pts_test.row(i)), 0.25, idx);
            }
            cout << "end\n";
            //    //cout << "find_n_radius\tpt" << i << "\t" << pts_test.row(i) << endl << "\tn = " << idx.size() << endl << endl;
            //    idx.clear();
            //    bruteforce_radius(Eigen::RowVector3d(pts_test.row(i)), 0.25, idx);
            //    cout << "bruteforce_radius\tpt" << i << "\t" << pts_test.row(i) << endl << "\tn = " << idx.size() << endl << endl;
            //}
        }
        /*** begin: sphere example, replace (at least partially) with your code ***/
        // Make grid
        createGrid();

        // Evaluate implicit function
        evaluateImplicitFunc();

        // get grid lines
        getLines();

        // Code for coloring and displaying the grid points and lines
        // Assumes that grid_values and grid_points have been correctly assigned.
        grid_colors.setZero(grid_points.rows(), 3);

        // Build color map
        for (int i = 0; i < grid_points.rows(); ++i) {
            double value = grid_values(i);
            if (value < 0) {
                grid_colors(i, 1) = 1;
            }
            else {
                if (value > 0)
                    grid_colors(i, 0) = 1;
            }
        }

        // Draw lines and points
        viewer.data().point_size = 8;
        viewer.data().add_points(grid_points, grid_colors);
        viewer.data().add_edges(grid_lines.block(0, 0, grid_lines.rows(), 3),
                              grid_lines.block(0, 3, grid_lines.rows(), 3),
                              Eigen::RowVector3d(0.8, 0.8, 0.8));
        /*** end: sphere example ***/
    }

    if (key == '4') {
        // Show reconstructed mesh
        viewer.data().clear();
        // Code for computing the mesh (V,F) from grid_points and grid_values
        if ((grid_points.rows() == 0) || (grid_values.rows() == 0)) {
            cerr << "Not enough data for Marching Cubes !" << endl;
            return true;
        }
        // Run marching cubes
        igl::copyleft::marching_cubes(grid_values, grid_points, resolution, resolution, resolution, V, F);
        if (V.rows() == 0) {
            cerr << "Marching Cubes failed!" << endl;
            return true;
        }

        igl::per_face_normals(V, F, FN);
        viewer.data().set_mesh(V, F);
        viewer.data().show_lines = true;
        viewer.data().show_faces = true;
        viewer.data().set_normals(FN);
    }

    return true;
}

bool callback_load_mesh(Viewer& viewer,string filename)
{
  igl::readOFF(filename,P,F,N);
  callback_key_down(viewer,'1',0);
  return true;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
      std::cout << "Usage ex2_bin <mesh.off>" << endl;
      igl::readOFF("../data/sphere.off",P,F,N);
    }
	  else
	  {
		  // Read points and normals
		  igl::readOFF(argv[1],P,F,N);
	  }

     Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    viewer.callback_key_down = callback_key_down;

    menu.callback_draw_viewer_menu = [&]()
    {
      // Draw parent menu content
      menu.draw_viewer_menu();

      // Add new group
      if (ImGui::CollapsingHeader("Reconstruction Options", ImGuiTreeNodeFlags_DefaultOpen))
      {
        // Expose variable directly ...
        ImGui::InputInt("Resolution", &resolution, 0, 0);
        if (ImGui::Button("Reset Grid", ImVec2(-1,0)))
        {
          std::cout << "ResetGrid\n";
          // Recreate the grid
          createGrid();
          // Switch view to show the grid
          callback_key_down(viewer,'3',0);
        }

        // TODO: Add more parameters to tweak here...
      }

    };

    callback_key_down(viewer, '1', 0);

    viewer.launch();
}
