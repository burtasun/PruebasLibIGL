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

#include <Eigen/Eigenvalues> 

using namespace std;
using Viewer = igl::opengl::glfw::Viewer;
#ifdef DEBUG
#define dbg(var) cout<<endl<<#var<<"--------\n"<<var<<endl;
#else
#define dbg(var) ;
#endif

// Input: imported points, #P x3
Eigen::MatrixXd P;

// Input: imported normals, #P x3
Eigen::MatrixXd N;

// Intermediate result: constrained points, #C x3
Eigen::MatrixXd constrained_points;

// Intermediate result: implicit function values at constrained points, #C x1
Eigen::VectorXd constrained_values;

//Epsilon val
double EpsilonConstr = 0.0;

// Parameter: degree of the polynomial
int polyDegree = 0;
int weighfunc= 0;

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

void getLines();
bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers);


/*@TODO
    Query n closest
    Query n in radius
    */
struct grid3d {
public:
    vector<vector<int>> grid;
    vector<int> binSizes;
    double res;
    Eigen::RowVector3d min;//bb
    Eigen::RowVector3d max;//bb
    Eigen::RowVector3i bin;//nx ny nz
    inline bool inside(const Eigen::RowVector3d& p)const {
        bool inside = true;
        for (int i = 0; i < 3; i++)
            inside &= (p[i] > min[i])& (p[i] < max[i]);
        return inside;
    };
    inline int atidx(int x, int y, int z) const {
        return x + y * bin[0] + z * bin[0] * bin[1];
    };
    inline const vector<int>& at(int i, int j, int k) const {
        return grid[atidx(i, j, k)];
    };
    inline const int sizeAt(int i, int j, int k) const {
        return binSizes[atidx(i, j, k)]; 
    }
    inline Eigen::RowVector3i binpoint(const Eigen::RowVector3d& p) const {
        return Eigen::RowVector3i(((p - min) / res).cast<int>());
    }
    bool at(const Eigen::RowVector3d& p, vector<int>& binp) const {
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

bool find_n_radius(
    const Eigen::MatrixX3d& pts,
    const grid3d& grid,
    const Eigen::RowVector3d& p,
    const double radius,
    vector<int>& idx 
);

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
    //Increased size to cover points closer to BB limits
    bb_min -= 1*(Eigen::RowVector3d(dx, dy, dz));
    bb_max += 1*(Eigen::RowVector3d(dx, dy, dz));
    const double res(resolution + 2);
    resolution += 2;
    // 3D positions of the grid points -- see slides or marching_cubes.h for ordering
    grid_points.resize(res * res * res, 3);
    // Create each gridpoint
    for (unsigned int x = 0; x < res; ++x) {
        for (unsigned int y = 0; y < res; ++y) {
            for (unsigned int z = 0; z < res; ++z) {
                // Linear index of the point at (x,y,z)
                int index = x + res * (y + res * z);
                // 3D point at (x,y,z)
                grid_points.row(index) = bb_min + Eigen::RowVector3d(x * dx, y * dy, z * dz);
            }
        }
    }
}

void VisGrid3D(Viewer& viewer) 
{
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
};


void DrawBB(Viewer& viewer, Eigen::Matrix<double, 2, 3>& bb) {
    //cube mesh
    /*
    |z -x /y
       7 --- 8
      /|    /|
    4 -+- 3  |
    |  5--|--6
    | /   | /
    1 --- 2

    1m / 8M

    */
    Eigen::RowVector3d m(bb.row(0));
    Eigen::RowVector3d M(bb.row(1));
    Eigen::MatrixX3d v(8, 3);
    v.row(0) = m;
    v.row(1) = Eigen::RowVector3d(M[0], m[1], m[2]);
    v.row(2) = Eigen::RowVector3d(M[0], m[1], M[2]);
    v.row(3) = Eigen::RowVector3d(m[0], m[1], M[2]);
    v.row(4) = Eigen::RowVector3d(m[0], M[1], m[2]);
    v.row(5) = Eigen::RowVector3d(M[0], M[1], m[2]);
    v.row(6) = Eigen::RowVector3d(m[0], M[1], M[2]);
    v.row(7) = Eigen::RowVector3d(M[0], M[1], M[2]);
    Eigen::MatrixXd bblines(12, 6);//line row / each row start/end point
    bblines.block<1, 3>(0, 0) = v.row(0); bblines.block<1, 3>(0, 3) = v.row(1);
    bblines.block<1, 3>(1, 0) = v.row(0); bblines.block<1, 3>(1, 3) = v.row(3);
    bblines.block<1, 3>(2, 0) = v.row(0); bblines.block<1, 3>(2, 3) = v.row(4);
    bblines.block<1, 3>(3, 0) = v.row(1); bblines.block<1, 3>(3, 3) = v.row(2);
    bblines.block<1, 3>(4, 0) = v.row(1); bblines.block<1, 3>(4, 3) = v.row(5);
    bblines.block<1, 3>(5, 0) = v.row(5); bblines.block<1, 3>(5, 3) = v.row(4);
    bblines.block<1, 3>(6, 0) = v.row(5); bblines.block<1, 3>(6, 3) = v.row(7);
    bblines.block<1, 3>(7, 0) = v.row(4); bblines.block<1, 3>(7, 3) = v.row(6);
    bblines.block<1, 3>(8, 0) = v.row(6); bblines.block<1, 3>(8, 3) = v.row(3);
    bblines.block<1, 3>(9, 0) = v.row(6); bblines.block<1, 3>(9, 3) = v.row(7);
    bblines.block<1, 3>(10, 0) = v.row(2); bblines.block<1, 3>(10, 3) = v.row(3);
    bblines.block<1, 3>(11, 0) = v.row(2); bblines.block<1, 3>(11, 3) = v.row(7);

    viewer.data().add_edges(
        bblines.block(0, 0, bblines.rows(), 3),
        bblines.block(0, 3, bblines.rows(), 3),
        Eigen::RowVector3d(0.8, 0.8, 0.8)//color
    );
}


// Function for explicitly evaluating the implicit function for a sphere of
// radius r centered at c : f(p) = ||p-c|| - r, where p = (x,y,z).
// This will NOT produce valid results for any mesh other than the given
// sphere.
// Replace this with your own function for evaluating the implicit function
// values at the grid points using MLS

//@TODO
/*
    compile time parametrization k degree
*/
template<int polyaprox, typename weighfunclambda>
void evaluateImplicitFunc(
    const Eigen::MatrixX3d& Pconstr,
    const Eigen::VectorXd& Vconstr,
    const grid3d &grid,
    const double radius,
    const weighfunclambda WeightFunc
) {
    /*deg0*/
    const auto evalPoint = [&](
        const Eigen::RowVector3d& p,
        const vector<int>& idx
        ) -> double
    {

        //centers -> ci & weights -> w(||p-ci||)
        /*A = [ci 1]*/
        Eigen::Matrix<double, -1, 1/*value->1*/> A;
        A.setOnes(idx.size());
        /*w*/
        Eigen::VectorXd W; W.resize(idx.size());
        /*d*/
        Eigen::VectorXd B; B.resize(idx.size());

        for (int i = 0; i < idx.size(); i++) {
            B[i] = Vconstr[idx[i]];
            W[i] = WeightFunc((Pconstr.row(idx[i]) - p).norm(), radius);
        }
        dbg(radius); dbg(p); dbg(B); dbg(A); dbg(W);

        A = W.asDiagonal() * A; //[wi*ci]
        B = W.asDiagonal() * B; //[wi*di]

        dbg(A); dbg(B);

        //a1 - a4
        Eigen::Matrix<double, 1, 1> sol = A.householderQr().solve(B);
        //pxa1+...pza3+a4
        double ret = sol[0];// p.dot(sol.head(3)) + sol[3];
        dbg(sol); dbg(ret);
        return ret;
    };
    /*deg1*/
    const auto evalPointPlane = [&](
        const Eigen::RowVector3d& p,
        const vector<int> &idx
        ) -> double
    {
        //centers -> ci & weights -> w(||p-ci||)
        if (idx.size() < 20)
            return evalPoint(p, idx);
        /*A = [ci 1]*/
        Eigen::Matrix<double, -1, 4/*plane->4*/> A; A.resize(idx.size(), Eigen::NoChange);
        /*w*/
        Eigen::VectorXd W; W.resize(idx.size());
        /*d*/
        Eigen::VectorXd B; B.resize(idx.size());
        
        for (int i = 0; i < idx.size(); i++) {
            B[i] = Vconstr[idx[i]];
            A.row(i).head(3) = Pconstr.row(idx[i]);
            A(i,3) = 1;
            W[i] = WeightFunc((Pconstr.row(idx[i]) - p).norm(), radius);
        }
        dbg(radius); dbg(p); dbg(B); dbg(A); dbg(W);
        
        A = W.asDiagonal() * A; //[wi*ci]
        B = W.asDiagonal() * B; //[wi*di]

        dbg(A); dbg(B);

        //a1 - a4
        Eigen::Matrix<double,4,1> sol = A.colPivHouseholderQr().solve(B);
        //pxa1+...pza3+a4
        double ret = p.dot(sol.head(3)) + sol[3];
        //if (abs(ret) < 1) {
        //    dbg1(sol); dbg1(ret);
        //    dbg1(radius); dbg1(p); dbg1(B); dbg1(A); dbg1(W);
        //}
        return ret;
    };
    /*deg2*/
    const auto evalPointCuadratic = [&](
        const Eigen::RowVector3d& p,
        const vector<int>& idx
        ) -> double
    {

        //centers -> ci & weights -> w(||p-ci||)
        if (idx.size() < 20)
            return evalPoint(p, idx);
        /*A = [ci 1]*/
        Eigen::Matrix<double, -1, 10/*value->1*/> A;
        A.resize(idx.size(), Eigen::NoChange);
        A.block(0, 0, A.rows(), 1).setConstant(1);
        /*w*/
        Eigen::VectorXd W; W.resize(idx.size());
        /*d*/
        Eigen::VectorXd B; B.resize(idx.size());

        for (int i = 0; i < idx.size(); i++) {
            auto& pt(Pconstr.row(idx[i]));
            double tmp[] = { 1, pt[0], pt[1], pt[2], /*1 x y z*/
                pt[0] * pt[1], pt[0] * pt[2], pt[1] * pt[2], /*xy xz yz*/
                pt[0] * pt[0], pt[1] * pt[1], pt[2] * pt[2] }; /*x2 y2 z2*/
            for (int j = 0; j < 10; j++)
                A(i, j) = tmp[j];
            B[i] = Vconstr[idx[i]];
            W[i] = WeightFunc((Pconstr.row(idx[i]) - p).norm(), radius);
        }
        dbg(radius); dbg(p); dbg(B); dbg(A); dbg(W);

        A = W.asDiagonal() * A; //[wi*ci]
        B = W.asDiagonal() * B; //[wi*di]

        dbg(A); dbg(B);

        //a1 - a4
        Eigen::Matrix<double, 10, 1> sol = A.colPivHouseholderQr().solve(B);
        //pxa1+...pza3+a4
        double ret = 0.0;
        {
            auto& pt(p);
            double tmp[] = { 1, pt[0], pt[1], pt[2], /*1 x y z*/
                pt[0] * pt[1], pt[0] * pt[2], pt[1] * pt[2], /*xy xz yz*/
                pt[0] * pt[0], pt[1] * pt[1], pt[2] * pt[2] };/*x2 y2 z2*/
            for (int i = 0; i < 10; i++)
                ret += tmp[i] * sol[i];
        }
        dbg(sol); dbg(ret);
        return ret;
    };



    grid_values.resize(grid_points.rows());
    // Evaluate sphere's signed distance function at each gridpoint.
    const auto loopgrid = [&](const auto evalpt_lambda)
    {
        const int idxMax = pow(resolution, 3);
#pragma omp parallel
        {
            vector<int> idx;
#pragma omp parallel for if(idxMax>1000)
            for (int index = 0; index < idxMax; index++) {
                // Linear index of the point at (x,y,z)
                //MLS eval
                if (!find_n_radius(Pconstr, grid, grid_points.row(index), radius, idx))
                    grid_values[index] = std::nan(""); //flag void points
                else
                    grid_values[index] = /*evalPoint*/evalpt_lambda(grid_points.row(index), idx);
            }
        }
    };

    switch (polyaprox)//compile time branches
    {
    case 0:
        loopgrid(evalPoint);
        break;
    case 1:
        loopgrid(evalPointPlane);
        break;
    case 2:
        loopgrid(evalPointCuadratic);
        break;
    default:
        loopgrid(evalPoint);
        break;
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

//Object Oriented Bounding Box
//[minxyz;maxxyz]
Eigen::Matrix<double, 2, 3> OOBB(const Eigen::MatrixX3d& V, const Eigen::MatrixX3d& N, Eigen::MatrixX3d& Vrot, Eigen::MatrixX3d& Nrot) {
    Eigen::RowVector3d centroid = V.colwise().sum() / V.rows();
    Eigen::MatrixX3d Vmean = V - centroid.replicate(V.rows(), 1);//xi-ux,yi-uy,zi-uz
    Eigen::Matrix3d Covariance;
    Covariance <<
        (Vmean.col(0).array()* Vmean.col(0).array()).sum(), (Vmean.col(0).array()* Vmean.col(1).array()).sum(), (Vmean.col(0).array()* Vmean.col(2).array()).sum(),
        (Vmean.col(1).array()* Vmean.col(0).array()).sum(), (Vmean.col(1).array()* Vmean.col(1).array()).sum(), (Vmean.col(1).array()* Vmean.col(2).array()).sum(),
        (Vmean.col(2).array()* Vmean.col(0).array()).sum(), (Vmean.col(2).array()* Vmean.col(1).array()).sum(), (Vmean.col(2).array()* Vmean.col(2).array()).sum();

    Eigen::EigenSolver<Eigen::Matrix3d> es;
    es.compute(Covariance, true);
    Eigen::Matrix3d EigVecs = es.pseudoEigenvectors();//'pseudo', only real part
    Eigen::Vector3d lambdas = es.pseudoEigenvalueMatrix().diagonal();
    dbg(EigVecs) dbg(lambdas);
    EigVecs.colwise().normalize();
    dbg(EigVecs);//Rotation matrix

    Eigen::Matrix4d T_orig_oobb(Eigen::Matrix4d::Identity());
    T_orig_oobb.topLeftCorner<3, 3>()=EigVecs;
    T_orig_oobb.topRightCorner<3, 1>() = centroid.transpose();
    Eigen::Matrix4d T_oobb_orig = T_orig_oobb.inverse();
    Vrot = (T_oobb_orig.topLeftCorner<3, 3>() * V.transpose() + T_oobb_orig.topRightCorner<3, 1>().replicate(1, V.rows())).transpose();
    Nrot = (EigVecs.transpose() * N.transpose()).transpose();//free vector, only rotation
    return AABB(Vrot);
}



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
bool find_n_radius(
    const Eigen::MatrixX3d& pts,
    const grid3d& grid,
    const Eigen::RowVector3d& p,
    const double radius,
    vector<int>& idx
) {
    idx.clear();

    if (!grid.inside(p)) {//Not implemented
        const auto bruteforce_radius = [&](const Eigen::RowVector3d& pt, const double radius, vector<int>& idx) {
            const auto m = pts.rows();
            double radiussq(radius * radius);
            //Eigen::VectorXd dist((((pts - pt.replicate(pts.rows(), 1)).rowwise().squaredNorm())));// .array() > Eigen::Matrix<double, 1, 1>(radiussq).replicate(pts.rows(), 1).array()).cast<int>());
            for (int i = 0; i < m; i++) {
                if (((pts(i, 0) - pt[0]) * (pts(i, 0) - pt[0]) + (pts(i, 1) - pt[1]) * (pts(i, 1) - pt[1]) + (pts(i, 2) - pt[2]) * (pts(i, 2) - pt[2])) < radiussq)// < radius)//Discard real distance
                //if(dist[i]<radiussq)
                    idx.push_back(i);
            }
        };
        bruteforce_radius(p, radius, idx);
        return idx.size();
    }

    //coarse search using Manhattan distance
    //number of bins width at grid's resolution
    auto voxel_dim = static_cast<int>(ceil((radius / grid.res)));
    
    int voxeldimminsq = ((voxel_dim-2)<0) ? 0 : ((voxel_dim - 2)* (voxel_dim - 2));

    Eigen::RowVector3i minvox, maxvox;
    double radiussqr = radius * radius;
    grid.truncated_voxel(p, voxel_dim, minvox, maxvox);
    Eigen::RowVector3i binp(grid.binpoint(p));
    for (int i = minvox[0]; i <= maxvox[0]; i++) {
        for (int j = minvox[1]; j <= maxvox[1]; j++) {
            for (int k = minvox[2]; k <= maxvox[2]; k++) {
                
                if (!grid.sizeAt(i,j,k))//micro-op
                    continue;
                
                auto& bin(grid.at(i, j, k));

                if (voxeldimminsq && 
                    (((i - binp[0]) * (i - binp[0]) + (j - binp[1]) * (j - binp[1]) + (k - binp[2]) * (k - binp[2])) <= voxeldimminsq)) 
                {//inside radius
                    idx.insert(idx.end(), bin.begin(), bin.end());
                    continue;
                }
                for (auto id : bin) {//Discard real distance
                    if( ((pts(id, 0) - p[0])* (pts(id, 0) - p[0]) + (pts(id, 1) - p[1])* (pts(id, 1) - p[1]) + (pts(id, 2) - p[2])* (pts(id, 2) - p[2])) < radiussqr)
                        idx.emplace_back(id);
                }
                
            }//k
        }//j
    }//i
    return idx.size();
}


//Builds a 3d grid with the vertex idx in each node
template<bool saveVgridIdx> //compile time branch
void spatial_indexer_3dgrid(
    const Eigen::MatrixX3d& V,
    const double resolution,
    grid3d &grid,
    Eigen::MatrixX2i &VgridIdx
) {
    auto bb = AABB(V);
    Eigen::RowVector3d bblength(bb.block<1, 3>(1, 0) - bb.block<1, 3>(0, 0));

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
    grid.binSizes.resize(grid.grid.size());
    for (int i = 0; i < grid.grid.size(); ++i)
        grid.binSizes[i] = grid.grid[i].size();
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

template<bool saveVgridIdx> //compile time branch
void spatial_indexer_3dgrid(
    const Eigen::MatrixX3d& V,
    const int factor,
    grid3d& grid,
    Eigen::MatrixX2i& VgridIdx
) {
    //determine the resolution
    auto bb = AABB(V);
    Eigen::RowVector3d bblength(bb.block<1, 3>(1, 0) - bb.block<1, 3>(0, 0));
    double resolution = bblength.minCoeff() / factor;
    spatial_indexer_3dgrid<saveVgridIdx>(V, resolution, grid, VgridIdx);
}

void add_constraints(
    const Eigen::MatrixX3d& V,
    const Eigen::MatrixX3d& N_norm,
    Eigen::MatrixX3d& Vconstr,
    Eigen::VectorXd& Valconstr
) {
    assert(V.rows() == N_norm.rows());

    const auto m = V.rows();

    auto bb = AABB(V);
    double Epsilon = 0.01 * abs((bb.block<1, 3>(0, 0) - bb.block<1, 3>(1, 0)).norm());//Epsilon 1/100th of bb diagonal

    //Epsilon fine tuning
    //  Assuming grid res > Epsilon => closest only in the same bin
    Eigen::MatrixX3d Vconstrloc;
    Vconstrloc.resize(m * 3, 3);

    double mindist = numeric_limits<double>::max();
    bool keepgoing = true;
    volatile double average_points_per_bin = 5;
    grid3d grid;
    volatile int factor_subdivision_grid = 2;
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
    
    Valconstr.resize(Vconstr.rows());
    Valconstr.block(0, 0, m, 1).setZero();
    Valconstr.block(m, 0, m, 1).setConstant(Epsilon);
    Valconstr.block(m * 2, 0, m, 1).setConstant(-Epsilon);
    
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
        // Visualize current bounding box
        Eigen::MatrixX3d Prot, Nrot;
        Eigen::Matrix4d T_oobb_orig;
        auto bb = OOBB(P, N, Prot, Nrot);
        P = Prot;
        N = Nrot;
        callback_key_down(viewer, '1', modifiers);//update to new rotated points
        DrawBB(viewer, bb);
    }

    if (key == '3') {
        // Add your code for computing auxiliary constraint points here
        Eigen::MatrixX3d Pconstr;
        N.rowwise().normalize();
        add_constraints(P, N, Pconstr, constrained_values);
        constrained_points = Pconstr;
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

    if (key == '4') {
        // Show grid points with colored nodes and connected with lines
        viewer.data().clear();
        viewer.core.align_camera_center(P);
        // Add code for creating a grid
        // Add your code for evaluating the implicit function at the grid points
        // Add code for displaying points and lines
        // You can use the following example:
        //////constexpr int n = 100000;
        auto aabb(AABB(P));
        const Eigen::RowVector3d sz(aabb.row(1) - aabb.row(0));
        double absres = sz.minCoeff() / resolution;
        //////grid3d grid;
        //////{
        //////    Eigen::MatrixX2i VgridIdx;
        //////    spatial_indexer_3dgrid<false>(P, absres, grid, VgridIdx);
        //////}
        //////{
        //////    const auto bruteforce_radius = [&](const Eigen::RowVector3d& pt, const double radius, vector<int> &idx) {
        //////        const auto m = P.rows();
        //////        double radiussq(radius * radius);
        //////        //Eigen::VectorXd dist((((P - pt.replicate(P.rows(), 1)).rowwise().squaredNorm())));// .array() > Eigen::Matrix<double, 1, 1>(radiussq).replicate(P.rows(), 1).array()).cast<int>());
        //////        for (int i = 0; i < m; i++) {
        //////            if (((P(i, 0) - pt[0]) * (P(i, 0) - pt[0]) + (P(i, 1) - pt[1]) * (P(i, 1) - pt[1]) + (P(i, 2) - pt[2]) * (P(i, 2) - pt[2])) < radiussq)// < radius)//Discard real distance
        //////            //if(dist[i]<radiussq)
        //////                idx.push_back(i);
        //////        }
        //////    };
        //////    /*DEBUG 3D GRID*/
        //////    auto bb = AABB(P); 
        //////    Eigen::RowVector3d dims(bb.row(1) - bb.row(0));
        //////    
        //////    
        //////    Eigen::Matrix<double, -1, 3> pts_test; pts_test.resize(n, 3);
        //////    for (int i = 0; i < n; i++) {
        //////        pts_test.row(i) = Eigen::RowVector3d((double)rand() / (double)RAND_MAX, (double)rand() / (double)RAND_MAX, (double)rand() / (double)RAND_MAX);
        //////        pts_test.row(i) = (pts_test.row(i).array() * dims.array()).matrix() + bb.row(0);
        //////    }
        //////    cout << "ini\n";
        //////    vector<int> idx_num_n_radius, idx_num_n_brute;
        //////    vector<int> idx;
        //////    for (int i = 0; i < n; i++) {
        //////        idx.clear();
        //////        find_n_radius(grid, Eigen::RowVector3d(pts_test.row(i)), radius, idx);
        //////        idx_num_n_radius.push_back(idx.size());
        //////    }
        //////    cout << "bruteforce\n" << endl;;
        //////    for (int i = 0; i < n; i++) {
        //////        idx.clear();
        //////        bruteforce_radius(Eigen::RowVector3d(pts_test.row(i)), radius, idx);
        //////        idx_num_n_brute.push_back(idx.size());
        //////    }
        //////    
        //////    cout << "idx_num_n_brute[10] " << idx_num_n_brute[10] << endl << "idx_num_n_radius[100] " << idx_num_n_radius[10] << endl;

        //////    cout << "end\n";
        //////    //    //cout << "find_n_radius\tpt" << i << "\t" << pts_test.row(i) << endl << "\tn = " << idx.size() << endl << endl;
        //////    //    idx.clear();
        //////    //    bruteforce_radius(Eigen::RowVector3d(pts_test.row(i)), 0.25, idx);
        //////    //    cout << "bruteforce_radius\tpt" << i << "\t" << pts_test.row(i) << endl << "\tn = " << idx.size() << endl << endl;
        //////    //}
        //////}
        /*** begin: sphere example, replace (at least partially) with your code ***/
        // Make grid
        createGrid();
        grid3d grid;
        {
            Eigen::MatrixX2i VgridIdx;
            spatial_indexer_3dgrid<false>(constrained_points, absres, grid, VgridIdx);
        }

        const auto WeightFuncWendland = [](const double r, const double h) -> double
        { return (4 * r / h + 1) * std::pow((1 - r / h), 4); };
        const auto WeightFuncGauss = [](const double r, const double h) -> double
        { return exp(-r * r / (h * h)); };
        const auto WeightFuncSingular = [=](const double r, const double h) -> double
        { return (1 / (r * r + EpsilonConstr / 10)); };

        polyDegree = polyDegree % 3;
        weighfunc = weighfunc % 3;

        // Evaluate implicit function
        if (polyDegree == 0 && weighfunc == 0)
            evaluateImplicitFunc<0>(constrained_points, constrained_values, grid, absres * 8, WeightFuncWendland);
        else if (polyDegree == 1 && weighfunc == 0)
            evaluateImplicitFunc<1>(constrained_points, constrained_values, grid, absres * 8, WeightFuncWendland);
        else if (polyDegree == 2 && weighfunc == 0)
            evaluateImplicitFunc<2>(constrained_points, constrained_values, grid, absres * 8, WeightFuncWendland);
        else if (polyDegree == 0 && weighfunc == 1)
            evaluateImplicitFunc<0>(constrained_points, constrained_values, grid, absres * 8, WeightFuncGauss);
        else if (polyDegree == 1 && weighfunc == 1)
            evaluateImplicitFunc<1>(constrained_points, constrained_values, grid, absres * 8, WeightFuncGauss);
        else if (polyDegree == 2 && weighfunc == 1)
            evaluateImplicitFunc<2>(constrained_points, constrained_values, grid, absres * 8, WeightFuncGauss);
        else if (polyDegree == 0 && weighfunc == 2)
            evaluateImplicitFunc<0>(constrained_points, constrained_values, grid, absres * 8, WeightFuncSingular);
        else if (polyDegree == 1 && weighfunc == 2)
            evaluateImplicitFunc<1>(constrained_points, constrained_values, grid, absres * 8, WeightFuncSingular);
        else if (polyDegree == 2 && weighfunc == 2)
            evaluateImplicitFunc<2>(constrained_points, constrained_values, grid, absres * 8, WeightFuncSingular);

        VisGrid3D(viewer);
    }
    if (key == '5') {
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
          ImGui::InputInt("Polydegree", &polyDegree, 0, 0);
          ImGui::InputInt("WeighFunc", &weighfunc, 0, 0);
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
