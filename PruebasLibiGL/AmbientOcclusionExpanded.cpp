#include "AmbientOcclusionExpanded.h"

#include "igl/ambient_occlusion.h"
#include "igl/random_dir.h"
#include "igl/ray_mesh_intersect.h"
#include "igl/EPS.h"
#include "igl/Hit.h"
#include "igl/parallel_for.h"
#include <functional>
#include <vector>
#include <algorithm>
//embree
#include "igl/embree/ambient_occlusion.h"
#include "igl/embree/../ambient_occlusion.h"
#include "igl/embree/EmbreeIntersector.h"
#include "igl/embree/../Hit.h"

using namespace std;

namespace AmbientOcclusionExpanded {
	template <
		typename DerivedP,
		typename DerivedN,
		typename DerivedS,
		typename DerivedAOVect>
		IGL_INLINE void ambient_occlusion_expanded_impl(
			const std::function<
			bool(
				const Eigen::Vector3f&,
				const Eigen::Vector3f&)
			> & shoot_ray,
			const Eigen::PlainObjectBase<DerivedP> & P,
			const Eigen::PlainObjectBase<DerivedN> & N,
			const int num_samples,
			Eigen::PlainObjectBase<DerivedS> & S,
			Eigen::PlainObjectBase<DerivedAOVect> & VO)
	{
		using namespace Eigen;
		const int n = P.rows();
		// Resize output
		S.resize(n, 1);
		VO.resize(n, 3);

		// Embree seems to be parallel when constructing but not when tracing rays
		const MatrixXf D = igl::random_dir_stratified(num_samples).cast<float>();

		const auto & inner = [&P, &N, &num_samples, &D, &S, &shoot_ray, &VO](const int p)
		{
			const Vector3f origin = P.row(p).template cast<float>();
			const Vector3f normal = N.row(p).template cast<float>();
			int num_hits = 0;
			Vector3f VectNoOcclud(0, 0, 0);
			for (int s = 0; s < num_samples; s++)
			{
				Vector3f d = D.row(s);
				if (d.dot(normal) < 0)
				{
					// reverse ray
					d *= -1;
				}
				if (shoot_ray(origin, d))
				{
					num_hits++;
				}
				else {
					VectNoOcclud = VectNoOcclud + d.normalized();
				}
			}
			//Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
			S(p) = (double)num_hits / (double)num_samples;
			auto a = (VectNoOcclud.normalized() / (num_samples - num_hits)).cast<RowVector3d>();
			//double factor = num_samples - num_hits;
			if (num_samples != num_hits)
				VO.row(p) = (VectNoOcclud / (num_samples - num_hits)).transpose().normalized().cast<double>();
		};

		//puenteo 1 hilo
		//for (int i = 0; i < n; i++) {
		//	inner(i);
		//}
		igl::parallel_for(n, inner, 1000);
	}


	template <
		typename DerivedV,
		typename DerivedF,
		typename DerivedP,
		typename DerivedN,
		typename DerivedS,
		typename DerivedAOVect>
		IGL_INLINE  void ambient_occlusion_expanded_embree(
			const Eigen::PlainObjectBase<DerivedV> & V,
			const Eigen::PlainObjectBase<DerivedF> & F,
			const Eigen::PlainObjectBase<DerivedP> & P,
			const Eigen::PlainObjectBase<DerivedN> & N,
			const int num_samples,
			Eigen::PlainObjectBase<DerivedS> & S,
			Eigen::PlainObjectBase< DerivedAOVect> &VO)//vector n occlud
	{
		using namespace Eigen;
		igl::embree::EmbreeIntersector ei;
		ei.init(V.template cast<float>(), F.template cast<int>());

		const auto & shoot_ray = [&ei](
			const Eigen::Vector3f& s,
			const Eigen::Vector3f& dir)->bool
		{
			igl::Hit hit;
			const float tnear = 1e-4f;
			return ei.intersectRay(s, dir, hit, tnear);
		};
		return ambient_occlusion_expanded_impl(shoot_ray, P, N, num_samples, S, VO);
	}

}