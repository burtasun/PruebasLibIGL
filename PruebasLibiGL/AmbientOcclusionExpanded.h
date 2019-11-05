#pragma once
//igl
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

namespace AmbientOcclusionExpanded 
{

	template <
		typename DerivedP,
		typename DerivedN,
		typename DerivedS,
		typename DerivedAOVect>
		IGL_INLINE static void ambient_occlusion_expanded_impl(
			const std::function<
			bool(
				const Eigen::Vector3f&,
				const Eigen::Vector3f&)
			> & shoot_ray,
			const Eigen::PlainObjectBase<DerivedP> & P,
			const Eigen::PlainObjectBase<DerivedN> & N,
			const int num_samples,
			Eigen::PlainObjectBase<DerivedS> & S,
			Eigen::PlainObjectBase<DerivedAOVect> & VO);


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
			Eigen::PlainObjectBase< DerivedAOVect> &VO);

	template void AmbientOcclusionExpanded::ambient_occlusion_expanded_embree<
		Eigen::Matrix<double, -1, 3, 0, -1, 3>,
		Eigen::Matrix<int, -1, 3, 0, -1, 3>,
		Eigen::Matrix<double, -1, 3, 0, -1, 3>,
		Eigen::Matrix<double, -1, 3, 0, -1, 3>,
		Eigen::Matrix<double, -1, 1, 0, -1, 1>,
		Eigen::Matrix<double, -1, 3, 0, -1, 3> >
		(
			Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> > const&,
			Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 3, 0, -1, 3> > const&,
			Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> > const&,
			Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> > const&, int,
			Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&,
			Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> >&);

	template void  AmbientOcclusionExpanded::ambient_occlusion_expanded_embree<
		Eigen::Matrix<double, -1, -1, 0, -1, -1>,
		Eigen::Matrix<int, -1, -1, 0, -1, -1>,
		Eigen::Matrix<double, -1, -1, 0, -1, -1>,
		Eigen::Matrix<double, -1, -1, 0, -1, -1>,
		Eigen::Matrix<double, -1, 1, 0, -1, 1>,
		Eigen::Matrix<double, -1, -1, 0, -1, -1> >
		(
			Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const &,
			Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const &,
			Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const &,
			Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const &,
			int,
			Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > &,
			Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > &
		);

	//template void igl::ambient_occlusion<
	//	Eigen::Matrix<double, -1, -1, 0, -1, -1>, 
	//	Eigen::Matrix<double, -1, -1, 0, -1, -1>, 
	//	Eigen::Matrix<double, -1, 1, 0, -1, 1> >
	//	(
	//		std::function<bool(Eigen::Matrix<float, 3, 1, 0, 3, 1> const&, Eigen::Matrix<float, 3, 1, 0, 3, 1> const&)> const&, 
	//		Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, 
	//		Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, 
	//		int, 
	//		Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
};
