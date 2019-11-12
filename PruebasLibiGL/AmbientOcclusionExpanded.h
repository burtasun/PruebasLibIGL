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

using namespace std;
using namespace Eigen;


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

	template <
		typename DerivedP,
		typename DerivedN,
		typename DerivedU,
		typename DerivedSampl,
		typename DerivedS,
		typename DerivedAOVect>
		IGL_INLINE bool ambient_occlusion_expanded_impl_sectorized(
			const std::function<
			bool(
				const Eigen::Vector3f&,
				const Eigen::Vector3f&)
			> & shoot_ray,
			const Eigen::PlainObjectBase<DerivedP> & P,
			const Eigen::PlainObjectBase<DerivedN> & N,
			const Eigen::PlainObjectBase<DerivedU> & U,//first line of face
			const Eigen::PlainObjectBase<DerivedSampl> &samples, //n_samples X (m_sectors x 3)
			Eigen::PlainObjectBase<DerivedS> & S, //ratio occlussion
			Eigen::PlainObjectBase<DerivedAOVect> & VO);//non occluded averaged direction


	template <
		typename DerivedV,
		typename DerivedF,
		typename DerivedP,
		typename DerivedN,
		typename DerivedSampl,
		typename DerivedS,
		typename DerivedAOVect>
		IGL_INLINE  bool ambient_occlusion_expanded_embree_sector(
			const Eigen::PlainObjectBase<DerivedV> & V,
			const Eigen::PlainObjectBase<DerivedF> & F,
			const Eigen::PlainObjectBase<DerivedP> & P,
			const Eigen::PlainObjectBase<DerivedN> & N,
			const Eigen::PlainObjectBase<DerivedSampl> & samples,
			Eigen::PlainObjectBase<DerivedS> & S,
			Eigen::PlainObjectBase< DerivedAOVect> &VO);

	bool face_baricentre(Eigen::MatrixXd &F_baricentres, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);

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
					d *= -1;
				if (shoot_ray(origin, d))
					num_hits++;
				else
					VectNoOcclud = VectNoOcclud + d.normalized();
			}
			S(p) = (double)num_hits / (double)num_samples;
			auto a = (VectNoOcclud.normalized() / (num_samples - num_hits)).cast<RowVector3d>();
			if (num_samples != num_hits)
				VO.row(p) = (VectNoOcclud / (num_samples - num_hits)).transpose().normalized().cast<double>();
		};
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





	template <
		typename DerivedP,
		typename DerivedN,
		typename DerivedU,
		typename DerivedSampl,
		typename DerivedS,
		typename DerivedAOVect>
		IGL_INLINE bool ambient_occlusion_expanded_impl_sectorized(
			const std::function<
			bool(
				const Eigen::Vector3f&,
				const Eigen::Vector3f&)
			> & shoot_ray,
			const Eigen::PlainObjectBase<DerivedP> & P,
			const Eigen::PlainObjectBase<DerivedN> & N,
			const Eigen::PlainObjectBase<DerivedU> & U,//first line of face
			const Eigen::PlainObjectBase<DerivedSampl> &samples, //n_samples X (m_sectors x 3)
			Eigen::PlainObjectBase<DerivedS> & S, //ratio occlussion
			Eigen::PlainObjectBase<DerivedAOVect> & VO)//non occluded averaged direction
	{
		using namespace Eigen;
		const int n = P.rows();

		int num_samples = samples.rows();
		if ((samples.cols() % 3) != 0) {
			cerr << "ambient_occlusion_expanded_impl_sectorized\n\tNumber of columns not a multiple of 3" << endl;
			return false;
		}
		int num_sectors = std::round(static_cast<double>(samples.cols()) / 3.0);

		S.resize(n, num_sectors);
		VO.resize(n, 3 * num_sectors);

		//Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

		const auto & inner = [&P, &N, &U, &num_samples, &num_sectors, &samples, &S, &shoot_ray, &VO/*, &CleanFmt*/](const int p)
		{
			const Vector3f origin = P.row(p).template cast<float>();
			const Vector3f normal = N.row(p).template cast<float>();
			const Vector3f u_iso = U.row(p).template cast<float>();

			//Vector3f v_prim_iso = normal.cross(u_iso).normalized();
			//Matrix3f RotSampl;
			//RotSampl <<
			//	u_iso(0), v_prim_iso(0), normal(0),
			//	u_iso(1), v_prim_iso(1), normal(1),
			//	u_iso(2), v_prim_iso(2), normal(2);
			Vector3f x_vect_rot = Vector3f::UnitY().cross(normal);
			Vector3f y_vect_rot = normal.cross(x_vect_rot);
			Matrix3f RotSampl;
			RotSampl <<
				x_vect_rot(0), y_vect_rot(0), normal(0),
				x_vect_rot(1), y_vect_rot(1), normal(1),
				x_vect_rot(2), y_vect_rot(2), normal(2);

			//int invert = 0;
			for (int sect = 0; sect < num_sectors; sect++)
			{
				Vector3f VectNoOcclud(0, 0, 0);
				int num_hits = 0, invert = 0;
				for (int s = 0; s < num_samples; s++)
				{
					Vector3f d = RotSampl * samples.block(s, sect * 3, 1, 3).transpose();
					if (d.dot(normal) < 0) {
						d *= -1;
						invert++;
					}
					if (shoot_ray(origin, d))
						num_hits++;
					else
						VectNoOcclud = VectNoOcclud + d.normalized();
				}
				S(p, sect) = (float)num_hits / (float)num_samples;
				if (num_samples != num_hits)
					VO.block(p, 3 * sect, 1, 3) = (VectNoOcclud / (num_samples - num_hits)).transpose().normalized().cast<float>();
			}
		};
		igl::parallel_for(n, inner, 1000);
		return true;
	}




	template <
		typename DerivedV,
		typename DerivedF,
		typename DerivedP,
		typename DerivedN,
		typename DerivedU,
		typename DerivedSampl,
		typename DerivedS,
		typename DerivedAOVect>
		IGL_INLINE  bool ambient_occlusion_expanded_embree_sector(
			const Eigen::PlainObjectBase<DerivedV> & V,
			const Eigen::PlainObjectBase<DerivedF> & F,
			const Eigen::PlainObjectBase<DerivedP> & P,
			const Eigen::PlainObjectBase<DerivedN> & N,
			const Eigen::PlainObjectBase<DerivedU> & U,
			const Eigen::PlainObjectBase<DerivedSampl> & samples,
			Eigen::PlainObjectBase<DerivedS> & S,
			Eigen::PlainObjectBase< DerivedAOVect> &VO)
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
		return AmbientOcclusionExpanded::ambient_occlusion_expanded_impl_sectorized(shoot_ray, P, N, U, samples, S, VO);
	}







	bool face_baricentre(Eigen::MatrixXd &F_baricentres, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) {
		int Frows = F.rows();
		F_baricentres.resize(Frows, 3);
#pragma omp parallel for if (Frows>10000)
		for (int i = 0; i < Frows; i++)
			F_baricentres.row(i) = ((V.row(F(i, 0)) + V.row(F(i, 1)) + V.row(F(i, 2))) / 3).normalized();
		return true;
	}

	bool iso_parameters(Eigen::MatrixXd &U_isoparam, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) {
		int Frows = F.rows();
		U_isoparam.resize(Frows, 3);
#pragma omp parallel for if (Frows>10000)
		for (int i = 0; i < Frows; i++)
			U_isoparam.row(i) = (V.row(F(i, 0)) - V.row(F(i, 2))).normalized();//P1 - P3 // counter-clockwise
		return true;
	}













};
