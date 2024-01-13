#pragma once
#include "commons.h"
#include "imu_processor.h"
#include "ieskf/ieskf.h"
#include "voxel_map/voxel_map.h"
#include <pcl/common/transforms.h>

namespace lio
{
    struct GICPCorrespond
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d meanA;
        Eigen::Vector3d meanB;
        Eigen::Matrix3d covA;
        Eigen::Matrix3d covB;
        GICPCorrespond(const Eigen::Vector3d &a, const Eigen::Vector3d &b, const Eigen::Matrix3d &ca, const Eigen::Matrix3d &cb) : meanA(a), meanB(b), covA(ca), covB(cb) {}
    };
    struct FASTLIOCorrspond
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d point;
        Eigen::Vector4d plane;
    };
    inline void CauchyLossFunction(const double e, const double delta, Eigen::Vector3d &rho);
    struct IGLIOParams
    {
        double point2plane_gain = 1000.0;
        double plane2plane_gain = 100.0;
        double scan_resolution = 0.5;
        double map_resolution = 0.5;
        int max_points_per_scan = 10000;
        size_t map_capacity = 5000000;
        size_t grid_capacity = 20;
        VoxelMap::MODE mode = VoxelMap::MODE::NEARBY_7;
        double esikf_min_iteration = 2;
        double esikf_max_iteration = 30;
        double imu_acc_cov = 0.01;
        double imu_gyro_cov = 0.01;
        double imu_acc_bias_cov = 0.0001;
        double imu_gyro_bias_cov = 0.0001;

        std::vector<double> imu_ext_rot = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::vector<double> imu_ext_pos = {-0.011, -0.02329, 0.04412};

        bool extrinsic_est_en = false;
        bool align_gravity = true;
    };
    class IGLIOBuilder
    {
    public:
        IGLIOBuilder(IGLIOParams &params);

        Status currentStatus() const { return status; }

        kf::State currentState() const { return kf_->x(); }

        void mapping(const MeasureGroup &meas);

        void sharedUpdateFunc(kf::State &, kf::SharedState &);

        void fastlioConstraint(kf::State &, kf::SharedState &);

        void gicpConstraint(kf::State &, kf::SharedState &);

        PointCloudXYZI::Ptr transformToWorld(const PointCloudXYZI::Ptr cloud);

        PointCloudXYZI::Ptr cloudUndistortedLidar() { return cloud_lidar_; }

        PointCloudXYZI::Ptr cloudUndistortedBody();

        PointCloudXYZI::Ptr cloudWorld();
        
        void reset();

    private:
        IGLIOParams params_;
        Status status = Status::INITIALIZE;
        std::shared_ptr<IMUProcessor> imu_processor_;
        std::shared_ptr<kf::IESKF> kf_;
        std::shared_ptr<VoxelMap> voxel_map_;
        std::shared_ptr<FastVoxelMap> fast_voxel_map_;
        PointCloudXYZI::Ptr cloud_lidar_;
        PointCloudXYZI::Ptr cloud_body_;
        PointCloudXYZI::Ptr cloud_world_;
        Eigen::Matrix3d key_rot_;
        Eigen::Vector3d key_pos_;
        size_t frame_count_ = 0;
        size_t key_frame_count_ = 0;

        std::vector<PointCov> point_array_lidar_;

        std::vector<FASTLIOCorrspond> fastlio_cache_;
        std::vector<GICPCorrespond> gicp_cache_;
        std::vector<bool> cache_flag_;
    };
} // namespace lio
