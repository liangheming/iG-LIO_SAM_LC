#pragma once
#include "commons.h"
#include "imu_processor.h"
#include "ikd-Tree/ikd_Tree.h"
#include <pcl/filters/voxel_grid.h>

namespace lio
{

    struct LocalMap
    {
        double cube_len;
        double det_range;
        double move_thresh;
        bool is_initialed = false;
        BoxPointType local_map_corner;
        std::vector<BoxPointType> cub_to_rm;
    };
    struct FASTLIOParams
    {
        double resolution = 0.5;
        double esikf_min_iteration = 2;
        double esikf_max_iteration = 5;
        double imu_acc_cov = 0.01;
        double imu_gyro_cov = 0.01;
        double imu_acc_bias_cov = 0.0001;
        double imu_gyro_bias_cov = 0.0001;

        std::vector<double> imu_ext_rot = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::vector<double> imu_ext_pos = {-0.011, -0.02329, 0.04412};

        double cube_len = 500.0;
        double det_range = 100.0;
        double move_thresh = 1.5;

        bool extrinsic_est_en = false;
        bool align_gravity = true;
    };

    class FastLIOBuilder
    {
    public:
        FastLIOBuilder(FASTLIOParams &params);

        void mapping(const MeasureGroup &meas);

        void sharedUpdateFunc(kf::State &, kf::SharedState &);

        PointCloudXYZI::Ptr transformToWorld(const PointCloudXYZI::Ptr cloud);

        void trimMap();

        void increaseMap();
        
        void reset();

        kf::State currentState() const { return kf_->x(); }
        Status currentStatus() const { return status; }
        std::shared_ptr<KD_TREE<PointType>> getIKDtree() { return ikdtree_; }
        std::shared_ptr<kf::IESKF> getKF() { return kf_; }
        std::shared_ptr<IMUProcessor> getIMUProcessor() { return imu_processor_; }
        PointCloudXYZI::Ptr cloudWorld() { return cloud_down_world_; }
        PointCloudXYZI::Ptr cloudLidar() { return cloud_down_lidar_; }
        PointCloudXYZI::Ptr cloudUndistortedLidar() { return cloud_undistorted_lidar_; }
        PointCloudXYZI::Ptr cloudUndistortedBody();
        PointCloudXYZI::Ptr cloudDownBody();

    private:
        FASTLIOParams params_;
        Status status = Status::INITIALIZE;
        std::shared_ptr<IMUProcessor> imu_processor_;
        std::shared_ptr<kf::IESKF> kf_;
        std::shared_ptr<KD_TREE<PointType>> ikdtree_;
        pcl::VoxelGrid<PointType> down_size_filter_;

        PointCloudXYZI::Ptr cloud_undistorted_lidar_;
        LocalMap local_map_;

        bool extrinsic_est_en_;
        PointCloudXYZI::Ptr cloud_down_lidar_;
        PointCloudXYZI::Ptr cloud_down_world_;
        std::vector<PointVector> nearest_points_;
        std::vector<bool> point_selected_flag_;
        PointCloudXYZI::Ptr norm_vec_;

        PointCloudXYZI::Ptr effect_cloud_lidar_;
        PointCloudXYZI::Ptr effect_norm_vec_;
    };
}
