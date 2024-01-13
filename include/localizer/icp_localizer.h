#pragma once
#include <string>
#include "commons.h"
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/registration/icp.h>
#include <pcl/features/normal_3d.h>

namespace lio
{

    struct LocalizerParams
    {
        double rough_resolution = 0.5;
        double refine_resolution = 0.2;
        double rough_iter = 10;
        double refine_iter = 5;
        double thresh = 0.15;
        double xy_offset = 2.0;
        int yaw_offset = 3;
        double yaw_resolution = M_PI / 6;
    };

    class IcpLocalizer
    {
    public:
        IcpLocalizer()
            : refine_resolution_(0.2),
              rough_resolution_(0.5),
              rough_iter_(10),
              refine_iter_(5),
              thresh_(0.15)
        {
            voxel_rough_filter_.setLeafSize(rough_resolution_, rough_resolution_, rough_resolution_);
            voxel_refine_filter_.setLeafSize(refine_resolution_, refine_resolution_, refine_resolution_);
        }
        IcpLocalizer(double refine_resolution, double rough_resolution, int refine_iter, int rough_iter, double thresh)
            : refine_resolution_(refine_resolution),
              rough_resolution_(rough_resolution),
              refine_iter_(refine_iter),
              rough_iter_(rough_iter),
              thresh_(thresh)
        {
            voxel_rough_filter_.setLeafSize(rough_resolution_, rough_resolution_, rough_resolution_);
            voxel_refine_filter_.setLeafSize(refine_resolution_, refine_resolution_, refine_resolution_);
        }
        double getScore() const { return score_; }

        bool isSuccess() const { return success_; }

        bool isInitialized() const { return initialized_; }

        void init(const std::string &pcd_path, bool with_norm);

        static PointCloudXYZI::Ptr addNorm(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);

        Eigen::Matrix4d align(pcl::PointCloud<pcl::PointXYZI>::Ptr source, Eigen::Matrix4d init_guess);

        Eigen::Matrix4d multi_align_sync(pcl::PointCloud<pcl::PointXYZI>::Ptr source, Eigen::Matrix4d init_guess);

        PointCloudXYZI::Ptr getRoughMap() { return rough_map_; }

        PointCloudXYZI::Ptr getRefineMap() { return refine_map_; }

        void writePCDToFile(const std::string &path, bool detail);

        void setParams(double refine_resolution, double rough_resolution, int refine_iter, int rough_iter, double thresh);

        void setSearchParams(double xy_offset, int yaw_offset, double yaw_res);

    private:
        std::string pcd_path_;
        PointCloudXYZI::Ptr refine_map_;
        PointCloudXYZI::Ptr rough_map_;
        double refine_resolution_;
        double rough_resolution_;
        int rough_iter_;
        int refine_iter_;
        double thresh_;
        double score_ = 10.0;
        bool success_ = false;
        bool initialized_ = false;
        double xy_offset_ = 2.0;
        int yaw_offset_ = 3;
        double yaw_resolution_ = M_PI / 6;
        pcl::VoxelGrid<pcl::PointXYZI> voxel_rough_filter_;
        pcl::VoxelGrid<pcl::PointXYZI> voxel_refine_filter_;
        pcl::IterativeClosestPointWithNormals<PointType, PointType> icp_rough_;
        pcl::IterativeClosestPointWithNormals<PointType, PointType> icp_refine_;
    };
}
