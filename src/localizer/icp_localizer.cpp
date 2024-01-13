#include "localizer/icp_localizer.h"
// #include <chrono>

namespace lio
{
    void IcpLocalizer::init(const std::string &pcd_path, bool with_norm)
    {
        if (!pcd_path_.empty() && pcd_path_ == pcd_path)
            return;
        pcl::PCDReader reader;
        pcd_path_ = pcd_path;
        if (!with_norm)
        {
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
            reader.read(pcd_path, *cloud);
            voxel_refine_filter_.setInputCloud(cloud);
            voxel_refine_filter_.filter(*cloud);
            refine_map_ = addNorm(cloud);
        }
        else
        {
            refine_map_.reset(new PointCloudXYZI);
            reader.read(pcd_path, *refine_map_);
        }
        pcl::PointCloud<pcl::PointXYZI>::Ptr point_rough(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr filterd_point_rough(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::copyPointCloud(*refine_map_, *point_rough);
        voxel_rough_filter_.setInputCloud(point_rough);
        voxel_rough_filter_.filter(*filterd_point_rough);
        rough_map_ = addNorm(filterd_point_rough);

        icp_rough_.setMaximumIterations(rough_iter_);
        icp_rough_.setInputTarget(rough_map_);

        icp_refine_.setMaximumIterations(refine_iter_);
        icp_refine_.setInputTarget(refine_map_);
        initialized_ = true;
    }

    Eigen::Matrix4d IcpLocalizer::align(pcl::PointCloud<pcl::PointXYZI>::Ptr source, Eigen::Matrix4d init_guess)
    {
        success_ = false;
        Eigen::Vector3d xyz = init_guess.block<3, 1>(0, 3);

        pcl::PointCloud<pcl::PointXYZI>::Ptr rough_source(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr refine_source(new pcl::PointCloud<pcl::PointXYZI>);

        voxel_rough_filter_.setInputCloud(source);
        voxel_rough_filter_.filter(*rough_source);
        voxel_refine_filter_.setInputCloud(source);
        voxel_refine_filter_.filter(*refine_source);

        PointCloudXYZI::Ptr rough_source_norm = addNorm(rough_source);
        PointCloudXYZI::Ptr refine_source_norm = addNorm(refine_source);
        PointCloudXYZI::Ptr align_point(new PointCloudXYZI);
        // auto tic = std::chrono::system_clock::now();
        icp_rough_.setInputSource(rough_source_norm);
        icp_rough_.align(*align_point, init_guess.cast<float>());

        score_ = icp_rough_.getFitnessScore();
        if (!icp_rough_.hasConverged())
            return Eigen::Matrix4d::Zero();

        icp_refine_.setInputSource(refine_source_norm);
        icp_refine_.align(*align_point, icp_rough_.getFinalTransformation());
        score_ = icp_refine_.getFitnessScore();

        if (!icp_refine_.hasConverged())
            return Eigen::Matrix4d::Zero();
        if (score_ > thresh_)
            return Eigen::Matrix4d::Zero();
        success_ = true;
        // auto toc = std::chrono::system_clock::now();
        // std::chrono::duration<double> duration = toc - tic;
        // std::cout << "align used: " << duration.count() * 1000 << std::endl;
        return icp_refine_.getFinalTransformation().cast<double>();
    }

    PointCloudXYZI::Ptr IcpLocalizer::addNorm(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
    {
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        pcl::search::KdTree<pcl::PointXYZI>::Ptr searchTree(new pcl::search::KdTree<pcl::PointXYZI>);
        searchTree->setInputCloud(cloud);

        pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> normalEstimator;
        normalEstimator.setInputCloud(cloud);
        normalEstimator.setSearchMethod(searchTree);
        normalEstimator.setKSearch(15);
        normalEstimator.compute(*normals);
        PointCloudXYZI::Ptr out(new PointCloudXYZI);
        pcl::concatenateFields(*cloud, *normals, *out);
        return out;
    }

    void IcpLocalizer::writePCDToFile(const std::string &path, bool detail)
    {
        if (!initialized_)
            return;
        pcl::PCDWriter writer;
        writer.writeBinaryCompressed(path, detail ? *refine_map_ : *rough_map_);
    }

    void IcpLocalizer::setParams(double refine_resolution, double rough_resolution, int refine_iter, int rough_iter, double thresh)
    {
        refine_resolution_ = refine_resolution;
        rough_resolution_ = rough_resolution;
        refine_iter_ = refine_iter;
        rough_iter_ = rough_iter;
        thresh_ = thresh;
    }

    void IcpLocalizer::setSearchParams(double xy_offset, int yaw_offset, double yaw_res){
        xy_offset_ = xy_offset;
        yaw_offset_ = yaw_offset;
        yaw_resolution_ = yaw_res;
    }

    Eigen::Matrix4d IcpLocalizer::multi_align_sync(pcl::PointCloud<pcl::PointXYZI>::Ptr source, Eigen::Matrix4d init_guess)
    {
        success_ = false;
        Eigen::Vector3d xyz = init_guess.block<3, 1>(0, 3);
        Eigen::Matrix3d rotation = init_guess.block<3, 3>(0, 0);
        Eigen::Vector3d rpy = rotate2rpy(rotation);
        Eigen::AngleAxisf rollAngle(rpy(0), Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf pitchAngle(rpy(1), Eigen::Vector3f::UnitY());
        std::vector<Eigen::Matrix4f> candidates;
        Eigen::Matrix4f temp_pose;

        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                for (int k = -yaw_offset_; k <= yaw_offset_; k++)
                {
                    Eigen::Vector3f pos(xyz(0) + i * xy_offset_, xyz(1) + j * xy_offset_, xyz(2));
                    Eigen::AngleAxisf yawAngle(rpy(2) + k * yaw_resolution_, Eigen::Vector3f::UnitZ());
                    temp_pose.setIdentity();
                    temp_pose.block<3, 3>(0, 0) = (rollAngle * pitchAngle * yawAngle).toRotationMatrix();
                    temp_pose.block<3, 1>(0, 3) = pos;
                    candidates.push_back(temp_pose);
                }
            }
        }
        pcl::PointCloud<pcl::PointXYZI>::Ptr rough_source(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr refine_source(new pcl::PointCloud<pcl::PointXYZI>);

        voxel_rough_filter_.setInputCloud(source);
        voxel_rough_filter_.filter(*rough_source);
        voxel_refine_filter_.setInputCloud(source);
        voxel_refine_filter_.filter(*refine_source);

        PointCloudXYZI::Ptr rough_source_norm = addNorm(rough_source);
        PointCloudXYZI::Ptr refine_source_norm = addNorm(refine_source);
        PointCloudXYZI::Ptr align_point(new PointCloudXYZI);

        Eigen::Matrix4f best_rough_transform;
        double best_rough_score = 10.0;
        bool rough_converge = false;
        // auto tic = std::chrono::system_clock::now();
        for (Eigen::Matrix4f &init_pose : candidates)
        {
            icp_rough_.setInputSource(rough_source_norm);
            icp_rough_.align(*align_point, init_pose);
            if (!icp_rough_.hasConverged())
                continue;
            double rough_score = icp_rough_.getFitnessScore();
            if (rough_score > 2 * thresh_)
                continue;
            if (rough_score < best_rough_score)
            {
                best_rough_score = rough_score;
                rough_converge = true;
                best_rough_transform = icp_rough_.getFinalTransformation();
            }
        }

        if (!rough_converge)
            return Eigen::Matrix4d::Zero();

        icp_refine_.setInputSource(refine_source_norm);
        icp_refine_.align(*align_point, best_rough_transform);
        score_ = icp_refine_.getFitnessScore();

        if (!icp_refine_.hasConverged())
            return Eigen::Matrix4d::Zero();
        if (score_ > thresh_)
            return Eigen::Matrix4d::Zero();
        success_ = true;
        // auto toc = std::chrono::system_clock::now();
        // std::chrono::duration<double> duration = toc - tic;
        // std::cout << "multi_align used: " << duration.count() * 1000 << std::endl;
        return icp_refine_.getFinalTransformation().cast<double>();
    }
}