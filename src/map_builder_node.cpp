#include <map>
#include <mutex>
#include <vector>
#include <thread>
#include <csignal>
#include <ros/ros.h>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "map_builder/iglio_builder.h"
#include "lio/SaveMap.h"
#include "localizer/icp_localizer.h"

class ZaxisPriorFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3>
{
    double z_;

public:
    ZaxisPriorFactor(gtsam::Key key, const gtsam::SharedNoiseModel &noise, double z)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(noise, key), z_(z)
    {
    }
    virtual ~ZaxisPriorFactor()
    {
    }
    virtual gtsam::Vector evaluateError(const gtsam::Pose3 &p, boost::optional<gtsam::Matrix &> H = boost::none) const
    {
        auto z = p.translation()(2);
        if (H)
        {
            gtsam::Matrix Jac = gtsam::Matrix::Zero(1, 6);
            Jac << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
            (*H) = Jac;
        }
        return gtsam::Vector1(z - z_);
    }
};

bool terminate_flag = false;

void signalHandler(int signum)
{
    std::cout << "SHUTTING DOWN MAPPING NODE!" << std::endl;
    terminate_flag = true;
}

struct LoopPair
{
    LoopPair(int p, int c, float s, Eigen::Matrix3d &dr, Eigen::Vector3d &dp) : pre_idx(p), cur_idx(c), score(s), diff_rot(dr), diff_pos(dp) {}
    int pre_idx;
    int cur_idx;
    Eigen::Matrix3d diff_rot;
    Eigen::Vector3d diff_pos;
    double score;
};

struct Pose6D
{
    Pose6D(int i, double t, Eigen::Matrix3d lr, Eigen::Vector3d lp) : index(i), time(t), local_rot(lr), local_pos(lp) {}
    void setGlobalPose(const Eigen::Matrix3d &gr, const Eigen::Vector3d &gp)
    {
        global_rot = gr;
        global_pos = gp;
    }
    void addOffset(const Eigen::Matrix3d &offset_rot, const Eigen::Vector3d &offset_pos)
    {
        global_rot = offset_rot * local_rot;
        global_pos = offset_rot * local_pos + offset_pos;
    }

    void getOffset(Eigen::Matrix3d &offset_rot, Eigen::Vector3d &offset_pos)
    {
        offset_rot = global_rot * local_rot.transpose();
        offset_pos = -global_rot * local_rot.transpose() * local_pos + global_pos;
    }
    int index;
    double time;
    Eigen::Matrix3d local_rot;
    Eigen::Vector3d local_pos;
    Eigen::Matrix3d global_rot;
    Eigen::Vector3d global_pos;
    // Eigen::Vector3d gravity;
};

struct SharedData
{
    bool key_pose_added = false;
    std::mutex mutex;
    Eigen::Matrix3d offset_rot = Eigen::Matrix3d::Identity();
    Eigen::Vector3d offset_pos = Eigen::Vector3d::Zero();
    std::vector<Pose6D> key_poses;
    std::vector<LoopPair> loop_pairs;
    std::vector<std::pair<int, int>> loop_history;
    std::vector<lio::PointCloudXYZI::Ptr> cloud_history;
};

struct LoopParams
{
    double rad_thresh = 0.4;
    double dist_thresh = 2.5;
    double time_thresh = 30.0;
    double loop_pose_search_radius = 10.0;
    int loop_pose_index_thresh = 5;
    double submap_resolution = 0.2;
    int submap_search_num = 20;
    double loop_icp_thresh = 0.3;
    bool activate = true;
    bool z_prior = false;
};

class LoopClosureThread
{
public:
    void init()
    {
        gtsam::ISAM2Params isam2_params;
        isam2_params.relinearizeThreshold = 0.01;
        isam2_params.relinearizeSkip = 1;
        isam2_ = std::make_shared<gtsam::ISAM2>(isam2_params);
        kdtree_history_poses_.reset(new pcl::KdTreeFLANN<pcl::PointXYZ>);
        cloud_history_poses_.reset(new pcl::PointCloud<pcl::PointXYZ>);
        sub_map_downsize_filter_.reset(new pcl::VoxelGrid<lio::PointType>);
        sub_map_downsize_filter_->setLeafSize(loop_params_.submap_resolution,
                                              loop_params_.submap_resolution,
                                              loop_params_.submap_resolution);

        icp_.reset(new pcl::IterativeClosestPoint<lio::PointType, lio::PointType>);
        icp_->setMaxCorrespondenceDistance(100);
        icp_->setMaximumIterations(50);
        icp_->setTransformationEpsilon(1e-6);
        icp_->setEuclideanFitnessEpsilon(1e-6);
        icp_->setRANSACIterations(0);
    }
    void setShared(std::shared_ptr<SharedData> share_data)
    {
        shared_data_ = share_data;
    }
    void setRate(const double &rate)
    {
        rate_ = std::make_shared<ros::Rate>(rate);
    }
    void setRate(std::shared_ptr<ros::Rate> rate)
    {
        rate_ = rate;
    }
    LoopParams &mutableParams()
    {
        return loop_params_;
    }

    lio::PointCloudXYZI::Ptr getSubMaps(std::vector<Pose6D> &pose_list,
                                        std::vector<lio::PointCloudXYZI::Ptr> &cloud_list,
                                        int index,
                                        int search_num)
    {
        lio::PointCloudXYZI::Ptr cloud(new lio::PointCloudXYZI);
        int max_size = pose_list.size();
        int min_index = std::max(0, index - search_num);
        int max_index = std::min(max_size - 1, index + search_num);
        for (int i = min_index; i <= max_index; i++)
        {
            Pose6D &p = pose_list[i];
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.block<3, 3>(0, 0) = p.global_rot;
            T.block<3, 1>(0, 3) = p.global_pos;
            lio::PointCloudXYZI::Ptr temp_cloud(new lio::PointCloudXYZI);
            pcl::transformPointCloud(*cloud_list[p.index], *temp_cloud, T);
            *cloud += *temp_cloud;
        }
        sub_map_downsize_filter_->setInputCloud(cloud);
        sub_map_downsize_filter_->filter(*cloud);
        return cloud;
    }

    void operator()()
    {
        while (ros::ok())
        {
            rate_->sleep();
            if (terminate_flag)
                break;
            if (!loop_params_.activate)
                continue;
            if (shared_data_->key_poses.size() < loop_params_.loop_pose_index_thresh)
                continue;
            if (!shared_data_->key_pose_added)
                continue;
            shared_data_->key_pose_added = false;
            {
                std::lock_guard<std::mutex> lock(shared_data_->mutex);
                lastest_index_ = shared_data_->key_poses.size() - 1;
                temp_poses_.clear();
                temp_poses_.assign(shared_data_->key_poses.begin(), shared_data_->key_poses.end());
            }

            loopCheck();
            addOdomFactor();
            addLoopFactor();
            smoothAndUpdate();
        }
    }

private:
    std::shared_ptr<SharedData> shared_data_;

    std::shared_ptr<ros::Rate> rate_;

    LoopParams loop_params_;

    std::vector<Pose6D> temp_poses_;

    int previous_index_ = 0;

    int lastest_index_;

    bool loop_found_ = false;

    gtsam::Values initialized_estimate_;

    gtsam::Values optimized_estimate_;

    std::shared_ptr<gtsam::ISAM2> isam2_;

    gtsam::NonlinearFactorGraph gtsam_graph_;

    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_history_poses_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_history_poses_;

    pcl::VoxelGrid<lio::PointType>::Ptr sub_map_downsize_filter_;

    pcl::IterativeClosestPoint<lio::PointType, lio::PointType>::Ptr icp_;

    void loopCheck()
    {
        if (temp_poses_.empty())
            return;
        int cur_index = temp_poses_.size() - 1;
        int pre_index = -1;

        cloud_history_poses_->clear();

        for (Pose6D &p : temp_poses_)
        {
            pcl::PointXYZ point;
            point.x = p.global_pos(0);
            point.y = p.global_pos(1);
            point.z = p.global_pos(2);
            cloud_history_poses_->push_back(point);
        }
        kdtree_history_poses_->setInputCloud(cloud_history_poses_);
        std::vector<int> ids;
        std::vector<float> sqdists;
        kdtree_history_poses_->radiusSearch(cloud_history_poses_->back(), loop_params_.loop_pose_search_radius, ids, sqdists, 0);

        for (int i = 0; i < ids.size(); i++)
        {
            int id = ids[i];
            if (std::abs(temp_poses_[id].time - temp_poses_.back().time) > loop_params_.time_thresh && std::abs(cur_index - id) >= loop_params_.loop_pose_index_thresh)
            {
                pre_index = id;
                break;
            }
        }
        if (pre_index == -1 || pre_index == cur_index)
            return;

        lio::PointCloudXYZI::Ptr cur_cloud = getSubMaps(temp_poses_, shared_data_->cloud_history, cur_index, 0);
        lio::PointCloudXYZI::Ptr sub_maps = getSubMaps(temp_poses_, shared_data_->cloud_history, pre_index, loop_params_.submap_search_num);
        
        icp_->setInputSource(cur_cloud);
        icp_->setInputTarget(sub_maps);

        lio::PointCloudXYZI::Ptr aligned(new lio::PointCloudXYZI);

        icp_->align(*aligned, Eigen::Matrix4f::Identity());

        float score = icp_->getFitnessScore();

        if (!icp_->hasConverged() || score > loop_params_.loop_icp_thresh)
            return;

        ROS_INFO("Detected LOOP: %d %d %f", pre_index, cur_index, score);
        shared_data_->loop_history.emplace_back(pre_index, cur_index);
        loop_found_ = true;

        Eigen::Matrix4d T_pre_cur = icp_->getFinalTransformation().cast<double>();
        Eigen::Matrix3d R12 = temp_poses_[pre_index].global_rot.transpose() * T_pre_cur.block<3, 3>(0, 0) * temp_poses_[cur_index].global_rot;
        Eigen::Vector3d t12 = temp_poses_[pre_index].global_rot.transpose() * (T_pre_cur.block<3, 3>(0, 0) * temp_poses_[cur_index].global_pos + T_pre_cur.block<3, 1>(0, 3) - temp_poses_[pre_index].global_pos);
        shared_data_->loop_pairs.emplace_back(pre_index, cur_index, score, R12, t12);
    }

    void addOdomFactor()
    {
        for (int i = previous_index_; i < lastest_index_; i++)
        {
            Pose6D &p1 = temp_poses_[i];
            Pose6D &p2 = temp_poses_[i + 1];

            if (i == 0)
            {
                initialized_estimate_.insert(i, gtsam::Pose3(gtsam::Rot3(p1.local_rot),
                                                             gtsam::Point3(p1.local_pos)));
                gtsam::noiseModel::Diagonal::shared_ptr noise = gtsam::noiseModel::Diagonal::Variances(gtsam::Vector6::Ones() * 1e-12);
                gtsam_graph_.add(gtsam::PriorFactor<gtsam::Pose3>(i, gtsam::Pose3(gtsam::Rot3(p1.local_rot), gtsam::Point3(p1.local_pos)), noise));
            }
            initialized_estimate_.insert(i + 1, gtsam::Pose3(gtsam::Rot3(p2.local_rot),
                                                             gtsam::Point3(p2.local_pos)));
            Eigen::Matrix3d R12 = p1.local_rot.transpose() * p2.local_rot;
            Eigen::Vector3d t12 = p1.local_rot.transpose() * (p2.local_pos - p1.local_pos);
            // ！！！！！！！！！！！carefully add this！！！！！！！！！！！
            if (loop_params_.z_prior)
            {
                gtsam::noiseModel::Diagonal::shared_ptr noise_prior = gtsam::noiseModel::Diagonal::Variances(gtsam::Vector1::Ones());
                gtsam_graph_.add(ZaxisPriorFactor(i + 1, noise_prior, p2.local_pos(2)));
            }

            gtsam::noiseModel::Diagonal::shared_ptr noise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-6).finished());
            gtsam_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(i,
                                                                i + 1,
                                                                gtsam::Pose3(gtsam::Rot3(R12), gtsam::Point3(t12)), noise));
        }
        previous_index_ = lastest_index_;
    }

    void addLoopFactor()
    {
        if (!loop_found_)
            return;
        if (shared_data_->loop_pairs.empty())
            return;
        for (LoopPair &lp : shared_data_->loop_pairs)
        {
            gtsam::Pose3 pose_between(gtsam::Rot3(lp.diff_rot), gtsam::Point3(lp.diff_pos));
            gtsam_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(lp.pre_idx,
                                                                lp.cur_idx,
                                                                pose_between,
                                                                gtsam::noiseModel::Diagonal::Variances(gtsam::Vector6::Ones() * lp.score)));
        }
        shared_data_->loop_pairs.clear();
    }

    void smoothAndUpdate()
    {
        isam2_->update(gtsam_graph_, initialized_estimate_);
        isam2_->update();
        if (loop_found_)
        {
            isam2_->update();
            isam2_->update();
            isam2_->update();
            isam2_->update();
            isam2_->update();
            loop_found_ = false;
        }
        gtsam_graph_.resize(0);
        initialized_estimate_.clear();

        optimized_estimate_ = isam2_->calculateBestEstimate();
        gtsam::Pose3 latest_estimate = optimized_estimate_.at<gtsam::Pose3>(lastest_index_);
        temp_poses_[lastest_index_].global_rot = latest_estimate.rotation().matrix().cast<double>();
        temp_poses_[lastest_index_].global_pos = latest_estimate.translation().matrix().cast<double>();
        Eigen::Matrix3d offset_rot;
        Eigen::Vector3d offset_pos;
        temp_poses_[lastest_index_].getOffset(offset_rot, offset_pos);

        shared_data_->mutex.lock();
        int current_size = shared_data_->key_poses.size();
        shared_data_->offset_rot = offset_rot;
        shared_data_->offset_pos = offset_pos;
        shared_data_->mutex.unlock();

        for (int i = 0; i < lastest_index_; i++)
        {
            gtsam::Pose3 temp_pose = optimized_estimate_.at<gtsam::Pose3>(i);
            shared_data_->key_poses[i].global_rot = temp_pose.rotation().matrix().cast<double>();
            shared_data_->key_poses[i].global_pos = temp_pose.translation().matrix().cast<double>();
        }

        for (int i = lastest_index_; i < current_size; i++)
        {
            shared_data_->key_poses[i].addOffset(offset_rot, offset_pos);
        }
    }
};

class MapBuilderROS
{
public:
    MapBuilderROS(tf2_ros::TransformBroadcaster &br, std::shared_ptr<SharedData> share_data) : br_(br)
    {
        shared_data_ = share_data;
        initPatams();
        initSubscribers();
        initPublishers();
        initServices();

        lio_builder_ = std::make_shared<lio::IGLIOBuilder>(lio_params_);
        loop_closure_.setRate(loop_rate_);
        loop_closure_.setShared(share_data);
        loop_closure_.init();
        loop_thread_ = std::make_shared<std::thread>(std::ref(loop_closure_));
    }

    void initPatams()
    {
        nh_.param<std::string>("map_frame", global_frame_, "map");
        nh_.param<std::string>("local_frame", local_frame_, "local");
        nh_.param<std::string>("body_frame", body_frame_, "body");
        nh_.param<std::string>("imu_topic", imu_data_.topic, "/livox/imu");
        nh_.param<std::string>("livox_topic", livox_data_.topic, "/livox/lidar");
        double local_rate, loop_rate;
        nh_.param<double>("local_rate", local_rate, 20.0);
        nh_.param<double>("loop_rate", loop_rate, 1.0);
        local_rate_ = std::make_shared<ros::Rate>(local_rate);
        loop_rate_ = std::make_shared<ros::Rate>(loop_rate);
        nh_.param<double>("lio_builder/scan_resolution", lio_params_.scan_resolution, 0.5);
        nh_.param<double>("lio_builder/map_resolution", lio_params_.map_resolution, 0.5);
        nh_.param<double>("lio_builder/point2plane_gain", lio_params_.point2plane_gain, 1000.0);
        nh_.param<double>("lio_builder/plane2plane_gain", lio_params_.plane2plane_gain, 100.0);
        int map_capacity, grid_capacity;
        nh_.param<int>("lio_builder/map_capacity", map_capacity, 5000000);
        nh_.param<int>("lio_builder/grid_capacity", grid_capacity, 20);

        lio_params_.map_capacity = static_cast<size_t>(map_capacity);
        lio_params_.grid_capacity = static_cast<size_t>(grid_capacity);
        nh_.param<bool>("lio_builder/align_gravity", lio_params_.align_gravity, true);
        nh_.param<bool>("lio_builder/extrinsic_est_en", lio_params_.extrinsic_est_en, false);
        nh_.param<std::vector<double>>("lio_builder/imu_ext_rot", lio_params_.imu_ext_rot, std::vector<double>());
        nh_.param<std::vector<double>>("lio_builder/imu_ext_pos", lio_params_.imu_ext_pos, std::vector<double>());
        int mode;
        nh_.param<int>("lio_builder/near_mode", mode, 1);
        switch (mode)
        {
        case 1:
            lio_params_.mode = lio::VoxelMap::MODE::NEARBY_1;
            break;
        case 2:
            lio_params_.mode = lio::VoxelMap::MODE::NEARBY_7;
            break;
        case 3:
            lio_params_.mode = lio::VoxelMap::MODE::NEARBY_19;
            break;
        case 4:
            lio_params_.mode = lio::VoxelMap::MODE::NEARBY_26;
            break;

        default:
            lio_params_.mode = lio::VoxelMap::MODE::NEARBY_1;
            break;
        }

        nh_.param<bool>("loop_closure/activate", loop_closure_.mutableParams().activate, true);
        nh_.param<double>("loop_closure/rad_thresh", loop_closure_.mutableParams().rad_thresh, 0.4);
        nh_.param<double>("loop_closure/dist_thresh", loop_closure_.mutableParams().dist_thresh, 2.5);
        nh_.param<double>("loop_closure/time_thresh", loop_closure_.mutableParams().time_thresh, 30.0);
        nh_.param<double>("loop_closure/loop_pose_search_radius", loop_closure_.mutableParams().loop_pose_search_radius, 10.0);
        nh_.param<int>("loop_closure/loop_pose_index_thresh", loop_closure_.mutableParams().loop_pose_index_thresh, 5);
        nh_.param<double>("loop_closure/submap_resolution", loop_closure_.mutableParams().submap_resolution, 0.2);
        nh_.param<int>("loop_closure/submap_search_num", loop_closure_.mutableParams().submap_search_num, 20);
        nh_.param<double>("loop_closure/loop_icp_thresh", loop_closure_.mutableParams().loop_icp_thresh, 0.3);
        nh_.param<bool>("loop_closure/z_prior", loop_closure_.mutableParams().z_prior, false);
    }

    void initSubscribers()
    {
        imu_sub_ = nh_.subscribe(imu_data_.topic, 1000, &ImuData::callback, &imu_data_);
        livox_sub_ = nh_.subscribe(livox_data_.topic, 1000, &LivoxData::callback, &livox_data_);
    }

    void initPublishers()
    {
        local_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("local_cloud", 1000);
        body_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("body_cloud", 1000);
        odom_pub_ = nh_.advertise<nav_msgs::Odometry>("slam_odom", 1000);
        loop_mark_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("loop_mark", 1000);

        local_path_pub_ = nh_.advertise<nav_msgs::Path>("local_path", 1000);
        global_path_pub_ = nh_.advertise<nav_msgs::Path>("global_path", 1000);
    }

    void initServices()
    {
        save_map_server_ = nh_.advertiseService("save_map", &MapBuilderROS::saveMapCallback, this);
    }

    void publishCloud(ros::Publisher &publisher, const sensor_msgs::PointCloud2 &cloud_to_pub)
    {
        if (publisher.getNumSubscribers() == 0)
            return;
        publisher.publish(cloud_to_pub);
    }

    void publishOdom(const nav_msgs::Odometry &odom_to_pub)
    {
        if (odom_pub_.getNumSubscribers() == 0)
            return;
        odom_pub_.publish(odom_to_pub);
    }

    void publishLocalPath()
    {
        if (local_path_pub_.getNumSubscribers() == 0)
            return;

        if (shared_data_->key_poses.empty())
            return;

        nav_msgs::Path path;
        path.header.frame_id = global_frame_;
        path.header.stamp = ros::Time().fromSec(current_time_);
        for (Pose6D &p : shared_data_->key_poses)
        {
            geometry_msgs::PoseStamped pose;
            pose.header.frame_id = global_frame_;
            pose.header.stamp = ros::Time().fromSec(current_time_);
            pose.pose.position.x = p.local_pos(0);
            pose.pose.position.y = p.local_pos(1);
            pose.pose.position.z = p.local_pos(2);
            Eigen::Quaterniond q(p.local_rot);
            pose.pose.orientation.x = q.x();
            pose.pose.orientation.y = q.y();
            pose.pose.orientation.z = q.z();
            pose.pose.orientation.w = q.w();
            path.poses.push_back(pose);
        }
        local_path_pub_.publish(path);
    }

    void publishGlobalPath()
    {
        if (global_path_pub_.getNumSubscribers() == 0)
            return;

        if (shared_data_->key_poses.empty())
            return;
        nav_msgs::Path path;
        path.header.frame_id = global_frame_;
        path.header.stamp = ros::Time().fromSec(current_time_);
        for (Pose6D &p : shared_data_->key_poses)
        {
            geometry_msgs::PoseStamped pose;
            pose.header.frame_id = global_frame_;
            pose.header.stamp = ros::Time().fromSec(current_time_);
            pose.pose.position.x = p.global_pos(0);
            pose.pose.position.y = p.global_pos(1);
            pose.pose.position.z = p.global_pos(2);
            Eigen::Quaterniond q(p.global_rot);
            pose.pose.orientation.x = q.x();
            pose.pose.orientation.y = q.y();
            pose.pose.orientation.z = q.z();
            pose.pose.orientation.w = q.w();
            path.poses.push_back(pose);
        }
        global_path_pub_.publish(path);
    }

    void publishLoopMark()
    {
        if (loop_mark_pub_.getNumSubscribers() == 0)
            return;
        if (shared_data_->loop_history.empty())
            return;
        visualization_msgs::MarkerArray marker_array;
        visualization_msgs::Marker nodes_marker;

        nodes_marker.header.frame_id = global_frame_;
        nodes_marker.header.stamp = ros::Time().fromSec(current_time_);
        nodes_marker.ns = "loop_nodes";
        nodes_marker.id = 0;
        nodes_marker.type = visualization_msgs::Marker::SPHERE_LIST;
        nodes_marker.action = visualization_msgs::Marker::ADD;
        nodes_marker.pose.orientation.w = 1.0;
        nodes_marker.scale.x = 0.3;
        nodes_marker.scale.y = 0.3;
        nodes_marker.scale.z = 0.3;
        nodes_marker.color.r = 1.0;
        nodes_marker.color.g = 0.8;
        nodes_marker.color.b = 0.0;
        nodes_marker.color.a = 1.0;

        visualization_msgs::Marker edges_marker;
        edges_marker.header.frame_id = global_frame_;
        edges_marker.header.stamp = ros::Time().fromSec(current_time_);
        edges_marker.ns = "loop_edges";
        edges_marker.id = 1;
        edges_marker.type = visualization_msgs::Marker::LINE_LIST;
        edges_marker.action = visualization_msgs::Marker::ADD;
        edges_marker.pose.orientation.w = 1.0;
        edges_marker.scale.x = 0.1;

        edges_marker.color.r = 0.0;
        edges_marker.color.g = 0.8;
        edges_marker.color.b = 0.0;
        edges_marker.color.a = 1.0;
        for (auto &p : shared_data_->loop_history)
        {
            Pose6D &p1 = shared_data_->key_poses[p.first];
            Pose6D &p2 = shared_data_->key_poses[p.second];
            geometry_msgs::Point point1;
            point1.x = p1.global_pos(0);
            point1.y = p1.global_pos(1);
            point1.z = p1.global_pos(2);
            geometry_msgs::Point point2;
            point2.x = p2.global_pos(0);
            point2.y = p2.global_pos(1);
            point2.z = p2.global_pos(2);
            nodes_marker.points.push_back(point1);
            nodes_marker.points.push_back(point2);
            edges_marker.points.push_back(point1);
            edges_marker.points.push_back(point2);
        }
        marker_array.markers.push_back(nodes_marker);
        marker_array.markers.push_back(edges_marker);
        loop_mark_pub_.publish(marker_array);
    }

    bool saveMapCallback(lio::SaveMap::Request &req, lio::SaveMap::Response &res)
    {
        std::string file_path = req.save_path;
        lio::PointCloudXYZI::Ptr cloud(new lio::PointCloudXYZI);
        for (Pose6D &p : shared_data_->key_poses)
        {
            lio::PointCloudXYZI::Ptr temp_cloud(new lio::PointCloudXYZI);
            pcl::transformPointCloud(*shared_data_->cloud_history[p.index],
                                     *temp_cloud,
                                     p.global_pos,
                                     Eigen::Quaterniond(p.global_rot));
            *cloud += *temp_cloud;
        }
        if (cloud->empty())
        {
            res.status = false;
            res.message = "Empty cloud!";
            return false;
        }
        res.status = true;
        res.message = "Save map success!";
        writer_.writeBinaryCompressed(file_path, *cloud);
        return true;
    }

    void addKeyPose()
    {
        int idx = shared_data_->key_poses.size();
        if (shared_data_->key_poses.empty())
        {
            std::lock_guard<std::mutex> lock(shared_data_->mutex);
            shared_data_->key_poses.emplace_back(idx, current_time_, current_state_.rot, current_state_.pos);
            shared_data_->key_poses.back().addOffset(shared_data_->offset_rot, shared_data_->offset_pos);
            // shared_data_->key_poses.back().gravity = current_state_.get_g();
            shared_data_->key_pose_added = true;
            shared_data_->cloud_history.push_back(lio_builder_->cloudUndistortedBody());
            return;
        }
        Pose6D &last_key_pose = shared_data_->key_poses.back();
        Eigen::Matrix3d diff_rot = last_key_pose.local_rot.transpose() * current_state_.rot;
        Eigen::Vector3d diff_pose = last_key_pose.local_rot.transpose() * (current_state_.pos - last_key_pose.local_pos);
        Eigen::Vector3d rpy = rotate2rpy(diff_rot);
        if (diff_pose.norm() > loop_closure_.mutableParams().dist_thresh ||
            std::abs(rpy(0)) > loop_closure_.mutableParams().rad_thresh ||
            std::abs(rpy(1)) > loop_closure_.mutableParams().rad_thresh ||
            std::abs(rpy(2)) > loop_closure_.mutableParams().rad_thresh)
        {
            std::lock_guard<std::mutex> lock(shared_data_->mutex);
            shared_data_->key_poses.emplace_back(idx, current_time_, current_state_.rot, current_state_.pos);
            shared_data_->key_poses.back().addOffset(shared_data_->offset_rot, shared_data_->offset_pos);
            // shared_data_->key_poses.back().gravity = current_state_.get_g();
            shared_data_->key_pose_added = true;
            shared_data_->cloud_history.push_back(lio_builder_->cloudUndistortedBody());
        }
    }

    void run()
    {
        while (ros::ok())
        {
            local_rate_->sleep();
            ros::spinOnce();
            if (terminate_flag)
                break;
            if (!measure_group_.syncPackage(imu_data_, livox_data_))
                continue;
            lio_builder_->mapping(measure_group_);
            if (lio_builder_->currentStatus() == lio::Status::INITIALIZE)
                continue;
            current_time_ = measure_group_.lidar_time_end;
            current_state_ = lio_builder_->currentState();
            br_.sendTransform(eigen2Transform(shared_data_->offset_rot,
                                              shared_data_->offset_pos,
                                              global_frame_,
                                              local_frame_,
                                              current_time_));
            br_.sendTransform(eigen2Transform(current_state_.rot,
                                              current_state_.pos,
                                              local_frame_,
                                              body_frame_,
                                              current_time_));

            publishOdom(eigen2Odometry(current_state_.rot,
                                       current_state_.pos,
                                       local_frame_,
                                       body_frame_,
                                       current_time_));

            addKeyPose();

            publishCloud(body_cloud_pub_,
                         pcl2msg(lio_builder_->cloudUndistortedBody(),
                                 body_frame_,
                                 current_time_));
            publishCloud(local_cloud_pub_,
                         pcl2msg(lio_builder_->cloudWorld(),
                                 local_frame_,
                                 current_time_));
            publishLocalPath();
            publishGlobalPath();
            publishLoopMark();
        }

        loop_thread_->join();
        std::cout << "MAPPING NODE IS DOWN!" << std::endl;
    }

private:
    ros::NodeHandle nh_;
    std::string global_frame_;
    std::string local_frame_;
    std::string body_frame_;
    double current_time_;
    kf::State current_state_;
    ImuData imu_data_;
    LivoxData livox_data_;
    MeasureGroup measure_group_;
    lio::IGLIOParams lio_params_;
    std::shared_ptr<lio::IGLIOBuilder> lio_builder_;
    std::shared_ptr<SharedData> shared_data_;
    std::shared_ptr<ros::Rate> local_rate_;
    std::shared_ptr<ros::Rate> loop_rate_;
    LoopClosureThread loop_closure_;
    std::shared_ptr<std::thread> loop_thread_;

    tf2_ros::TransformBroadcaster &br_;

    ros::Subscriber imu_sub_;

    ros::Subscriber livox_sub_;

    ros::Publisher body_cloud_pub_;

    ros::Publisher local_cloud_pub_;

    ros::Publisher odom_pub_;

    ros::Publisher loop_mark_pub_;

    ros::Publisher local_path_pub_;

    ros::Publisher global_path_pub_;

    ros::ServiceServer save_map_server_;

    pcl::PCDWriter writer_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "map_builder_node");
    tf2_ros::TransformBroadcaster br;
    signal(SIGINT, signalHandler);
    std::shared_ptr<SharedData> share_date = std::make_shared<SharedData>();
    MapBuilderROS map_builder(br, share_date);
    map_builder.run();
    return 0;
}