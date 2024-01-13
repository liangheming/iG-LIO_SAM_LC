#pragma once
#include <mutex>
#include <vector>
#include <queue>
#include <string>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <livox_ros_driver2/CustomMsg.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/TransformStamped.h>
namespace lio
{
#define NUM_MATCH_POINTS (5)
#define NUM_MAX_POINTS (10000)
    enum Status
    {
        INITIALIZE,
        RELOCALIZATION,
        LOCALIZATION,
        MAPPING
    };
    using PointType = pcl::PointXYZINormal;
    using PointCloudXYZI = pcl::PointCloud<PointType>;

    typedef std::vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;
    struct IMU
    {
        IMU() : acc(Eigen::Vector3d::Zero()), gyro(Eigen::Vector3d::Zero()) {}
        IMU(double t, Eigen::Vector3d a, Eigen::Vector3d g)
            : timestamp(t), acc(a), gyro(g) {}
        IMU(double t, double a1, double a2, double a3, double g1, double g2, double g3)
            : timestamp(t), acc(a1, a2, a3), gyro(g1, g2, g3) {}
        double timestamp;
        Eigen::Vector3d acc;
        Eigen::Vector3d gyro;
    };

    bool esti_plane(Eigen::Vector4d &out, const PointVector &points, const double &thresh);

    bool esti_plane(Eigen::Vector4d &out, const std::vector<Eigen::Vector3d> &points, const double &thresh, bool none);

    float sq_dist(const PointType &p1, const PointType &p2);
}

struct ImuData
{
    std::string topic;
    std::mutex mutex;
    std::deque<lio::IMU> buffer;
    double last_timestamp = 0;
    void callback(const sensor_msgs::Imu::ConstPtr &msg);
};

struct LivoxData
{
    std::string topic;
    std::mutex mutex;
    std::deque<lio::PointCloudXYZI::Ptr> buffer;
    std::deque<double> time_buffer;
    double blind = 0.5;
    int filter_num = 3;
    double last_timestamp = 0;
    void callback(const livox_ros_driver2::CustomMsg::ConstPtr &msg);
    void livox2pcl(const livox_ros_driver2::CustomMsg::ConstPtr &msg, lio::PointCloudXYZI::Ptr &out);
};

struct MeasureGroup
{
    double lidar_time_begin = 0.0;
    double lidar_time_end = 0.0;
    bool lidar_pushed = false;
    lio::PointCloudXYZI::Ptr lidar;
    std::deque<lio::IMU> imus;
    bool syncPackage(ImuData &imu_data, LivoxData &livox_data);
};

sensor_msgs::PointCloud2 pcl2msg(lio::PointCloudXYZI::Ptr inp, std::string &frame_id, const double &timestamp);

nav_msgs::Odometry eigen2Odometry(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp);

geometry_msgs::TransformStamped eigen2Transform(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp);

Eigen::Vector3d rotate2rpy(Eigen::Matrix3d &rot);