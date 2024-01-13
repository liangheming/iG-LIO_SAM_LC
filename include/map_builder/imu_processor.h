#pragma once
#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "ieskf/ieskf.h"
#include "commons.h"

namespace lio
{
    struct Pose
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d acc;
        Eigen::Vector3d gyro;
        Eigen::Matrix3d rot;
        Eigen::Vector3d pos;
        Eigen::Vector3d vel;
        Pose();
        Pose(double t, Eigen::Vector3d a, Eigen::Vector3d g, Eigen::Vector3d v, Eigen::Vector3d p, Eigen::Matrix3d r)
            : offset(t), acc(a), gyro(g), vel(v), pos(p), rot(r) {}
        double offset;
    };

    class IMUProcessor
    {
    public:
        IMUProcessor(std::shared_ptr<kf::IESKF> kf);

        void init(const MeasureGroup &meas);

        void undistortPointcloud(const MeasureGroup &meas, PointCloudXYZI::Ptr &out);

        bool operator()(const MeasureGroup &meas, PointCloudXYZI::Ptr &out);

        bool isInitialized() const { return init_flag_; }

        void setMaxInitCount(int max_init_count) { max_init_count_ = max_init_count; }

        void setExtParams(Eigen::Matrix3d &rot_ext, Eigen::Vector3d &pos_ext);

        void setAccCov(Eigen::Vector3d acc_cov) { acc_cov_ = acc_cov; }

        void setGyroCov(Eigen::Vector3d gyro_cov) { gyro_cov_ = gyro_cov; }

        void setAccBiasCov(Eigen::Vector3d acc_bias_cov) { acc_bias_cov_ = acc_bias_cov; }

        void setGyroBiasCov(Eigen::Vector3d gyro_bias_cov) { gyro_bias_cov_ = gyro_bias_cov; }

        void setCov(Eigen::Vector3d gyro_cov, Eigen::Vector3d acc_cov, Eigen::Vector3d gyro_bias_cov, Eigen::Vector3d acc_bias_cov);

        void setCov(double gyro_cov, double acc_cov, double gyro_bias_cov, double acc_bias_cov);

        void setAlignGravity(bool align_gravity) { align_gravity_ = align_gravity; }

        void reset();

    private:
        int init_count_ = 0;
        int max_init_count_ = 20;
        Eigen::Matrix3d rot_ext_;
        Eigen::Vector3d pos_ext_;
        std::shared_ptr<kf::IESKF> kf_;

        IMU last_imu_;
        bool init_flag_ = false;
        bool align_gravity_ = true;

        Eigen::Vector3d mean_acc_;
        Eigen::Vector3d mean_gyro_;

        Eigen::Vector3d last_acc_;
        Eigen::Vector3d last_gyro_;

        std::vector<Pose> imu_poses_;

        double last_lidar_time_end_;

        Eigen::Vector3d gyro_cov_;
        Eigen::Vector3d acc_cov_;
        Eigen::Vector3d gyro_bias_cov_;
        Eigen::Vector3d acc_bias_cov_;

        kf::Matrix12d Q_;
    };
} // namespace fastlio
