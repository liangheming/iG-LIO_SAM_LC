#include "map_builder/iglio_builder.h"
// #include <chrono>
namespace lio
{
    inline void CauchyLossFunction(const double e, const double delta, Eigen::Vector3d &rho)
    {
        double dsqr = delta * delta;
        if (e <= dsqr)
        { // inlier
            rho[0] = e;
            rho[1] = 1.;
            rho[2] = 0.;
        }
        else
        {
            double sqrte = std::sqrt(e);
            rho[0] = 2 * sqrte * delta - dsqr;
            rho[1] = delta / sqrte;
            rho[2] = -0.5 * rho[1] / e;
        }
    }

    IGLIOBuilder::IGLIOBuilder(IGLIOParams &params) : params_(params)
    {
        kf_ = std::make_shared<kf::IESKF>(params_.esikf_max_iteration);
        kf_->set_share_function(
            [this](kf::State &s, kf::SharedState &d)
            { sharedUpdateFunc(s, d); });

        // 初始化IMUProcessor
        imu_processor_ = std::make_shared<IMUProcessor>(kf_);
        imu_processor_->setCov(params.imu_gyro_cov, params.imu_acc_cov, params.imu_gyro_bias_cov, params.imu_acc_bias_cov);
        Eigen::Matrix3d rot_ext;
        Eigen::Vector3d pos_ext;
        rot_ext << params.imu_ext_rot[0], params.imu_ext_rot[1], params.imu_ext_rot[2],
            params.imu_ext_rot[3], params.imu_ext_rot[4], params.imu_ext_rot[5],
            params.imu_ext_rot[6], params.imu_ext_rot[7], params.imu_ext_rot[8];
        pos_ext << params.imu_ext_pos[0], params.imu_ext_pos[1], params.imu_ext_pos[2];
        imu_processor_->setExtParams(rot_ext, pos_ext);
        imu_processor_->setAlignGravity(params.align_gravity);

        fast_voxel_map_ = std::make_shared<FastVoxelMap>(params.scan_resolution);
        voxel_map_ = std::make_shared<VoxelMap>(params.map_resolution, params.map_capacity, params.grid_capacity, params.mode);
        point_array_lidar_.reserve(params.max_points_per_scan);
        cache_flag_.reserve(params.max_points_per_scan);
        fastlio_cache_.reserve(params.max_points_per_scan);
        gicp_cache_.reserve(params_.max_points_per_scan * voxel_map_->searchRange().size());

        cloud_body_.reset(new PointCloudXYZI);
    }

    void IGLIOBuilder::mapping(const MeasureGroup &meas)
    {
        if (!imu_processor_->operator()(meas, cloud_lidar_))
            return;

        if (status == Status::INITIALIZE)
        {
            // 初始化VoxelMap
            cloud_world_ = transformToWorld(cloud_lidar_);
            voxel_map_->addCloud(cloud_world_);
            frame_count_++;
            key_frame_count_++;
            key_rot_ = kf_->x().rot;
            key_pos_ = kf_->x().pos;
            status = Status::MAPPING;
            return;
        }
        fast_voxel_map_->filter(cloud_lidar_, point_array_lidar_);
        // auto tic = std::chrono::system_clock::now();
        kf_->update();
        // auto toc = std::chrono::system_clock::now();
        // std::chrono::duration<double> duration = toc - tic;
        // std::cout << duration.count() * 1000 << std::endl;

        frame_count_++;
        if (frame_count_ < 10)
        {
            cloud_world_ = transformToWorld(cloud_lidar_);
            voxel_map_->addCloud(cloud_world_);
            key_rot_ = kf_->x().rot;
            key_pos_ = kf_->x().pos;
            return;
        }

        if (cloud_lidar_->size() < 1000 || Sophus::SO3d(kf_->x().rot.transpose() * key_rot_).log().norm() > 0.18 || (kf_->x().pos - key_pos_).norm() > 0.5)
        {
            key_frame_count_++;
            cloud_world_ = transformToWorld(cloud_lidar_);
            voxel_map_->addCloud(cloud_world_);
            key_rot_ = kf_->x().rot;
            key_pos_ = kf_->x().pos;
        }
    }

    void IGLIOBuilder::sharedUpdateFunc(kf::State &state, kf::SharedState &shared_state)
    {
        if (key_frame_count_ >= 20)
        {
            gicpConstraint(state, shared_state);
        }
        else
        {
            fastlioConstraint(state, shared_state);
        }
    }

    void IGLIOBuilder::fastlioConstraint(kf::State &state, kf::SharedState &shared_state)
    {
        int size = point_array_lidar_.size();
        if (shared_state.iter_num < 3)
        {
            for (int i = 0; i < size; i++)
            {
                Eigen::Vector3d p_lidar = point_array_lidar_[i].point;
                Eigen::Vector3d p_body = state.rot_ext * p_lidar + state.pos_ext;
                Eigen::Vector3d p_world = state.rot * p_body + state.pos;
                std::vector<Eigen::Vector3d> nearest_points;
                nearest_points.reserve(5);
                voxel_map_->searchKNN(p_world, 5, 5.0, nearest_points);
                Eigen::Vector4d pabcd;
                cache_flag_[i] = false;
                if (nearest_points.size() >= 3 && esti_plane(pabcd, nearest_points, 0.1, false))
                {
                    double pd2 = pabcd(0) * p_world(0) + pabcd(1) * p_world(1) + pabcd(2) * p_world(2) + pabcd(3);
                    // 和点面距离正相关，和点的远近距离负相关
                    double s = 1 - 0.9 * std::fabs(pd2) / std::sqrt(point_array_lidar_[i].point.norm());
                    if (s > 0.9)
                    {
                        cache_flag_[i] = true;
                        fastlio_cache_[i].point = p_lidar;
                        fastlio_cache_[i].plane(0) = pabcd(0);
                        fastlio_cache_[i].plane(1) = pabcd(1);
                        fastlio_cache_[i].plane(2) = pabcd(2);
                        fastlio_cache_[i].plane(3) = pd2;
                    }
                }
            }
        }
        int effect_feat_num = 0;
        shared_state.H.setZero();
        shared_state.b.setZero();
        Eigen::Matrix<double, 1, 12> J;
        for (int i = 0; i < size; i++)
        {
            if (!cache_flag_[i])
                continue;
            effect_feat_num++;
            J.setZero();
            Eigen::Vector3d norm_vec = fastlio_cache_[i].plane.segment<3>(0);
            double error = fastlio_cache_[i].plane(3);
            Eigen::Vector3d p_lidar = fastlio_cache_[i].point;
            Eigen::Matrix<double, 1, 3> B = -norm_vec.transpose() * state.rot * Sophus::SO3d::hat(state.rot_ext * p_lidar + state.pos_ext);
            J.block<1, 3>(0, 0) = norm_vec.transpose();
            J.block<1, 3>(0, 3) = B;
            if (params_.extrinsic_est_en)
            {
                Eigen::Matrix<double, 1, 3> C = -norm_vec.transpose() * state.rot * state.rot_ext * Sophus::SO3d::hat(p_lidar);
                Eigen::Matrix<double, 1, 3> D = norm_vec.transpose() * state.rot;
                J.block<1, 3>(0, 6) = C;
                J.block<1, 3>(0, 9) = D;
            }
            shared_state.H += J.transpose() * params_.point2plane_gain * J;
            shared_state.b += J.transpose() * params_.point2plane_gain * error;
        }

        if (effect_feat_num < 1)
            std::cout << "NO EFFECTIVE POINTS!" << std::endl;
    }

    void IGLIOBuilder::gicpConstraint(kf::State &state, kf::SharedState &shared_state)
    {
        int size = point_array_lidar_.size();
        if (shared_state.iter_num < 3)
        {
            gicp_cache_.clear();
            Eigen::Vector3d mean_B = Eigen::Vector3d::Zero();
            Eigen::Matrix3d cov_B = Eigen::Matrix3d::Zero();
            for (int i = 0; i < size; i++)
            {
                Eigen::Vector3d p_lidar = point_array_lidar_[i].point;
                Eigen::Vector3d p_world = state.rot * (state.rot_ext * p_lidar + state.pos_ext) + state.pos;
                Eigen::Matrix3d cov_A = point_array_lidar_[i].cov;

                for (Eigen::Vector3d &r : voxel_map_->searchRange())
                {
                    Eigen::Vector3d pw_near = p_world + r;
                    if (voxel_map_->getCentroidAndCovariance(pw_near, mean_B, cov_B) && voxel_map_->isSameGrid(pw_near, mean_B))
                    {
                        gicp_cache_.emplace_back(p_lidar, mean_B, cov_A, cov_B);
                    }
                }
            }
        }
        shared_state.H.setZero();
        shared_state.b.setZero();

        Eigen::Matrix<double, 3, 12> J;
        for (int i = 0; i < gicp_cache_.size(); i++)
        {
            GICPCorrespond &gicp_corr = gicp_cache_[i];
            Eigen::Vector3d p_lidar = gicp_corr.meanA;
            Eigen::Vector3d p_body = state.rot_ext * p_lidar + state.pos_ext;
            Eigen::Vector3d mean_A = state.rot * p_body + state.pos;
            Eigen::Matrix3d omiga = (gicp_corr.covB + state.rot * state.rot_ext * gicp_corr.covA * state.rot_ext.transpose() * state.rot.transpose()).inverse();
            Eigen::Vector3d error = gicp_corr.meanB - mean_A;
            double chi2_error = error.transpose() * omiga * error;
            if (shared_state.iter_num > 2 && chi2_error > 7.815)
                continue;
            Eigen::Vector3d rho;
            CauchyLossFunction(chi2_error, 10.0, rho);
            J.setZero();
            J.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
            J.block<3, 3>(0, 3) = state.rot * Sophus::SO3d::hat(p_body);
            if (params_.extrinsic_est_en)
            {
                J.block<3, 3>(0, 6) = state.rot * state.rot_ext * Sophus::SO3d::hat(p_lidar);
                J.block<3, 3>(0, 9) = -state.rot;
            }
            Eigen::Matrix3d robust_information_matrix = params_.plane2plane_gain * (rho[1] * omiga + 2.0 * rho[2] * omiga * error * error.transpose() * omiga);
            shared_state.H += (J.transpose() * robust_information_matrix * J);
            shared_state.b += (params_.plane2plane_gain * rho[1] * J.transpose() * omiga * error);
        }

        if (gicp_cache_.size() < 1)
            std::cout << "NO EFFECTIVE POINTS!" << std::endl;
    }

    PointCloudXYZI::Ptr IGLIOBuilder::transformToWorld(const PointCloudXYZI::Ptr cloud)
    {
        PointCloudXYZI::Ptr cloud_world(new PointCloudXYZI);
        Eigen::Matrix3d rot = kf_->x().rot;
        Eigen::Vector3d pos = kf_->x().pos;
        Eigen::Matrix3d rot_ext = kf_->x().rot_ext;
        Eigen::Vector3d pos_ext = kf_->x().pos_ext;
        cloud_world->reserve(cloud->size());
        for (auto &p : cloud->points)
        {
            Eigen::Vector3d point(p.x, p.y, p.z);
            point = rot * (rot_ext * point + pos_ext) + pos;
            PointType p_world;
            p_world.x = point(0);
            p_world.y = point(1);
            p_world.z = point(2);
            p_world.intensity = p.intensity;
            cloud_world->points.push_back(p_world);
        }
        return cloud_world;
    }

    PointCloudXYZI::Ptr IGLIOBuilder::cloudUndistortedBody()
    {
        PointCloudXYZI::Ptr cloud_body(new PointCloudXYZI);
        pcl::transformPointCloud(*cloud_lidar_, *cloud_body, kf_->x().pos_ext, Eigen::Quaterniond(kf_->x().rot_ext));
        return cloud_body;
    }

    PointCloudXYZI::Ptr IGLIOBuilder::cloudWorld()
    {
        return transformToWorld(cloud_lidar_);
    }

    void IGLIOBuilder::reset()
    {
        status = Status::INITIALIZE;
        imu_processor_->reset();
        kf::State state = kf_->x();
        state.rot.setIdentity();
        state.pos.setZero();
        state.vel.setZero();
        state.rot_ext.setIdentity();
        state.pos_ext.setZero();
        state.ba.setZero();
        state.bg.setZero();
        kf_->change_x(state);
        voxel_map_->reset();
    }
} // namespace lio
