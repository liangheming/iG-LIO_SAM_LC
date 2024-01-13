#include "map_builder/fastlio_builder.h"
// #include <chrono>
namespace lio
{
    /**
     * @brief
     */
    FastLIOBuilder::FastLIOBuilder(FASTLIOParams &params) : params_(params)
    { // 初始化ESIKF
        kf_ = std::make_shared<kf::IESKF>();
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
        // 初始化KDTree
        ikdtree_ = std::make_shared<KD_TREE<PointType>>();
        ikdtree_->set_downsample_param(params.resolution);
        // 初始化点云采样器
        down_size_filter_.setLeafSize(params.resolution, params.resolution, params.resolution);

        // 初始化local_map
        local_map_.cube_len = params.cube_len;
        local_map_.det_range = params.det_range;

        extrinsic_est_en_ = params.extrinsic_est_en;

        cloud_down_lidar_.reset(new PointCloudXYZI);

        cloud_down_world_.reset(new PointCloudXYZI(NUM_MAX_POINTS, 1));
        norm_vec_.reset(new PointCloudXYZI(NUM_MAX_POINTS, 1));

        effect_cloud_lidar_.reset(new PointCloudXYZI(NUM_MAX_POINTS, 1));
        effect_norm_vec_.reset(new PointCloudXYZI(NUM_MAX_POINTS, 1));

        nearest_points_.resize(NUM_MAX_POINTS);
        point_selected_flag_.resize(NUM_MAX_POINTS, false);
    }

    /**
     * @brief 将点云转换到世界坐标系下
     * @param cloud: lidar系下的点云
     *
     */
    PointCloudXYZI::Ptr FastLIOBuilder::transformToWorld(const PointCloudXYZI::Ptr cloud)
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

    /**
     * @brief
     */
    void FastLIOBuilder::mapping(const MeasureGroup &meas)
    {
        if (!imu_processor_->operator()(meas, cloud_undistorted_lidar_))
            return;

        down_size_filter_.setInputCloud(cloud_undistorted_lidar_);
        down_size_filter_.filter(*cloud_down_lidar_);

        if (status == Status::INITIALIZE)
        {
            // 初始化ikd_tree
            PointCloudXYZI::Ptr point_world = transformToWorld(cloud_down_lidar_);
            ikdtree_->Build(point_world->points);
            status = Status::MAPPING;
            return;
        }

        trimMap();
        // auto tic = std::chrono::system_clock::now();
        kf_->update();
        // auto toc = std::chrono::system_clock::now();
        // std::chrono::duration<double> duration = toc - tic;
        // std::cout << duration.count() * 1000 << std::endl;
        increaseMap();
    }

    /**
     * @brief 根据局部地图的范围修建ikdtree中点云
     */
    void FastLIOBuilder::trimMap()
    {
        local_map_.cub_to_rm.clear();
        kf::State state = kf_->x();
        Eigen::Vector3d pos_lidar = state.pos + state.rot * state.pos_ext;
        // 根据lidar的位置进行局部地图的初始化
        if (!local_map_.is_initialed)
        {
            for (int i = 0; i < 3; i++)
            {
                local_map_.local_map_corner.vertex_min[i] = pos_lidar[i] - local_map_.cube_len / 2.0;
                local_map_.local_map_corner.vertex_max[i] = pos_lidar[i] + local_map_.cube_len / 2.0;
            }
            local_map_.is_initialed = true;
            return;
        }

        float dist_to_map_edge[3][2];
        bool need_move = false;
        double det_thresh = local_map_.move_thresh * local_map_.det_range;
        // 如果靠近地图边缘 则需要进行地图的移动
        for (int i = 0; i < 3; i++)
        {
            dist_to_map_edge[i][0] = fabs(pos_lidar(i) - local_map_.local_map_corner.vertex_min[i]);
            dist_to_map_edge[i][1] = fabs(pos_lidar(i) - local_map_.local_map_corner.vertex_max[i]);

            if (dist_to_map_edge[i][0] <= det_thresh || dist_to_map_edge[i][1] <= det_thresh)
                need_move = true;
        }
        if (!need_move)
            return;

        BoxPointType new_corner, temp_corner;
        new_corner = local_map_.local_map_corner;
        float mov_dist = std::max((local_map_.cube_len - 2.0 * local_map_.move_thresh * local_map_.det_range) * 0.5 * 0.9,
                                  double(local_map_.det_range * (local_map_.move_thresh - 1)));
        // 更新局部地图
        for (int i = 0; i < 3; i++)
        {
            temp_corner = local_map_.local_map_corner;
            if (dist_to_map_edge[i][0] <= det_thresh)
            {
                new_corner.vertex_max[i] -= mov_dist;
                new_corner.vertex_min[i] -= mov_dist;
                temp_corner.vertex_min[i] = local_map_.local_map_corner.vertex_max[i] - mov_dist;
                local_map_.cub_to_rm.push_back(temp_corner);
            }
            else if (dist_to_map_edge[i][1] <= det_thresh)
            {
                new_corner.vertex_max[i] += mov_dist;
                new_corner.vertex_min[i] += mov_dist;
                temp_corner.vertex_max[i] = local_map_.local_map_corner.vertex_min[i] + mov_dist;
                local_map_.cub_to_rm.push_back(temp_corner);
            }
        }
        local_map_.local_map_corner = new_corner;
        // 强制删除历史点云
        PointVector points_history;
        ikdtree_->acquire_removed_points(points_history);

        // 删除局部地图之外的点云
        if (local_map_.cub_to_rm.size() > 0)
            ikdtree_->Delete_Point_Boxes(local_map_.cub_to_rm);
        return;
    }

    /**
     * @brief
     */
    void FastLIOBuilder::increaseMap()
    {
        if (status == Status::INITIALIZE)
            return;
        if (cloud_down_lidar_->empty())
            return;

        int size = cloud_down_lidar_->size();

        PointVector point_to_add;
        PointVector point_no_need_downsample;

        point_to_add.reserve(size);
        point_no_need_downsample.reserve(size);

        kf::State state = kf_->x();
        for (int i = 0; i < size; i++)
        {
            const PointType &p = cloud_down_lidar_->points[i];
            Eigen::Vector3d point(p.x, p.y, p.z);
            point = state.rot * (state.rot_ext * point + state.pos_ext) + state.pos;
            cloud_down_world_->points[i].x = point(0);
            cloud_down_world_->points[i].y = point(1);
            cloud_down_world_->points[i].z = point(2);
            cloud_down_world_->points[i].intensity = cloud_down_lidar_->points[i].intensity;
            // 如果该点附近没有近邻点则需要添加到地图中
            if (nearest_points_[i].empty())
            {
                point_to_add.push_back(cloud_down_world_->points[i]);
                continue;
            }

            const PointVector &points_near = nearest_points_[i];
            bool need_add = true;
            PointType downsample_result, mid_point;
            mid_point.x = std::floor(cloud_down_world_->points[i].x / params_.resolution) * params_.resolution + 0.5 * params_.resolution;
            mid_point.y = std::floor(cloud_down_world_->points[i].y / params_.resolution) * params_.resolution + 0.5 * params_.resolution;
            mid_point.z = std::floor(cloud_down_world_->points[i].z / params_.resolution) * params_.resolution + 0.5 * params_.resolution;

            // 如果该点所在的voxel没有点，则直接加入地图，且不需要降采样
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * params_.resolution && fabs(points_near[0].y - mid_point.y) > 0.5 * params_.resolution && fabs(points_near[0].z - mid_point.z) > 0.5 * params_.resolution)
            {
                point_no_need_downsample.push_back(cloud_down_world_->points[i]);
                continue;
            }
            float dist = sq_dist(cloud_down_world_->points[i], mid_point);

            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
            {
                // 如果该点的近邻点较少，则需要加入到地图中
                if (points_near.size() < NUM_MATCH_POINTS)
                    break;
                // 如果该点的近邻点距离voxel中心点的距离比该点距离voxel中心点更近，则不需要加入该点
                if (sq_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                point_to_add.push_back(cloud_down_world_->points[i]);
        }
        int add_point_size = ikdtree_->Add_Points(point_to_add, true);
        ikdtree_->Add_Points(point_no_need_downsample, false);
    }

    /**
     * @brief
     */
    void FastLIOBuilder::sharedUpdateFunc(kf::State &state, kf::SharedState &share_data)
    {
        int size = cloud_down_lidar_->size();
#ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for

#endif
        for (int i = 0; i < size; i++)
        {
            PointType &point_body = cloud_down_lidar_->points[i];
            PointType &point_world = cloud_down_world_->points[i];
            Eigen::Vector3d point_body_vec(point_body.x, point_body.y, point_body.z);
            Eigen::Vector3d point_world_vec = state.rot * (state.rot_ext * point_body_vec + state.pos_ext) + state.pos;
            point_world.x = point_world_vec(0);
            point_world.y = point_world_vec(1);
            point_world.z = point_world_vec(2);
            point_world.intensity = point_body.intensity;

            std::vector<float> point_sq_dist(NUM_MATCH_POINTS);
            auto &points_near = nearest_points_[i];

            ikdtree_->Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, point_sq_dist);
            if (points_near.size() >= NUM_MATCH_POINTS && point_sq_dist[NUM_MATCH_POINTS - 1] <= 5)
                point_selected_flag_[i] = true;
            else
                point_selected_flag_[i] = false;

            if (!point_selected_flag_[i])
                continue;

            Eigen::Vector4d pabcd;
            point_selected_flag_[i] = false;

            // 估计平面法向量，同时计算点面距离，计算的值存入intensity
            if (esti_plane(pabcd, points_near, 0.1))
            {
                double pd2 = pabcd(0) * point_world_vec(0) + pabcd(1) * point_world_vec(1) + pabcd(2) * point_world_vec(2) + pabcd(3);
                // 和点面距离正相关，和点的远近距离负相关
                double s = 1 - 0.9 * std::fabs(pd2) / std::sqrt(point_body_vec.norm());
                if (s > 0.9)
                {
                    point_selected_flag_[i] = true;
                    norm_vec_->points[i].x = pabcd(0);
                    norm_vec_->points[i].y = pabcd(1);
                    norm_vec_->points[i].z = pabcd(2);
                    norm_vec_->points[i].intensity = pd2;
                }
            }
        }

        int effect_feat_num = 0;
        for (int i = 0; i < size; i++)
        {
            if (!point_selected_flag_[i])
                continue;
            effect_cloud_lidar_->points[effect_feat_num] = cloud_down_lidar_->points[i];
            effect_norm_vec_->points[effect_feat_num] = norm_vec_->points[i];
            effect_feat_num++;
        }

        share_data.H.setZero();
        share_data.b.setZero();
        if (effect_feat_num < 1)
        {
            ROS_INFO("NO Effective Points!");
            return;
        }
        Eigen::Matrix<double, 1, 12> J;
        for (int i = 0; i < effect_feat_num; i++)
        {
            J.setZero();
            const PointType &laser_p = effect_cloud_lidar_->points[i];
            const PointType &norm_p = effect_norm_vec_->points[i];
            Eigen::Vector3d laser_p_vec(laser_p.x, laser_p.y, laser_p.z);
            Eigen::Vector3d norm_vec(norm_p.x, norm_p.y, norm_p.z);
            Eigen::Matrix<double, 1, 3> B = -norm_vec.transpose() * state.rot * Sophus::SO3d::hat(state.rot_ext * laser_p_vec + state.pos_ext);
            J.block<1, 3>(0, 0) = norm_vec.transpose();
            J.block<1, 3>(0, 3) = B;

            if (extrinsic_est_en_)
            {
                Eigen::Matrix<double, 1, 3> C = -norm_vec.transpose() * state.rot * state.rot_ext * Sophus::SO3d::hat(laser_p_vec);
                Eigen::Matrix<double, 1, 3> D = norm_vec.transpose() * state.rot;
                J.block<1, 3>(0, 6) = C;
                J.block<1, 3>(0, 9) = D;
            }
            share_data.H += J.transpose() * 1000 * J;
            share_data.b += J.transpose() * 1000 * norm_p.intensity;
        }
    }

    PointCloudXYZI::Ptr FastLIOBuilder::cloudUndistortedBody()
    {
        PointCloudXYZI::Ptr cloud_undistorted_body(new PointCloudXYZI);
        Eigen::Matrix3d rot = kf_->x().rot_ext;
        Eigen::Vector3d pos = kf_->x().pos_ext;
        cloud_undistorted_body->reserve(cloud_undistorted_lidar_->size());
        for (auto &p : cloud_undistorted_lidar_->points)
        {
            Eigen::Vector3d point(p.x, p.y, p.z);
            point = rot * point + pos;
            PointType p_body;
            p_body.x = point(0);
            p_body.y = point(1);
            p_body.z = point(2);
            p_body.intensity = p.intensity;
            cloud_undistorted_body->points.push_back(p_body);
        }
        return cloud_undistorted_body;
    }

    PointCloudXYZI::Ptr FastLIOBuilder::cloudDownBody()
    {
        PointCloudXYZI::Ptr cloud_down_body(new PointCloudXYZI);
        Eigen::Matrix3d rot = kf_->x().rot_ext;
        Eigen::Vector3d pos = kf_->x().pos_ext;
        cloud_down_body->reserve(cloud_down_lidar_->size());
        for (auto &p : cloud_down_lidar_->points)
        {
            Eigen::Vector3d point(p.x, p.y, p.z);
            point = rot * point + pos;
            PointType p_body;
            p_body.x = point(0);
            p_body.y = point(1);
            p_body.z = point(2);
            p_body.intensity = p.intensity;
            cloud_down_body->points.push_back(p_body);
        }
        return cloud_down_body;
    }

    void FastLIOBuilder::reset()
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
        ikdtree_.reset(new KD_TREE<PointType>);
    }
}