#pragma once

#include <list>
#include <vector>
#include <memory>
#include <Eigen/SVD>
#include <Eigen/Core>
#include <unordered_map>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

// #include "commons.h"

namespace lio
{
    struct PointCov
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        PointCov(Eigen::Vector3d &p, Eigen::Matrix3d &c) : point(p), cov(c) {}
        Eigen::Vector3d point;
        Eigen::Matrix3d cov;
    };
    bool compare(std::pair<Eigen::Vector3d, double> &p1, std::pair<Eigen::Vector3d, double> &p2);

    class HashUtil
    {
    public:
        HashUtil() = default;
        HashUtil(double resolution) : resolution_inv_(1.0 / resolution) {}
        HashUtil(double resolution, size_t hash_p, size_t max_n) : hash_p_(hash_p), max_n_(max_n), resolution_inv_(1.0 / resolution) {}
        size_t operator()(const Eigen::Vector3d &point);

    private:
        size_t hash_p_ = 116101;
        size_t max_n_ = 10000000000;
        double resolution_inv_ = 1.0;
    };

    struct Grid
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Grid(size_t num) : max_num(num), is_updated(false) { points.reserve(2 * max_num); }
        size_t hash = 0;
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        Eigen::Matrix3d conv = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d conv_inv = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d conv_sum = Eigen::Matrix3d::Zero();
        Eigen::Vector3d points_sum = Eigen::Vector3d::Zero();
        size_t points_num = 0;
        size_t min_num = 6;
        size_t max_num = 20;
        bool is_valid = false;
        std::vector<Eigen::Vector3d> points;
        void setMinMax(size_t min_n, size_t max_n);
        void updateConv();
        void addPoint(Eigen::Vector3d &point, bool insert);
        bool is_updated = false;
    };

    class VoxelMap
    {
    public:
        enum MODE
        {
            NEARBY_1,
            NEARBY_7,
            NEARBY_19,
            NEARBY_26
        };
        VoxelMap(double resolution, size_t capacity, size_t grid_capacity, MODE mode);

        bool isSameGrid(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2);

        size_t addCloud(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud);

        bool searchKNN(const Eigen::Vector3d &point, size_t K, double range, std::vector<Eigen::Vector3d> &results);

        bool getCentroidAndCovariance(size_t hash_idx, Eigen::Vector3d &centroid, Eigen::Matrix3d &cov);

        bool getCentroidAndCovariance(const Eigen::Vector3d &point, Eigen::Vector3d &centroid, Eigen::Matrix3d &cov);

        std::vector<Eigen::Vector3d> &searchRange() { return delta_; };

        size_t size() { return cache_.size(); }

        void reset();

    private:
        MODE mode_ = MODE::NEARBY_7;
        double resolution_ = 1.0;
        double resolution_inv_ = 1.0;
        size_t capacity_ = 5000000;
        size_t grid_capacity_ = 20;
        HashUtil hash_util_;
        std::vector<Eigen::Vector3d> delta_;

        std::vector<size_t> current_idx_;
        std::list<size_t> cache_;
        std::unordered_map<size_t, std::pair<std::list<size_t>::iterator, std::shared_ptr<Grid>>> storage_;
        void initializeDelta();
    };
    
    struct FastGrid
    {
    public:
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        size_t num = 0;
        void addPoint(const Eigen::Vector3d &point);
    };
    
    class FastVoxelMap
    {
    public:
        FastVoxelMap(double resolution);
        void filter(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud, std::vector<PointCov> &outs);

    private:
        HashUtil hash_util_;
        double resolution_;
        std::vector<Eigen::Vector3d> search_range_;
        std::vector<std::shared_ptr<FastGrid>> grid_array_;
        std::unordered_map<size_t, std::shared_ptr<FastGrid>> grid_map_;
    };
} // namespace lio
