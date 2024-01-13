# iG-LIO_SAM_LC

## 主要工作
1. 对[iG-LIO](https://github.com/zijiechenrobotics/ig_lio)进行梳理和结构改写(未完全对齐原REPO)
2. 参考[FASTLIO](https://github.com/hku-mars/FAST_LIO)的实现,添加了重力优化和外参优化
3. 基于GTSAM和基础ICP匹配,增加了回环线程用于PGO
4. 添加重定位线程，用于支持基于已知地图的定位功能
5. 目前暂时支持MID_360的传感器
6. 改写了并增加了首帧重力对齐，用于应对传感器非水平放置的情况
7. 完全替换了FASTLIO中的IKFoM_toolkit，使用高斯牛顿法等价替换迭代误差卡尔曼，算法思路更加平铺直叙
8. 暂时去掉了iG-LIO 中对于TBB的依赖，其实VoxelMap如果使用NEAR_1模式，即便是单线程也很快，后续再考虑多线程加速。

## 环境说明
```text
系统版本: ubuntu20.04
机器人操作系统: ros1-noetic
```

## 编译依赖
1. livox_ros_driver2
2. gtsam (noetic版本 可以使用 sudo apt install ros-noetic-gtsam 直接安装，其他版本可能需要独立安装)
3. pcl
4. sophus
5. eigen

### 1.安装 LIVOX-SDK2
```shell
git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd ./Livox-SDK2/
mkdir build
cd build
cmake .. && make -j
sudo make install
```

### 2.安装 livox_ros_driver2
```shell
mkdir -r ws_livox/src
git clone https://github.com/Livox-SDK/livox_ros_driver2.git ws_livox/src/livox_ros_driver2
cd ws_livox/src/livox_ros_driver2
./build.sh ROS1
```

### 3. 安装gtsam(ros noetic 可以直接安装)
```shell
sudo apt install ros-noetic-gtsam
```
### 4. 安装Sophus
```
git clone https://github.com/strasdat/Sophus.git
cd Sophus
mkdir build && cd build
cmake .. -DSOPHUS_USE_BASIC_LOGGING=ON
make
sudo make install
```
**新的Sophus依赖fmt，可以在CMakeLists.txt中添加add_compile_definitions(SOPHUS_USE_BASIC_LOGGING)去除，否则会报错**
### 5.安装 iG-LIO_SAM_LC
```shell
mkdir -r ws_igli/src
cd ws_iglio/src
git clone https://github.com/liangheming/iG-LIO_SAM_LC.git
cd ..
catkin_make 
source devel/setup.bash
```

## DEMO 数据
```text
链接: https://pan.baidu.com/s/1ZPUwWyHmvpGHuL9TiFrsmA?pwd=k9vx 提取码: k9vx 
--来自百度网盘超级会员v7的分享
```

## 启动脚本
1. 建图线程
```shell
roslaunch lio mapping.launch
rosbag play your_bag.bag
```
2. 保存地图
```shell
rosservice call /save_map "save_path: 'you_pcd_save_path.pcd'
resolution: 0.0"
``` 

3. 定位线程
```shell
roslaunch lio localize.launch
rosservice call /slam_reloc "{pcd_path: 'you_pcd_path.pcd', x: 0.0, y: 0.0, z: 0.0, roll: 0.0, pitch: 0.0, yaw: 0.0}" 
rosbag play your_bag.bag
```

## 特别感谢
1. [FASTLIO2](https://github.com/hku-mars/FAST_LIO)
2. [iG-LIO](https://github.com/zijiechenrobotics/ig_lio)
3. [FASTLIO-SAM](https://github.com/kahowang/FAST_LIO_SAM)
4. [FASTLIO-LC](https://github.com/HViktorTsoi/FAST_LIO_LOCALIZATION)