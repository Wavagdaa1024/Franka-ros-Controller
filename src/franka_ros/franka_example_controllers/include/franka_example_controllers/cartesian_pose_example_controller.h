#pragma once

#include <array>
#include <memory>
#include <string>

#include <controller_interface/multi_interface_controller.h>
#include <franka_hw/franka_state_interface.h>
#include <hardware_interface/robot_hw.h>
#include <realtime_tools/realtime_buffer.h>
#include <ros/node_handle.h>
#include <ros/subscriber.h>
#include <ros/time.h>
#include <geometry_msgs/PoseStamped.h>

#include <franka_hw/franka_cartesian_command_interface.h>

namespace franka_example_controllers {

struct CartesianPoseCommand {
  std::array<double, 3> position{{0.0, 0.0, 0.0}};
  ros::Time stamp{0.0};
  bool valid{false};
};

class CartesianPoseExampleController
    : public controller_interface::MultiInterfaceController<
          franka_hw::FrankaPoseCartesianInterface,
          franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hardware,
            ros::NodeHandle& node_handle) override;

  void starting(const ros::Time&) override;

  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  void targetPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg);

  double clamp(double value, double low, double high) const;

  franka_hw::FrankaPoseCartesianInterface* cartesian_pose_interface_{nullptr};
  std::unique_ptr<franka_hw::FrankaCartesianPoseHandle> cartesian_pose_handle_;

  ros::Subscriber target_pose_sub_;
  realtime_tools::RealtimeBuffer<CartesianPoseCommand> command_buffer_;

  std::array<double, 16> initial_pose_{};
  std::array<double, 16> commanded_pose_{};

  double command_timeout_{0.3};

  // 工作空间限制
  double x_min_{0.20};
  double x_max_{0.70};
  double y_min_{-0.30};
  double y_max_{0.30};
  double z_min_{0.10};
  double z_max_{0.80};

  // 关键参数：最大末端线速度 (m/s)
  double max_linear_velocity_{0.20};
};

}  // namespace franka_example_controllers
