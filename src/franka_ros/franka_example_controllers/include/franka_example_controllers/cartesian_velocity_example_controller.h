#pragma once

#include <array>
#include <memory>
#include <string>

#include <controller_interface/multi_interface_controller.h>
#include <franka_hw/franka_cartesian_command_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <geometry_msgs/Twist.h>
#include <hardware_interface/robot_hw.h>
#include <realtime_tools/realtime_buffer.h>
#include <ros/node_handle.h>
#include <ros/time.h>

namespace franka_example_controllers {

struct CartesianVelocityCommand {
  std::array<double, 6> velocity{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  ros::Time stamp{0.0};
};

class CartesianVelocityExampleController : public controller_interface::MultiInterfaceController<
                                               franka_hw::FrankaVelocityCartesianInterface,
                                               franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& node_handle) override;
  void update(const ros::Time&, const ros::Duration& period) override;
  void starting(const ros::Time&) override;
  void stopping(const ros::Time&) override;

 private:
  void commandCallback(const geometry_msgs::TwistConstPtr& msg);

  franka_hw::FrankaVelocityCartesianInterface* velocity_cartesian_interface_{nullptr};
  std::unique_ptr<franka_hw::FrankaCartesianVelocityHandle> velocity_cartesian_handle_;
  ros::Subscriber sub_command_;
  realtime_tools::RealtimeBuffer<CartesianVelocityCommand> command_buffer_;

  std::array<double, 6> current_velocity_{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  std::array<double, 6> last_target_velocity_{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

  double command_timeout_{0.10};
  double command_hold_time_{0.20};
  double filter_time_constant_{0.01};
  double max_linear_velocity_{0.15};
  double max_angular_velocity_{0.80};
  double max_linear_acceleration_{0.25};
  double max_angular_acceleration_{0.80};
  double velocity_deadband_{2e-4};
};

}  // namespace franka_example_controllers

