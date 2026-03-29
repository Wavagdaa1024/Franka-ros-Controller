#pragma once

#include <memory>
#include <string>
#include <array>

#include <controller_interface/multi_interface_controller.h>
#include <franka_hw/franka_cartesian_command_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <geometry_msgs/Twist.h> 

namespace franka_example_controllers {

class CartesianVelocityExampleController : public controller_interface::MultiInterfaceController<
                                               franka_hw::FrankaVelocityCartesianInterface,
                                               franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& node_handle) override;
  void update(const ros::Time&, const ros::Duration& period) override;
  void starting(const ros::Time&) override;
  void stopping(const ros::Time&) override;

 private:
  franka_hw::FrankaVelocityCartesianInterface* velocity_cartesian_interface_;
  std::unique_ptr<franka_hw::FrankaCartesianVelocityHandle> velocity_cartesian_handle_;

  ros::Subscriber sub_command_;
  void commandCallback(const geometry_msgs::TwistConstPtr& msg); 

  // 目标速度（来自 Python）
  std::array<double, 6> target_velocity_{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}; 
  // 当前平滑速度（发给电机）
  std::array<double, 6> current_velocity_{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}; 
  
  ros::Time last_command_time_; 
};

}  // namespace franka_example_controllers
