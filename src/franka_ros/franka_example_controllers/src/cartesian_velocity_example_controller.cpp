#include <franka_example_controllers/cartesian_velocity_example_controller.h>

#include <array>
#include <cmath>
#include <memory>
#include <string>

#include <controller_interface/controller_base.h>
#include <hardware_interface/hardware_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

namespace franka_example_controllers {

bool CartesianVelocityExampleController::init(hardware_interface::RobotHW* robot_hardware,
                                              ros::NodeHandle& node_handle) {
  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("CartesianVelocityExampleController: Could not get parameter arm_id");
    return false;
  }

  velocity_cartesian_interface_ = robot_hardware->get<franka_hw::FrankaVelocityCartesianInterface>();
  if (velocity_cartesian_interface_ == nullptr) {
    ROS_ERROR("CartesianVelocityExampleController: Could not get Cartesian velocity interface from hardware");
    return false;
  }
  try {
    velocity_cartesian_handle_ = std::make_unique<franka_hw::FrankaCartesianVelocityHandle>(
        velocity_cartesian_interface_->getHandle(arm_id + "_robot"));
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM("CartesianVelocityExampleController: Exception getting Cartesian handle: " << e.what());
    return false;
  }

  auto state_interface = robot_hardware->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR("CartesianVelocityExampleController: Could not get state interface from hardware");
    return false;
  }

  // 注册订阅器：监听 Python 发来的速度指令
  sub_command_ = node_handle.subscribe(
      "target_velocity", 20, &CartesianVelocityExampleController::commandCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());

  return true;
}

void CartesianVelocityExampleController::commandCallback(const geometry_msgs::TwistConstPtr& msg) {
  target_velocity_[0] = msg->linear.x;
  target_velocity_[1] = msg->linear.y;
  target_velocity_[2] = msg->linear.z;
  target_velocity_[3] = msg->angular.x;
  target_velocity_[4] = msg->angular.y;
  target_velocity_[5] = msg->angular.z;
  last_command_time_ = ros::Time::now();
}

void CartesianVelocityExampleController::starting(const ros::Time& /* time */) {
  target_velocity_.fill(0.0);
  current_velocity_.fill(0.0); // 初始平滑速度也清零
  last_command_time_ = ros::Time::now();
}

void CartesianVelocityExampleController::update(const ros::Time& /* time */,
                                                const ros::Duration& period) {
  // 1. 看门狗机制：0.1秒没收到指令则目标设为0
  if ((ros::Time::now() - last_command_time_).toSec() > 0.1) {
    target_velocity_.fill(0.0);
  }

  // 2. 一阶低通滤波器：彻底解决加速度/速度不连续导致的报错
  // 0.005 是一个非常平滑的值，如果觉得响应太慢可以微调至 0.01
  double filter_factor = 0.005; 
  std::array<double, 6> command;
  
  for (size_t i = 0; i < 6; i++) {
    current_velocity_[i] = filter_factor * target_velocity_[i] + (1.0 - filter_factor) * current_velocity_[i];
    command[i] = current_velocity_[i];
  }

  // 3. 发送平滑后的指令
  velocity_cartesian_handle_->setCommand(command);
}

void CartesianVelocityExampleController::stopping(const ros::Time& /*time*/) {
  // 留空以允许底层平滑停止
}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianVelocityExampleController,
                       controller_interface::ControllerBase)
