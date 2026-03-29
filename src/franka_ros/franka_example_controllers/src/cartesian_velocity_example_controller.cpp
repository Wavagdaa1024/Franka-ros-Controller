#include <franka_example_controllers/cartesian_velocity_example_controller.h>

#include <algorithm>
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

namespace {
double clampAbs(double value, double limit) {
  return std::max(-limit, std::min(value, limit));
}

double limitDelta(double target, double current, double max_delta) {
  const double delta = target - current;
  if (delta > max_delta) return current + max_delta;
  if (delta < -max_delta) return current - max_delta;
  return target;
}
}  // namespace

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

  CartesianVelocityCommand init_command;
  init_command.stamp = ros::Time(0.0);
  command_buffer_.writeFromNonRT(init_command);

  sub_command_ = node_handle.subscribe("target_velocity", 20,
                                       &CartesianVelocityExampleController::commandCallback, this,
                                       ros::TransportHints().reliable().tcpNoDelay());

  ROS_INFO_STREAM("CartesianVelocityExampleController loaded with buffered timeout stop");
  return true;
}

void CartesianVelocityExampleController::commandCallback(const geometry_msgs::TwistConstPtr& msg) {
  CartesianVelocityCommand command;
  command.velocity[0] = msg->linear.x;
  command.velocity[1] = msg->linear.y;
  command.velocity[2] = msg->linear.z;
  command.velocity[3] = msg->angular.x;
  command.velocity[4] = msg->angular.y;
  command.velocity[5] = msg->angular.z;
  command.stamp = ros::Time::now();
  command_buffer_.writeFromNonRT(command);
}

void CartesianVelocityExampleController::starting(const ros::Time& time) {
  current_velocity_.fill(0.0);
  last_target_velocity_.fill(0.0);
  CartesianVelocityCommand command;
  command.stamp = time;
  command_buffer_.writeFromNonRT(command);
}

void CartesianVelocityExampleController::update(const ros::Time& time, const ros::Duration& period) {
  CartesianVelocityCommand command = *(command_buffer_.readFromRT());
  std::array<double, 6> target_velocity = command.velocity;

  const double dt = std::max(period.toSec(), 1e-6);
  const double command_age = (time - command.stamp).toSec();

  if (command_age <= command_timeout_) {
    last_target_velocity_ = target_velocity;
  } else if (command_age <= command_timeout_ + command_hold_time_) {
    target_velocity = last_target_velocity_;
  } else {
    target_velocity.fill(0.0);
  }

  for (size_t i = 0; i < 3; ++i) target_velocity[i] = clampAbs(target_velocity[i], max_linear_velocity_);
  for (size_t i = 3; i < 6; ++i) target_velocity[i] = clampAbs(target_velocity[i], max_angular_velocity_);

  const double tau = std::max(filter_time_constant_, 1e-4);
  const double alpha = dt / (tau + dt);

  std::array<double, 6> filtered_target;
  for (size_t i = 0; i < 6; ++i) {
    filtered_target[i] = current_velocity_[i] + alpha * (target_velocity[i] - current_velocity_[i]);
  }

  const double max_linear_delta = max_linear_acceleration_ * dt;
  const double max_angular_delta = max_angular_acceleration_ * dt;
  std::array<double, 6> command_out;

  for (size_t i = 0; i < 3; ++i) {
    current_velocity_[i] = limitDelta(filtered_target[i], current_velocity_[i], max_linear_delta);
    if (std::abs(current_velocity_[i]) < velocity_deadband_) current_velocity_[i] = 0.0;
    command_out[i] = current_velocity_[i];
  }
  for (size_t i = 3; i < 6; ++i) {
    current_velocity_[i] = limitDelta(filtered_target[i], current_velocity_[i], max_angular_delta);
    if (std::abs(current_velocity_[i]) < velocity_deadband_) current_velocity_[i] = 0.0;
    command_out[i] = current_velocity_[i];
  }

  velocity_cartesian_handle_->setCommand(command_out);
}

void CartesianVelocityExampleController::stopping(const ros::Time& /*time*/) {
  current_velocity_.fill(0.0);
  last_target_velocity_.fill(0.0);
  velocity_cartesian_handle_->setCommand(current_velocity_);
}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianVelocityExampleController,
                       controller_interface::ControllerBase)

