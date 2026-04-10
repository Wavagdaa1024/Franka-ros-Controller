#include <franka_example_controllers/cartesian_pose_example_controller.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include <controller_interface/controller_base.h>
#include <franka_hw/franka_cartesian_command_interface.h>
#include <hardware_interface/hardware_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

namespace franka_example_controllers {

double CartesianPoseExampleController::clamp(double value, double low, double high) const {
  return std::max(low, std::min(value, high));
}

bool CartesianPoseExampleController::init(hardware_interface::RobotHW* robot_hardware,
                                          ros::NodeHandle& node_handle) {
  cartesian_pose_interface_ =
      robot_hardware->get<franka_hw::FrankaPoseCartesianInterface>();
  if (cartesian_pose_interface_ == nullptr) {
    ROS_ERROR("Could not get Cartesian Pose interface");
    return false;
  }

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("Could not get parameter arm_id");
    return false;
  }

  try {
    cartesian_pose_handle_ =
        std::make_unique<franka_hw::FrankaCartesianPoseHandle>(
            cartesian_pose_interface_->getHandle(arm_id + "_robot"));
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM("Exception getting Cartesian handle: " << e.what());
    return false;
  }

  auto state_interface =
      robot_hardware->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR("Could not get state interface");
    return false;
  }

  try {
    auto state_handle = state_interface->getHandle(arm_id + "_robot");

    std::array<double, 7> q_start{
        {0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
    for (size_t i = 0; i < q_start.size(); i++) {
      if (std::abs(state_handle.getRobotState().q_d[i] - q_start[i]) > 0.1) {
        ROS_ERROR("Robot not in start position");
        return false;
      }
    }
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM("Exception getting state handle: " << e.what());
    return false;
  }

  node_handle.param("command_timeout", command_timeout_, 0.3);

  node_handle.param("x_min", x_min_, 0.20);
  node_handle.param("x_max", x_max_, 0.70);
  node_handle.param("y_min", y_min_, -0.30);
  node_handle.param("y_max", y_max_, 0.30);
  node_handle.param("z_min", z_min_, 0.10);
  node_handle.param("z_max", z_max_, 0.80);

  node_handle.param("max_linear_velocity", max_linear_velocity_, 0.20);

  target_pose_sub_ = node_handle.subscribe(
      "target_pose", 1,
      &CartesianPoseExampleController::targetPoseCallback, this);

  CartesianPoseCommand init_cmd;
  command_buffer_.writeFromNonRT(init_cmd);

  return true;
}

void CartesianPoseExampleController::starting(const ros::Time&) {
  initial_pose_ =
      cartesian_pose_handle_->getRobotState().O_T_EE_d;

  commanded_pose_ = initial_pose_;

  CartesianPoseCommand cmd;
  cmd.position = {initial_pose_[12], initial_pose_[13], initial_pose_[14]};
  cmd.stamp = ros::Time::now();
  cmd.valid = true;

  command_buffer_.writeFromNonRT(cmd);
}

void CartesianPoseExampleController::targetPoseCallback(
    const geometry_msgs::PoseStampedConstPtr& msg) {

  CartesianPoseCommand cmd;

  cmd.position = {
      clamp(msg->pose.position.x, x_min_, x_max_),
      clamp(msg->pose.position.y, y_min_, y_max_),
      clamp(msg->pose.position.z, z_min_, z_max_)};

  cmd.stamp =
      msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;

  cmd.valid = true;

  command_buffer_.writeFromNonRT(cmd);
}

void CartesianPoseExampleController::update(
    const ros::Time& time,
    const ros::Duration& period) {

  CartesianPoseCommand cmd = *(command_buffer_.readFromRT());

  std::array<double, 3> target = {
      commanded_pose_[12],
      commanded_pose_[13],
      commanded_pose_[14]};

  if (cmd.valid && (time - cmd.stamp).toSec() <= command_timeout_) {
    target = cmd.position;
  }

  double dx = target[0] - commanded_pose_[12];
  double dy = target[1] - commanded_pose_[13];
  double dz = target[2] - commanded_pose_[14];

  double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

  double max_step = max_linear_velocity_ * period.toSec();

  if (dist <= max_step || dist < 1e-6) {
    commanded_pose_[12] = target[0];
    commanded_pose_[13] = target[1];
    commanded_pose_[14] = target[2];
  } else {
    double scale = max_step / dist;
    commanded_pose_[12] += dx * scale;
    commanded_pose_[13] += dy * scale;
    commanded_pose_[14] += dz * scale;
  }

  cartesian_pose_handle_->setCommand(commanded_pose_);
}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(
    franka_example_controllers::CartesianPoseExampleController,
    controller_interface::ControllerBase)
