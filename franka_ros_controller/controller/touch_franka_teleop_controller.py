#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import rospy

from controller.base_controller.touch_controller import TouchController
from controller.base_controller.franka_cartesian_vel_controller import FrankaCartesianVelocityController


class TouchFrankaTeleopController:
    #============ 初始化与配置 ================

    def __init__(self,
                 touch=None,
                 arm=None,
                 control_hz=100,
                 touch_scale=0.0015,
                 kp=1.5,
                 ki=0.08,
                 max_linear_vel=0.08,
                 touch_deadband=0.5,
                 robot_deadband=0.001,
                 integral_limit=0.03):
        """
        初始化主从控制器。

        参数:
            touch: TouchController 对象，默认自动创建
            arm: FrankaCartesianVelocityController 对象，默认自动创建
            control_hz: 主循环频率
            touch_scale: Touch 位移映射到机器人末端位移的缩放系数
            kp: 位置误差到速度的比例系数
            ki: 位置误差到速度的积分系数
            max_linear_vel: 最大末端线速度
            touch_deadband: Touch 端死区，单位与设备原始位置一致
            robot_deadband: 机器人末端位置误差死区，单位 m
            integral_limit: 积分限幅，防止积分饱和
        """
        self.touch = touch if touch is not None else TouchController()
        self.arm = arm if arm is not None else FrankaCartesianVelocityController()

        self.control_hz = control_hz
        self.touch_scale = touch_scale
        self.kp = kp
        self.ki = ki
        self.max_linear_vel = max_linear_vel
        self.touch_deadband = touch_deadband
        self.robot_deadband = robot_deadband
        self.integral_limit = integral_limit

        self._running = False
        self._teleop_enabled = False
        self._gripper_closed = False

        self._touch_anchor = np.zeros(3, dtype=np.float64)
        self._franka_anchor = np.zeros(3, dtype=np.float64)
        self._target_pos = np.zeros(3, dtype=np.float64)

        self._integral_error = np.zeros(3, dtype=np.float64)
        self._last_step_time = None

    #============ 初始化与标定 ================

    def initialize(self, timeout=3.0):
        """等待 Touch 和 Franka 状态就绪。"""
        if not self.touch.wait_until_ready(timeout=timeout):
            rospy.logerr("Touch device not ready.")
            return False

        start_time = time.time()
        while time.time() - start_time < timeout:
            pos, _ = self.arm.get_cartesian_pose()
            if pos is not None:
                rospy.loginfo("Touch and Franka are ready.")
                return True
            rospy.sleep(0.01)

        rospy.logerr("Franka cartesian state not ready.")
        return False

    def reset_anchor(self):
        """重新记录 Touch 零点和 Franka 当前末端位置。"""
        current_pos, _ = self.arm.get_cartesian_pose()
        if current_pos is None:
            rospy.logwarn("Franka pose not ready, reset anchor failed.")
            return False

        self._touch_anchor = self.touch.zero()
        self._franka_anchor = current_pos.copy()
        self._target_pos = current_pos.copy()
        self.touch.reset_button_edges()
        self._reset_integral()

        rospy.loginfo("Teleop anchor reset.")
        rospy.loginfo("Franka anchor: x={:.4f}, y={:.4f}, z={:.4f}".format(
            self._franka_anchor[0], self._franka_anchor[1], self._franka_anchor[2]
        ))
        return True

    #============ 映射与控制 ================

    def _reset_integral(self):
        """清空位置误差积分项。"""
        self._integral_error[:] = 0.0
        self._last_step_time = None

    def _apply_touch_deadband(self, delta):
        """对 Touch 相对位移加死区，避免手柄微抖。"""
        filtered = delta.copy()
        for i in range(3):
            if abs(filtered[i]) < self.touch_deadband:
                filtered[i] = 0.0
        return filtered

    def compute_target_position(self):
        """根据 Touch 相对位移计算机器人目标末端位置。"""
        mapped_delta = self.touch.get_mapped_delta()
        mapped_delta = self._apply_touch_deadband(mapped_delta)
        self._target_pos = self._franka_anchor + self.touch_scale * mapped_delta
        return self._target_pos.copy()

    def compute_velocity_command(self):
        """根据目标位置和当前位置计算末端线速度指令。"""
        current_pos, _ = self.arm.get_cartesian_pose()
        if current_pos is None:
            self._reset_integral()
            return np.zeros(3, dtype=np.float64)

        now = time.time()
        if self._last_step_time is None:
            dt = 1.0 / float(self.control_hz)
        else:
            dt = max(now - self._last_step_time, 1e-4)
        self._last_step_time = now

        error = self._target_pos - current_pos
        distance = np.linalg.norm(error)

        if distance < self.robot_deadband:
            self._reset_integral()
            return np.zeros(3, dtype=np.float64)

        self._integral_error += error * dt
        self._integral_error = np.clip(self._integral_error, -self.integral_limit, self.integral_limit)

        linear_cmd = self.kp * error + self.ki * self._integral_error
        norm = np.linalg.norm(linear_cmd)
        if norm > self.max_linear_vel:
            linear_cmd = linear_cmd / norm * self.max_linear_vel

        return linear_cmd

    def step(self):
        """执行一次主从控制更新。"""
        if not self._teleop_enabled:
            self.arm.stop_motion()
            return

        self.compute_target_position()
        linear_cmd = self.compute_velocity_command()
        self.arm.set_cartesian_twist(linear=linear_cmd.tolist(), angular=[0.0, 0.0, 0.0])

    #============ 模式切换 ================

    def start_teleop(self):
        """开始主从控制，并记录当前锚点。"""
        if self.reset_anchor():
            self._teleop_enabled = True
            rospy.loginfo("Teleoperation started.")

    def stop_teleop(self):
        """停止主从控制，并发送零速度。"""
        self._teleop_enabled = False
        self._reset_integral()
        self.arm.stop_motion()
        rospy.loginfo("Teleoperation stopped.")

    def toggle_teleop(self):
        """切换主从控制状态。"""
        if self._teleop_enabled:
            self.stop_teleop()
        else:
            self.start_teleop()

    def toggle_gripper(self):
        """切换夹爪开合状态。"""
        try:
            if self._gripper_closed:
                self.arm.open_gripper()
                self._gripper_closed = False
                rospy.loginfo("Gripper opened.")
            else:
                self.arm.close_gripper()
                self._gripper_closed = True
                rospy.loginfo("Gripper closed.")
        except AttributeError:
            rospy.logwarn("Current arm controller does not provide gripper control interface.")

    #============ 主循环 ================

    def run(self):
        """
        运行主从控制主循环。

        按键逻辑:
            up_pressed: 切换主从控制状态
            down_pressed: 切换夹爪开合状态
        """
        if not self.initialize():
            return

        rate = rospy.Rate(self.control_hz)
        self._running = True
        rospy.loginfo("Touch teleop loop started. Press Touch UP to toggle teleop, DOWN to toggle gripper.")

        try:
            while not rospy.is_shutdown() and self._running:
                edges = self.touch.get_button_edges()

                if edges["up_pressed"]:
                    self.toggle_teleop()

                if edges["down_pressed"]:
                    self.toggle_gripper()

                self.step()
                rate.sleep()

        except rospy.ROSInterruptException:
            pass
        finally:
            self.arm.stop_motion()
            self.touch.close()

    def stop(self):
        """外部主动停止主循环。"""
        self._running = False
        self.stop_teleop()


if __name__ == "__main__":
    teleop = TouchFrankaTeleopController(
        control_hz=100,
        touch_scale=0.0015 * 2,
        kp=1.5 * 2,
        ki=0.15,
        max_linear_vel=0.5 * 2,
        touch_deadband=0.5,
        robot_deadband=0.001,
        integral_limit=0.03
    )
    teleop.run()
