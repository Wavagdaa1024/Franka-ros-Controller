#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import tty
import termios
import select
import h5py
import numpy as np
import rospy

from controller.base_controller.franka_cartesian_vel_controller import FrankaCartesianVelocityController


class RecordedTrajectoryReplayer:
    #============ 初始化与配置 ================

    def __init__(self,
                 hdf5_path,
                 arm=None,
                 control_hz=100,
                 sample_dt=0.05,
                 kp=2.0,
                 ki=0.03,
                 max_linear_vel=0.08,
                 pos_tolerance=0.002,
                 integral_limit=0.01,
                 close_threshold=0.04):
        """
        初始化轨迹重放器。

        参数:
            hdf5_path: 录制得到的 hdf5 文件路径
            arm: FrankaCartesianVelocityController 对象，默认自动创建
            control_hz: 重放控制频率
            sample_dt: 数据集相邻两帧时间间隔，默认 0.05s
            kp: 位置误差比例系数
            ki: 位置误差积分系数
            max_linear_vel: 最大末端线速度
            pos_tolerance: 单点位置误差容差
            integral_limit: 积分限幅
            close_threshold: 夹爪闭合阈值
        """
        self.hdf5_path = hdf5_path
        self.arm = arm if arm is not None else FrankaCartesianVelocityController()

        self.control_hz = control_hz
        self.sample_dt = sample_dt
        self.kp = kp
        self.ki = ki
        self.max_linear_vel = max_linear_vel
        self.pos_tolerance = pos_tolerance
        self.integral_limit = integral_limit
        self.close_threshold = close_threshold

        self.positions = None
        self.quaternions = None
        self.gripper_widths = None
        self.timestamps = None
        self.frequency = None

        self._integral_error = np.zeros(3, dtype=np.float64)
        self._last_time = None
        self._gripper_closed = False
        self._running = False
        self._replaying = False

        self._keyboard_fd = None
        self._keyboard_old_settings = None

    #============ 键盘控制 ================

    def _setup_keyboard(self):
        """设置终端为非阻塞按键读取模式。"""
        self._keyboard_fd = sys.stdin.fileno()
        self._keyboard_old_settings = termios.tcgetattr(self._keyboard_fd)
        tty.setcbreak(self._keyboard_fd)

    def _restore_keyboard(self):
        """恢复终端原始设置。"""
        if self._keyboard_fd is not None and self._keyboard_old_settings is not None:
            termios.tcsetattr(self._keyboard_fd, termios.TCSADRAIN, self._keyboard_old_settings)
        self._keyboard_fd = None
        self._keyboard_old_settings = None

    def _read_key(self):
        """非阻塞读取一个按键，没有输入则返回 None。"""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    #============ 数据加载 ================

    def load_data(self):
        """从 hdf5 文件中加载轨迹数据。"""
        with h5py.File(self.hdf5_path, "r") as f:
            obs = f["observations"]

            self.positions = np.asarray(obs["robot_eef_pos"], dtype=np.float64)
            self.quaternions = np.asarray(obs["robot_eef_quat"], dtype=np.float64)
            self.gripper_widths = np.asarray(obs["robot_gripper_width"], dtype=np.float64).reshape(-1)

            if "timestamp" in obs:
                self.timestamps = np.asarray(obs["timestamp"], dtype=np.float64).reshape(-1)
            else:
                self.timestamps = None

            self.frequency = float(f.attrs["frequency"]) if "frequency" in f.attrs else None

        if self.timestamps is None and self.frequency is not None:
            self.sample_dt = 1.0 / self.frequency

        print("Loaded trajectory from:", self.hdf5_path)
        print("Frames:", len(self.positions))

    #============ 状态与控制 ================

    def initialize(self, timeout=5.0):
        """等待机械臂状态就绪。"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            pos, _ = self.arm.get_cartesian_pose()
            if pos is not None:
                rospy.loginfo("Replayer ready.")
                return True
            rospy.sleep(0.01)
        rospy.logerr("Franka cartesian state not ready.")
        return False

    def _reset_integral(self):
        """清空积分项。"""
        self._integral_error[:] = 0.0
        self._last_time = None

    def _compute_velocity(self, target_pos):
        """根据目标末端位置计算线速度指令。"""
        current_pos, _ = self.arm.get_cartesian_pose()
        if current_pos is None:
            self._reset_integral()
            return np.zeros(3, dtype=np.float64), None

        now = time.time()
        if self._last_time is None:
            dt = 1.0 / float(self.control_hz)
        else:
            dt = max(now - self._last_time, 1e-4)
        self._last_time = now

        error = target_pos - current_pos
        distance = np.linalg.norm(error)

        if distance < self.pos_tolerance:
            self._reset_integral()
            return np.zeros(3, dtype=np.float64), distance

        self._integral_error += error * dt
        self._integral_error = np.clip(self._integral_error, -self.integral_limit, self.integral_limit)

        linear_cmd = self.kp * error + self.ki * self._integral_error
        norm = np.linalg.norm(linear_cmd)
        if norm > self.max_linear_vel:
            linear_cmd = linear_cmd / norm * self.max_linear_vel

        return linear_cmd, distance

    def _update_gripper(self, target_width):
        """根据记录的夹爪宽度切换夹爪状态。"""
        if target_width < self.close_threshold and not self._gripper_closed:
            self.arm.close_gripper()
            self._gripper_closed = True
            rospy.loginfo("Replay gripper: close")
        elif target_width >= self.close_threshold and self._gripper_closed:
            self.arm.open_gripper()
            self._gripper_closed = False
            rospy.loginfo("Replay gripper: open")


    def _get_replay_index(self, elapsed, total_frames):
        """根据 elapsed time 获取当前应重放的轨迹索引。"""
        if self.timestamps is not None and len(self.timestamps) == total_frames:
            rel_time = self.timestamps - self.timestamps[0]
            idx = np.searchsorted(rel_time, elapsed, side="right") - 1
            idx = max(0, min(idx, total_frames - 1))
            return idx

        idx = min(int(elapsed / self.sample_dt), total_frames - 1)
        return idx

    #============ 重放流程 ================

    def move_to_start_pose(self, timeout=10.0):
        """先移动到轨迹起点位置。"""
        if self.positions is None:
            self.load_data()

        target_pos = self.positions[0]
        self._reset_integral()
        start_time = time.time()

        rospy.loginfo("Moving to replay start pose...")
        rate = rospy.Rate(self.control_hz)

        while not rospy.is_shutdown() and self._running:
            key = self._read_key()
            if key == "q":
                self._running = False
                self.arm.stop_motion()
                return False

            linear_cmd, distance = self._compute_velocity(target_pos)
            self.arm.set_cartesian_twist(linear=linear_cmd.tolist(), angular=[0.0, 0.0, 0.0])

            if distance is not None and distance < self.pos_tolerance:
                self.arm.stop_motion()
                rospy.loginfo("Reached replay start pose.")
                return True

            if time.time() - start_time > timeout:
                self.arm.stop_motion()
                rospy.logwarn("Move to start pose timeout.")
                return False

            rate.sleep()

        self.arm.stop_motion()
        return False

    def replay_once(self):
        """执行一次完整轨迹重放。"""
        if self.positions is None:
            self.load_data()

        if not self.move_to_start_pose():
            return False

        rate = rospy.Rate(self.control_hz)
        self._reset_integral()
        start_time = time.time()
        total_frames = len(self.positions)
        self._replaying = True

        rospy.loginfo("Start replay trajectory...")

        while not rospy.is_shutdown() and self._running and self._replaying:
            key = self._read_key()
            if key == "q":
                self._running = False
                break

            elapsed = time.time() - start_time
            idx = self._get_replay_index(elapsed, total_frames)

            target_pos = self.positions[idx]
            target_width = self.gripper_widths[idx]

            self._update_gripper(target_width)

            linear_cmd, distance = self._compute_velocity(target_pos)
            self.arm.set_cartesian_twist(linear=linear_cmd.tolist(), angular=[0.0, 0.0, 0.0])

            rospy.loginfo_throttle(
                0.5,
                "Replay idx = {} / {} | dist = {:.4f} m".format(
                    idx, total_frames - 1, 0.0 if distance is None else distance
                )
            )

            if idx >= total_frames - 1 and distance is not None and distance < self.pos_tolerance:
                break

            rate.sleep()

        self.arm.stop_motion()
        self._replaying = False
        rospy.loginfo("Replay finished.")
        return True

    #============ 主循环 ================

    def run(self):
        """
        运行重放主循环。

        键盘逻辑:
            r: 开始 replay
            q: 停止并退出
        """
        try:
            self.load_data()
            if not self.initialize():
                return

            self._setup_keyboard()
            self._running = True

            print("RecordedTrajectoryReplayer ready.")
            print("Keys: r=start replay, q=quit")

            while not rospy.is_shutdown() and self._running:
                key = self._read_key()

                if key == "r" and not self._replaying:
                    self.replay_once()
                elif key == "q":
                    self._running = False

                time.sleep(0.01)

        finally:
            self.arm.stop_motion()
            self._restore_keyboard()


if __name__ == "__main__":
    try:
        replayer = RecordedTrajectoryReplayer(
            hdf5_path="real_dir/episode_1.hdf5",
            control_hz=100,
            sample_dt=0.05,
            kp=2.0,
            ki=0.03,
            max_linear_vel=0.08,
            pos_tolerance=0.002,
            integral_limit=0.01,
            close_threshold=0.06
        )
        replayer.run()

    except rospy.ROSInterruptException:
        pass
