#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import tty
import h5py
import termios
import select
import numpy as np
import cv2
import rospy

from controller.base_controller.touch_controller import TouchController
from controller.base_controller.RealSenseCamera import RealSenseCamera
from controller.base_controller.franka_cartesian_vel_controller import FrankaCartesianVelocityController


class TeleopDatasetRecorder:
    #============ 初始化与配置 ================

    def __init__(self,
                 arm=None,
                 touch=None,
                 cameras=None,
                 camera_names=None,
                 save_dir="real_dir",
                 record_rate_hz=20,
                 show_preview=True):
        """
        初始化数据记录器。

        原始数据结构:
            episode_x.hdf5
            ├── attributes
            │   ├── sim = False
            │   └── frequency = 20
            └── observations
                ├── images/top
                ├── robot_joint
                ├── robot_eef_pos
                ├── robot_eef_quat
                ├── robot_gripper_width
                ├── touch_position
                └── timestamp

        其中:
            robot_joint: [q1, q2, q3, q4, q5, q6, q7]
            robot_eef_pos: [x, y, z]
            robot_eef_quat: [qx, qy, qz, qw]
            robot_gripper_width: [gripper_width]
            touch_position: [tx, ty, tz]
            timestamp: [timestamp]

        说明:
            原始录制数据不再保存 action。
            训练用 action 在后处理转换阶段根据观测重新生成。
        """
        self.arm = arm if arm is not None else FrankaCartesianVelocityController()
        self.touch = touch if touch is not None else TouchController()

        if cameras is None:
            cameras = [RealSenseCamera()]
        self.cameras = cameras

        if camera_names is None:
            camera_names = ["top"] if len(self.cameras) == 1 else [f"camera_{i}" for i in range(len(self.cameras))]
        self.camera_names = camera_names

        if len(self.cameras) != len(self.camera_names):
            raise ValueError("cameras 和 camera_names 数量必须一致")

        self.save_dir = save_dir
        self.record_rate_hz = record_rate_hz
        self.show_preview = show_preview

        self.recording = False
        self._running = False
        self._last_record_time = 0.0

        self._keyboard_fd = None
        self._keyboard_old_settings = None

        self.image_data = {name: [] for name in self.camera_names}
        self.robot_joint_data = []
        self.robot_eef_pos_data = []
        self.robot_eef_quat_data = []
        self.robot_gripper_width_data = []
        self.touch_position_data = []
        self.timestamp_data = []

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

    #============ 状态读取 ================

    def get_joint_angles(self):
        """获取当前机械臂 7 维关节角。"""
        return self.arm.get_joint_positions()

    def get_gripper_width(self):
        """获取当前夹爪宽度。"""
        width = self.arm.get_gripper_width()
        if width is None:
            return 0.0
        return float(width)

    def get_eef_pos_quat(self):
        """获取当前末端位置和四元数。"""
        pos, quat = self.arm.get_cartesian_pose()
        if pos is None or quat is None:
            return None, None
        return pos.astype(np.float64), quat.astype(np.float64)

    def get_touch_position(self):
        """获取 Touch 当前三维位置。"""
        if not self.touch.has_state():
            return None
        return self.touch.get_position().astype(np.float64)

    def get_ready_status(self):
        """返回当前各数据源是否就绪。"""
        pos, quat = self.arm.get_cartesian_pose()
        joint_angles = self.arm.get_joint_positions()
        return {
            "joint_ok": joint_angles is not None,
            "cartesian_ok": pos is not None and quat is not None,
            "touch_ok": self.touch.has_state()
        }

    def initialize(self, timeout=5.0):
        """等待关节状态、末端状态和 Touch 状态全部就绪。"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_ready_status()
            if status["joint_ok"] and status["cartesian_ok"] and status["touch_ok"]:
                print("Recorder ready.")
                return True
            time.sleep(0.05)

        print("Recorder initialize failed:", self.get_ready_status())
        return False

    #============ 记录流程控制 ================

    def start(self):
        """开始记录一个新 episode。"""
        self.recording = True
        self._last_record_time = 0.0
        self.image_data = {name: [] for name in self.camera_names}
        self.robot_joint_data = []
        self.robot_eef_pos_data = []
        self.robot_eef_quat_data = []
        self.robot_gripper_width_data = []
        self.touch_position_data = []
        self.timestamp_data = []
        print("Recording started.")

    def stop(self, save=True):
        """
        停止记录。

        参数:
            save: 是否停止后立刻保存
        """
        if not self.recording:
            return
        self.recording = False
        print("Recording stopped.")
        if save:
            self.save_data()

    def record_step(self, timestamp=None):
        """
        记录一个时间步的数据。

        参数:
            timestamp: 当前时间戳; 若为 None 则用 time.time()
        """
        joint_angles = self.get_joint_angles()
        eef_pos, eef_quat = self.get_eef_pos_quat()
        gripper_width = self.get_gripper_width()
        touch_position = self.get_touch_position()

        if joint_angles is None or eef_pos is None or eef_quat is None or touch_position is None:
            return False

        for camera, name in zip(self.cameras, self.camera_names):
            _, color_image_rgb, _ = camera.get_frames()
            if color_image_rgb is None:
                return False
            self.image_data[name].append(color_image_rgb)

        self.robot_joint_data.append(np.asarray(joint_angles, dtype=np.float64))
        self.robot_eef_pos_data.append(np.asarray(eef_pos, dtype=np.float64))
        self.robot_eef_quat_data.append(np.asarray(eef_quat, dtype=np.float64))
        self.robot_gripper_width_data.append(np.asarray([gripper_width], dtype=np.float64))
        self.touch_position_data.append(np.asarray(touch_position, dtype=np.float64))
        self.timestamp_data.append(np.asarray([time.time() if timestamp is None else timestamp], dtype=np.float64))

        return True

    #============ 数据保存 ================

    def _get_next_episode_path(self):
        """获取下一个可用的 episode 文件路径。"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        episode_idx = 0
        while os.path.exists(os.path.join(self.save_dir, f"episode_{episode_idx}.hdf5")):
            episode_idx += 1

        return os.path.join(self.save_dir, f"episode_{episode_idx}.hdf5")

    def save_data(self):
        """保存当前缓存数据到 HDF5。"""
        if len(self.robot_joint_data) == 0:
            print("No data to save.")
            return None

        file_path = self._get_next_episode_path()

        with h5py.File(file_path, "w") as root:
            root.attrs["sim"] = False
            root.attrs["frequency"] = self.record_rate_hz

            obs = root.create_group("observations")
            images = obs.create_group("images")

            for name in self.camera_names:
                image_array = np.asarray(self.image_data[name], dtype=np.uint8)
                if len(image_array) == 0:
                    continue
                images.create_dataset(
                    name,
                    data=image_array,
                    dtype="uint8",
                    chunks=(1, image_array.shape[1], image_array.shape[2], image_array.shape[3])
                )

            obs.create_dataset("robot_joint", data=np.asarray(self.robot_joint_data, dtype=np.float64))
            obs.create_dataset("robot_eef_pos", data=np.asarray(self.robot_eef_pos_data, dtype=np.float64))
            obs.create_dataset("robot_eef_quat", data=np.asarray(self.robot_eef_quat_data, dtype=np.float64))
            obs.create_dataset("robot_gripper_width", data=np.asarray(self.robot_gripper_width_data, dtype=np.float64))
            obs.create_dataset("touch_position", data=np.asarray(self.touch_position_data, dtype=np.float64))
            obs.create_dataset("timestamp", data=np.asarray(self.timestamp_data, dtype=np.float64))

        print("Saved episode to:", file_path)
        print("Frames:", len(self.robot_joint_data))
        return file_path

    #============ 预览与主循环 ================

    def _show_preview(self):
        """显示相机预览。"""
        if not self.show_preview:
            return
        for camera, name in zip(self.cameras, self.camera_names):
            color_image, _, _ = camera.get_frames()
            if color_image is not None:
                cv2.imshow(f"Preview - {name}", color_image)
        cv2.waitKey(1)

    def run(self):
        """
        运行键盘控制记录循环。

        按键逻辑:
            s: 开始记录
            e: 结束并保存
            p: 打印当前就绪状态
            q: 退出程序
        """
        try:
            if not self.initialize():
                return

            self._setup_keyboard()
            self._running = True
            dt = 1.0 / float(self.record_rate_hz)

            print("Recorder loop started.")
            print("Keys: s=start, e=end/save, p=print status, q=quit")

            while not rospy.is_shutdown() and self._running:
                key = self._read_key()

                if key == "s":
                    self.start()
                elif key == "e":
                    self.stop(save=True)
                elif key == "p":
                    print(self.get_ready_status())
                elif key == "q":
                    self._running = False

                self._show_preview()

                now = time.time()
                if self.recording and (now - self._last_record_time) >= dt:
                    ok = self.record_step(timestamp=now)
                    if ok:
                        self._last_record_time = now
                    else:
                        print("Skip one frame: state not ready.")

                time.sleep(0.005)

        finally:
            if self.recording:
                self.stop(save=True)
            self._restore_keyboard()
            self.close()

    #============ 资源释放 ================

    def close(self):
        """关闭相机和 Touch 资源。"""
        for camera in self.cameras:
            camera.stop()
        self.touch.close()


if __name__ == "__main__":
    recorder = TeleopDatasetRecorder(
        save_dir="../real_dir",
        record_rate_hz=20,
        show_preview=True
    )
    recorder.run()

