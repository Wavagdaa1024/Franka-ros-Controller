#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import rospy

from controller.base_controller.touch_controller import TouchController
from controller.base_controller.RealSenseCamera import RealSenseCamera
from controller.base_controller.franka_cartesian_vel_controller import FrankaCartesianVelocityController

from controller.touch_franka_teleop_controller import TouchFrankaTeleopController
from controller.teleop_dataset_recorder import TeleopDatasetRecorder


class RecorderMain:
    #============ 初始化与配置 ================

    def __init__(self,
                 save_dir="real_dir",
                 teleop_hz=100,
                 record_hz=20,
                 touch_scale=0.0015 * 2,
                 kp=1.5 * 2,
                 ki=0.15,
                 max_linear_vel=0.5 * 2,
                 touch_deadband=0.5,
                 robot_deadband=0.001,
                 integral_limit=0.03,
                 show_preview=True):
        """
        初始化总控类，统一管理 Touch、Franka、相机、主从控制和数据记录。
        """
        self.touch = TouchController()
        self.arm = FrankaCartesianVelocityController()
        self.cameras = [RealSenseCamera()]

        self.teleop = TouchFrankaTeleopController(
            touch=self.touch,
            arm=self.arm,
            control_hz=teleop_hz,
            touch_scale=touch_scale,
            kp=kp,
            ki=ki,
            max_linear_vel=max_linear_vel,
            touch_deadband=touch_deadband,
            robot_deadband=robot_deadband,
            integral_limit=integral_limit
        )

        self.recorder = TeleopDatasetRecorder(
            arm=self.arm,
            touch=self.touch,
            cameras=self.cameras,
            camera_names=["top"],
            save_dir=save_dir,
            record_rate_hz=record_hz,
            show_preview=show_preview
        )

        self.teleop_hz = teleop_hz
        self._running = False

    #============ 初始化 ================

    def initialize(self, timeout=5.0):
        """等待所有设备与状态就绪。"""
        if not self.recorder.initialize(timeout=timeout):
            return False
        rospy.loginfo("RecorderMain ready.")
        return True

    #============ 主从控制执行 ================

    def teleop_step(self):
        """
        执行一步主从控制。
        这里不直接调用 teleop.run()，而是把控制逻辑合并到总循环中。
        """
        if not self.teleop._teleop_enabled:
            self.arm.stop_motion()
            return np.zeros(3, dtype=np.float64)

        self.teleop.compute_target_position()
        linear_cmd = self.teleop.compute_velocity_command()
        self.arm.set_cartesian_twist(linear=linear_cmd.tolist(), angular=[0.0, 0.0, 0.0])
        return linear_cmd

    #============ 输入处理 ================

    def handle_touch_buttons(self):
        """处理 Touch 按钮逻辑。"""
        edges = self.touch.get_button_edges()

        if edges["up_pressed"]:
            self.teleop.toggle_teleop()

        if edges["down_pressed"]:
            self.teleop.toggle_gripper()

    def handle_keyboard(self):
        """处理键盘逻辑。"""
        key = self.recorder._read_key()

        if key == "s":
            self.recorder.start()
        elif key == "e":
            self.recorder.stop(save=True)
        elif key == "p":
            print("teleop_enabled =", self.teleop._teleop_enabled)
            print("recorder_status =", self.recorder.get_ready_status())
        elif key == "q":
            self._running = False

    #============ 主循环 ================

    def run(self):
        """
        运行统一主循环。

        Touch:
            up: 切换 teleop 开关
            down: 切换夹爪开合

        Keyboard:
            s: 开始记录
            e: 结束并保存
            p: 打印状态
            q: 退出
        """
        try:
            if not self.initialize():
                return

            self.recorder._setup_keyboard()
            self._running = True
            rate = rospy.Rate(self.teleop_hz)

            print("RecorderMain loop started.")
            print("Touch: up=toggle teleop, down=toggle gripper")
            print("Keyboard: s=start record, e=end/save, p=print status, q=quit")

            while not rospy.is_shutdown() and self._running:
                self.handle_keyboard()
                self.handle_touch_buttons()

                linear_cmd = self.teleop_step()

                self.recorder._show_preview()

                now = time.time()
                record_dt = 1.0 / float(self.recorder.record_rate_hz)
                if self.recorder.recording and (now - self.recorder._last_record_time) >= record_dt:
                    ok = self.recorder.record_step(action=None)
                    if ok:
                        self.recorder._last_record_time = now
                    else:
                        print("Skip one frame: state not ready.")

                rate.sleep()

        except rospy.ROSInterruptException:
            pass
        finally:
            self.arm.stop_motion()
            if self.recorder.recording:
                self.recorder.stop(save=True)
            self.recorder._restore_keyboard()
            self.recorder.close()

    #============ 外部停止 ================

    def stop(self):
        """外部主动停止总循环。"""
        self._running = False
        self.teleop.stop()


if __name__ == "__main__":
    app = RecorderMain(
        save_dir="real_dir",
        teleop_hz=100,
        record_hz=20,
        touch_scale=0.0015 * 2,
        kp=1.5 * 2,
        ki=0.15,
        max_linear_vel=0.5 * 2,
        touch_deadband=0.5,
        robot_deadband=0.001,
        integral_limit=0.03,
        show_preview=True
    )
    app.run()

