#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from franka_msgs.msg import FrankaState
from franka_gripper.msg import MoveActionGoal, GraspActionGoal, GraspGoal
from tf.transformations import quaternion_from_matrix


class FrankaCartesianVelocityController:
    # ==================== 初始化与配置 ====================
    def __init__(self, controller_topic='/cartesian_velocity_example_controller/target_velocity'):
        """
        初始化 Franka 末端笛卡尔速度控制器。
        借鉴了 robot_state.py 的类封装结构。
        """
        if not rospy.core.is_initialized():
            rospy.init_node('franka_cartesian_vel_controller', anonymous=True)

        # 速度指令发布者，连接到底层 C++ 控制器
        self.vel_pub = rospy.Publisher(controller_topic, Twist, queue_size=1)

        # 状态订阅者，获取包含 O_T_EE 和 q 的机械臂状态
        self.state_sub = rospy.Subscriber(
            '/franka_state_controller/franka_states',
            FrankaState,
            self._state_callback
        )

        # 夹爪控制与状态
        self.gripper_move_pub = rospy.Publisher('/franka_gripper/move/goal', MoveActionGoal, queue_size=10)
        self.gripper_grasp_pub = rospy.Publisher('/franka_gripper/grasp/goal', GraspActionGoal, queue_size=10)
        self.gripper_state_sub = rospy.Subscriber('/franka_gripper/joint_states', JointState, self._gripper_state_callback)
        self.gripper_positions = []

        self.gripper_command_state = 'unknown'
        self.open_trigger_count = 0
        self.close_trigger_count = 0
        self.gripper_last_cmd_time = 0.0
        self.gripper_cmd_cooldown = 0.5

        # 控制频率设为 200Hz 以满足平滑要求
        self.rate = rospy.Rate(200)

        # 存储机械臂状态的内部变量
        self.current_pose_matrix = None
        self.current_pos = np.zeros(3, dtype=np.float64)
        self.current_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.current_joint_pos = None

        # 固定初始位姿
        self.initial_pose_matrix = np.array([
            [9.9999032943e-01, -2.2293657136e-04, -1.9555093429e-04, 3.0687314493e-01],
            [-2.2299660341e-04, -9.9999030141e-01, -3.0702421719e-04, -6.6142970518e-05],
            [-1.9548059080e-04, 3.0706485529e-04, -9.9999993375e-01, 4.8692429186e-01],
            [0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 1.0000000000e+00]
        ], dtype=np.float64)

        self.initial_pos = np.array([
            3.0687314493e-01,
            -6.6142970518e-05,
            4.8692429186e-01
        ], dtype=np.float64)

        self.initial_quat = np.array([
            9.9999757057e-01,
            -1.1148356454e-04,
            -9.7758118769e-05,
            1.5352264109e-04
        ], dtype=np.float64)

        rospy.loginfo("Franka Cartesian Velocity Controller 初始化完成，等待状态更新...")

    # ==================== 主要可调用函数 ====================
    def get_cartesian_pose(self):
        """
        获取当前末端笛卡尔位姿。
        返回: position (3,), quaternion (4,)
        """
        if self.current_pose_matrix is None:
            rospy.logwarn_once("尚未接收到 Franka 末端笛卡尔状态数据。")
            return None, None
        return self.current_pos, self.current_quat

    def get_joint_positions(self):
        """
        获取当前 7 维关节角。
        返回: joint_positions (7,)
        """
        if self.current_joint_pos is None:
            rospy.logwarn_once("尚未接收到 Franka 关节状态数据。")
            return None
        return self.current_joint_pos.copy()

    def set_cartesian_twist(self, linear=[0.0, 0.0, 0.0], angular=[0.0, 0.0, 0.0]):
        """
        下发笛卡尔速度指令。
        """
        vel_msg = Twist()
        vel_msg.linear.x, vel_msg.linear.y, vel_msg.linear.z = linear
        vel_msg.angular.x, vel_msg.angular.y, vel_msg.angular.z = angular
        self.vel_pub.publish(vel_msg)

    def stop_motion(self):
        """紧急停止末端运动。"""
        self.set_cartesian_twist([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def get_gripper_width(self):
        """获取当前夹爪张开宽度。"""
        if not self.gripper_positions:
            rospy.logwarn("No gripper positions received yet.")
            return None
        return float(sum(self.gripper_positions))

    def open_gripper(self, width=0.08, speed=0.5):
        """打开夹爪。"""
        while self.gripper_move_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.sleep(0.05)
        move_goal = MoveActionGoal()
        move_goal.goal.width = width
        move_goal.goal.speed = speed
        self.gripper_move_pub.publish(move_goal)

    def close_gripper(self, width=0.055, force=5.0, speed=0.1, inner_epsilon=0.005, outer_epsilon=0.005):
        """闭合夹爪。"""
        while self.gripper_grasp_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.sleep(0.05)
        grasp_goal = GraspGoal()
        grasp_goal.width = width
        grasp_goal.speed = speed
        grasp_goal.force = force
        grasp_goal.epsilon.inner = inner_epsilon
        grasp_goal.epsilon.outer = outer_epsilon
        action_goal = GraspActionGoal()
        action_goal.goal = grasp_goal
        self.gripper_grasp_pub.publish(action_goal)

    def update_gripper_with_protection(self, output_value,
                                   open_threshold=0.8,
                                   close_threshold=-0.8,
                                   trigger_count=3,
                                   cooldown=0.5):
        """
        带防抖保护的夹爪控制：
        - 连续 trigger_count 次超过 open_threshold 才打开
        - 连续 trigger_count 次低于 close_threshold 才关闭
        - 中间区域清零计数
        - 同状态不重复发命令
        - 加 cooldown 防止短时间重复触发
        """
        now = rospy.Time.now().to_sec()

        if now - self.gripper_last_cmd_time < cooldown:
            return

        if output_value >= open_threshold:
            self.open_trigger_count += 1
            self.close_trigger_count = 0
        elif output_value <= close_threshold:
            self.close_trigger_count += 1
            self.open_trigger_count = 0
        else:
            self.open_trigger_count = 0
            self.close_trigger_count = 0
            return

        if self.open_trigger_count >= trigger_count:
            if self.gripper_command_state != 'open':
                rospy.loginfo("Gripper protected open triggered.")
                self.open_gripper()
                self.gripper_command_state = 'open'
                self.gripper_last_cmd_time = now
            self.open_trigger_count = 0
            self.close_trigger_count = 0

        elif self.close_trigger_count >= trigger_count:
            if self.gripper_command_state != 'close':
                rospy.loginfo("Gripper protected close triggered.")
                self.close_gripper()
                self.gripper_command_state = 'close'
                self.gripper_last_cmd_time = now
            self.open_trigger_count = 0
            self.close_trigger_count = 0



    def go_to_initial_position(self, kp=1.2, ki=0.08, max_linear_vel=0.05, pos_tolerance=0.001, timeout=15.0, integral_limit=0.05):
        """使用 PI 控制器将末端平滑地回到初始位置。"""
        if self.initial_pos is None:
            rospy.logwarn("initial_pos 未设置")
            return False

        start_time = rospy.Time.now().to_sec()
        last_time = start_time
        integral = np.zeros(3, dtype=np.float64)

        rospy.loginfo("开始回到初始位置")
        while not rospy.is_shutdown():
            if self.current_pose_matrix is None:
                self.stop_motion()
                self.rate.sleep()
                continue

            now = rospy.Time.now().to_sec()
            dt = max(now - last_time, 1e-4)
            last_time = now

            error = self.initial_pos - self.current_pos
            distance = np.linalg.norm(error)

            if distance < pos_tolerance:
                self.stop_motion()
                rospy.loginfo("已回到初始位置, 误差 {:.6f} m".format(distance))
                return True

            integral += error * dt
            integral = np.clip(integral, -integral_limit, integral_limit)

            linear_cmd = kp * error + ki * integral
            norm = np.linalg.norm(linear_cmd)
            if norm > max_linear_vel:
                linear_cmd = linear_cmd / norm * max_linear_vel

            self.set_cartesian_twist(linear=linear_cmd.tolist(), angular=[0.0, 0.0, 0.0])

            if now - start_time > timeout:
                self.stop_motion()
                rospy.logwarn("回初始位置超时")
                return False

            self.rate.sleep()

        self.stop_motion()
        return False

    # ==================== 内部回调函数 ====================
    def _state_callback(self, msg):
        """
        [内部调用] 解析 FrankaState 消息，提取末端位姿和关节角。
        """
        matrix = np.array(msg.O_T_EE, dtype=np.float64).reshape(4, 4, order='F')
        self.current_pose_matrix = matrix
        self.current_pos = matrix[:3, 3]
        self.current_quat = quaternion_from_matrix(matrix)
        self.current_joint_pos = np.array(msg.q, dtype=np.float64)

    def _gripper_state_callback(self, msg):
        """更新当前夹爪关节状态。"""
        if msg.position is not None:
            self.gripper_positions = list(msg.position)

    # ==================== 测试函数 ====================
    def test_soft_start_x_axis(self, target_v=0.03, step=0.0001):
        """
        X 轴平滑启动测试，用于验证底层看门狗机制与滤波效果。
        """
        rospy.loginfo("开始执行末端 X 轴平滑加速测试...")
        current_v = 0.0

        try:
            while not rospy.is_shutdown():
                if current_v < target_v:
                    current_v += step

                self.set_cartesian_twist(linear=[current_v, 0.0, 0.0], angular=[0.0, 0.0, 0.0])

                pos, _ = self.get_cartesian_pose()
                if pos is not None:
                    rospy.loginfo_throttle(0.5, "当前末端 X 坐标: {:.4f} m".format(pos[0]))

                self.rate.sleep()

        except rospy.ROSInterruptException:
            rospy.loginfo("测试手动中断，底层看门狗将自动归零速度。")

    def test_check_franka_state_once(self, timeout=3.0):
        """一次性检查 FrankaState 中的末端位姿和关节角是否都能收到。"""
        rospy.loginfo("等待一帧 /franka_state_controller/franka_states ...")
        msg = rospy.wait_for_message('/franka_state_controller/franka_states', FrankaState, timeout=timeout)

        matrix = np.array(msg.O_T_EE, dtype=np.float64).reshape(4, 4, order='F')
        pos = matrix[:3, 3]
        quat = quaternion_from_matrix(matrix)
        q = np.array(msg.q, dtype=np.float64)

        rospy.loginfo("O_T_EE length: {}".format(len(msg.O_T_EE)))
        rospy.loginfo("q length: {}".format(len(msg.q)))
        rospy.loginfo("pos: {}".format(np.round(pos, 6)))
        rospy.loginfo("quat: {}".format(np.round(quat, 6)))
        rospy.loginfo("q: {}".format(np.round(q, 6)))

        self.current_pose_matrix = matrix
        self.current_pos = pos
        self.current_quat = quat
        self.current_joint_pos = q

    def test_gripper_toggle(self):
        """测试夹爪开合。"""
        rospy.loginfo("打开夹爪...")
        self.open_gripper()
        rospy.sleep(2.0)
        rospy.loginfo("闭合夹爪...")
        self.close_gripper()

    def test_print_gripper_width(self):
        """持续打印夹爪宽度。"""
        while not rospy.is_shutdown():
            width = self.get_gripper_width()
            if width is not None:
                rospy.loginfo_throttle(0.5, "当前夹爪宽度: {:.4f} m".format(width))
            self.rate.sleep()

    def test_print_joint_positions(self):
        """持续打印当前 7 维关节角。"""
        while not rospy.is_shutdown():
            q = self.get_joint_positions()
            if q is not None:
                rospy.loginfo_throttle(0.5, "当前关节角: {}".format(np.round(q, 4)))
            self.rate.sleep()


if __name__ == '__main__':
    try:
        arm = FrankaCartesianVelocityController()
        rospy.sleep(1.0)

        # =============测试函数写这里=============
        arm.test_check_franka_state_once()

    except rospy.ROSInterruptException:
        pass