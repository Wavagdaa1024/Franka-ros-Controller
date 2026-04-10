#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from franka_msgs.msg import FrankaState
from franka_gripper.msg import MoveActionGoal, GraspActionGoal, GraspGoal
from tf.transformations import quaternion_from_matrix


class FrankaCartesianPositionController:
    # ==================== 初始化与配置 ====================
    def __init__(self,
                 controller_topic='/cartesian_pose_example_controller/target_pose',
                 state_topic='/franka_state_controller/franka_states'):
        """
        初始化 Franka 末端笛卡尔位置控制器。
        风格参考你给的 velocity controller，只把输出改成 PoseStamped。
        """
        if not rospy.core.is_initialized():
            rospy.init_node('franka_cartesian_pos_controller', anonymous=True)

        # 位置指令发布者，连接到底层 C++ pose controller
        self.pose_pub = rospy.Publisher(controller_topic, PoseStamped, queue_size=1)

        # 状态订阅者，获取包含 O_T_EE 和 q 的机械臂状态
        self.state_sub = rospy.Subscriber(
            state_topic,
            FrankaState,
            self._state_callback
        )

        # 夹爪控制与状态（沿用你的结构）
        self.gripper_move_pub = rospy.Publisher('/franka_gripper/move/goal', MoveActionGoal, queue_size=10)
        self.gripper_grasp_pub = rospy.Publisher('/franka_gripper/grasp/goal', GraspActionGoal, queue_size=10)
        self.gripper_state_sub = rospy.Subscriber('/franka_gripper/joint_states', JointState, self._gripper_state_callback)
        self.gripper_positions = []

        # 发布频率
        self.rate = rospy.Rate(10)   # 位置命令更适合低频更新，默认 10Hz

        # 当前状态
        self.current_pose_matrix = None
        self.current_pos = np.zeros(3, dtype=np.float64)
        self.current_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.current_joint_pos = None

        # 固定初始位姿（沿用你给的风格）
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

        rospy.loginfo("Franka Cartesian Position Controller 初始化完成，等待状态更新...")

    # ==================== 主要可调用函数 ====================
    def get_cartesian_pose(self):
        """
        获取当前末端笛卡尔位姿。
        返回: position (3,), quaternion (4,)
        """
        if self.current_pose_matrix is None:
            rospy.logwarn_once("尚未接收到 Franka 末端笛卡尔状态数据。")
            return None, None
        return self.current_pos.copy(), self.current_quat.copy()

    def get_joint_positions(self):
        """
        获取当前 7 维关节角。
        返回: joint_positions (7,)
        """
        if self.current_joint_pos is None:
            rospy.logwarn_once("尚未接收到 Franka 关节状态数据。")
            return None
        return self.current_joint_pos.copy()

    def set_cartesian_position(self, position, frame_id='panda_link0', stamp_now=True):
        """
        下发末端位置指令。姿态固定为 initial_quat。
        position: [x, y, z]
        """
        pos = np.asarray(position, dtype=np.float64).reshape(3,)

        msg = PoseStamped()
        msg.header.frame_id = frame_id
        msg.header.stamp = rospy.Time.now() if stamp_now else rospy.Time()

        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])

        # 当前底层 C++ 控制器只使用 position，不使用 orientation，
        # 但这里仍然补齐一个固定姿态，保持接口完整。
        msg.pose.orientation.x = float(self.initial_quat[0])
        msg.pose.orientation.y = float(self.initial_quat[1])
        msg.pose.orientation.z = float(self.initial_quat[2])
        msg.pose.orientation.w = float(self.initial_quat[3])

        self.pose_pub.publish(msg)

    def hold_cartesian_position(self, position, duration=1.0, frame_id='panda_link0'):
        """
        持续发送一段时间的位置指令，适合测试。
        """
        start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            if now - start_time >= duration:
                break
            self.set_cartesian_position(position, frame_id=frame_id, stamp_now=True)
            self.rate.sleep()

    def go_to_position(self,
                       target_pos,
                       hold_duration=0.2,
                       pos_tolerance=0.005,
                       timeout=10.0,
                       frame_id='panda_link0',
                       verbose=True):
        """
        持续发布目标位置，直到当前末端接近目标。
        说明：
        - 这里上层只是重复发目标点
        - 实际平滑/限速由底层 C++ controller 完成
        """
        target = np.asarray(target_pos, dtype=np.float64).reshape(3,)
        start_time = rospy.Time.now().to_sec()

        if verbose:
            rospy.loginfo("开始移动到目标位置: {}".format(np.round(target, 4)))

        while not rospy.is_shutdown():
            self.set_cartesian_position(target, frame_id=frame_id, stamp_now=True)

            if self.current_pose_matrix is not None:
                error = target - self.current_pos
                distance = np.linalg.norm(error)

                if verbose:
                    rospy.loginfo_throttle(
                        0.5,
                        "当前末端位置: {} | 目标: {} | 误差: {:.4f} m".format(
                            np.round(self.current_pos, 4),
                            np.round(target, 4),
                            distance
                        )
                    )

                if distance < pos_tolerance:
                    if hold_duration > 0.0:
                        self.hold_cartesian_position(target, duration=hold_duration, frame_id=frame_id)
                    rospy.loginfo("已到达目标位置，误差 {:.6f} m".format(distance))
                    return True

            if rospy.Time.now().to_sec() - start_time > timeout:
                rospy.logwarn("移动到目标位置超时")
                return False

            self.rate.sleep()

        return False

    def go_to_initial_position(self,
                               hold_duration=0.2,
                               pos_tolerance=0.005,
                               timeout=10.0):
        """
        回到初始位置。
        """
        rospy.loginfo("开始回到初始位置")
        return self.go_to_position(
            self.initial_pos,
            hold_duration=hold_duration,
            pos_tolerance=pos_tolerance,
            timeout=timeout
        )

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

    def test_print_cartesian_pose(self):
        """持续打印当前末端位姿。"""
        while not rospy.is_shutdown():
            pos, quat = self.get_cartesian_pose()
            if pos is not None:
                rospy.loginfo_throttle(
                    0.5,
                    "当前末端位置: {} | 当前姿态四元数: {}".format(
                        np.round(pos, 4),
                        np.round(quat, 4)
                    )
                )
            self.rate.sleep()

    def test_print_joint_positions(self):
        """持续打印当前 7 维关节角。"""
        while not rospy.is_shutdown():
            q = self.get_joint_positions()
            if q is not None:
                rospy.loginfo_throttle(0.5, "当前关节角: {}".format(np.round(q, 4)))
            self.rate.sleep()

    def test_gripper_toggle(self):
        """测试夹爪开合。"""
        rospy.loginfo("打开夹爪...")
        self.open_gripper()
        rospy.sleep(2.0)
        rospy.loginfo("闭合夹爪...")
        self.close_gripper()

    def test_move_single_target(self,
                                target_pos=None,
                                hold_duration=0.5,
                                pos_tolerance=0.005,
                                timeout=10.0):
        """
        测试：移动到单个目标点。
        """
        if target_pos is None:
            target_pos = self.initial_pos + np.array([0.05, 0.00, 0.00], dtype=np.float64)

        rospy.loginfo("执行单点位置测试...")
        success = self.go_to_position(
            target_pos=target_pos,
            hold_duration=hold_duration,
            pos_tolerance=pos_tolerance,
            timeout=timeout
        )
        rospy.loginfo("单点测试结果: {}".format(success))
        return success

    def test_x_axis_step(self,
                         delta_x=0.03,
                         hold_duration=2.0,
                         return_back=True):
        """
        测试：沿 X 方向移动一个小步长。
        """
        if self.current_pose_matrix is None:
            rospy.logwarn("尚未接收到状态，无法执行 test_x_axis_step")
            return False

        start_pos = self.current_pos.copy()
        target_pos = start_pos + np.array([delta_x, 0.0, 0.0], dtype=np.float64)

        rospy.loginfo("开始执行 X 轴单步测试")
        ok1 = self.go_to_position(target_pos, hold_duration=hold_duration, timeout=10.0)

        ok2 = True
        if return_back and ok1:
            rospy.loginfo("返回起始点")
            ok2 = self.go_to_position(start_pos, hold_duration=hold_duration, timeout=10.0)

        return ok1 and ok2

    def test_square_motion(self,
                           side=0.03,
                           hold_duration=0.5,
                           timeout_each=8.0):
        """
        测试：在当前位置附近走一个小方形轨迹（实际上是按离散目标点依次发送）。
        """
        if self.current_pose_matrix is None:
            rospy.logwarn("尚未接收到状态，无法执行 test_square_motion")
            return False

        p0 = self.current_pos.copy()
        points = [
            p0,
            p0 + np.array([side, 0.0, 0.0], dtype=np.float64),
            p0 + np.array([side, side, 0.0], dtype=np.float64),
            p0 + np.array([0.0, side, 0.0], dtype=np.float64),
            p0
        ]

        rospy.loginfo("开始执行小方形位置测试...")
        for i, p in enumerate(points):
            rospy.loginfo("前往第 {} 个点: {}".format(i, np.round(p, 4)))
            ok = self.go_to_position(
                p,
                hold_duration=hold_duration,
                pos_tolerance=0.005,
                timeout=timeout_each
            )
            if not ok:
                rospy.logwarn("方形测试中断于第 {} 个点".format(i))
                return False

        rospy.loginfo("小方形位置测试完成")
        return True


if __name__ == '__main__':
    try:
        arm = FrankaCartesianPositionController()
        rospy.sleep(1.0)

        # ============= 测试函数写这里 =============
        # 先确认状态正常
        arm.test_check_franka_state_once()

        # 例1：移动到初始位置前方 5cm
        target = arm.initial_pos + np.array([0.05, 0.0, 0.0], dtype=np.float64)
        arm.test_move_single_target(target_pos=target)

        # 例2：回到初始位置
        arm.go_to_initial_position()

        # 你也可以改成下面这些测试之一：
        # arm.test_print_cartesian_pose()
        # arm.test_print_joint_positions()
        # arm.test_x_axis_step(delta_x=0.03)
        # arm.test_square_motion(side=0.02)
        # arm.test_gripper_toggle()

    except rospy.ROSInterruptException:
        pass
