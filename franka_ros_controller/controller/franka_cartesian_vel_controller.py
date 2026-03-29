#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from franka_msgs.msg import FrankaState
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

        # 状态订阅者，获取包含 O_T_EE (末端位姿) 的机械臂状态
        self.state_sub = rospy.Subscriber(
            '/franka_state_controller/franka_states', 
            FrankaState, 
            self._state_callback
        )

        # 控制频率设为 200Hz 以满足平滑要求
        self.rate = rospy.Rate(200) 
        
        # 存储笛卡尔状态的内部变量
        self.current_pose_matrix = None 
        self.current_pos = np.zeros(3)  
        self.current_quat = np.array([0.0, 0.0, 0.0, 1.0]) 

        # 固定初始位姿（你刚刚采集到的）
        self.initial_pose_matrix = np.array([
            [ 9.9999032943e-01, -2.2293657136e-04, -1.9555093429e-04,  3.0687314493e-01],
            [-2.2299660341e-04, -9.9999030141e-01, -3.0702421719e-04, -6.6142970518e-05],
            [-1.9548059080e-04,  3.0706485529e-04, -9.9999993375e-01,  4.8692429186e-01],
            [ 0.0000000000e+00,  0.0000000000e+00,  0.0000000000e+00,  1.0000000000e+00]
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

    def set_cartesian_twist(self, linear=[0.0, 0.0, 0.0], angular=[0.0, 0.0, 0.0]):
        """
        下发笛卡尔速度指令。
        底层 C++ 已实现 0.005 的一阶低通滤波，因此这里可以直接输入目标速度。
        """
        vel_msg = Twist()
        vel_msg.linear.x, vel_msg.linear.y, vel_msg.linear.z = linear
        vel_msg.angular.x, vel_msg.angular.y, vel_msg.angular.z = angular
        self.vel_pub.publish(vel_msg)

    def stop_motion(self):
        """ 紧急停止末端运动 """
        self.set_cartesian_twist([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


    def go_to_initial_position(self, kp=1.2, ki=0.08, max_linear_vel=0.05, pos_tolerance=0.001, timeout=15.0, integral_limit=0.05):
        """使用 PI 控制器将末端平滑地回到初始位置"""
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
        [内部调用] 解析 FrankaState 消息，提取末端执行器在基座坐标系下的位姿 (O_T_EE)。
        """
        # O_T_EE 是一维数组，需按列优先 (Fortran order) 转换为 4x4 变换矩阵
        matrix = np.array(msg.O_T_EE).reshape(4, 4, order='F') 
        self.current_pose_matrix = matrix
        
        # 提取末端三维坐标 (x, y, z)
        self.current_pos = matrix[:3, 3]
        
        # 提取旋转矩阵并转为四元数 (x, y, z, w)，方便后续与 Touch 设备的姿态对齐
        self.current_quat = quaternion_from_matrix(matrix)


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
                    
                self.set_cartesian_twist(linear=[current_v, 0.0, 0.0])
                
                # 实时反馈末端位置
                pos, _ = self.get_cartesian_pose()
                if pos is not None:
                    rospy.loginfo_throttle(0.5, "当前末端 X 坐标: {:.4f} m".format(pos[0]))
                
                self.rate.sleep()
                
        except rospy.ROSInterruptException:
            # 依靠 C++ 底层控制器的低通滤波和看门狗来处理中断
            rospy.loginfo("测试手动中断，底层看门狗将自动归零速度。")

if __name__ == '__main__':
    
    try:
        arm = FrankaCartesianVelocityController()
        rospy.sleep(1.0) # 等待订阅器建立连接

        # =============测试函数写这里=============

        # 测试末端速度控制程序
        arm.test_soft_start_x_axis()

        # 测试回到初始位置
        # arm.go_to_initial_position()



    except rospy.ROSInterruptException:
        pass
        