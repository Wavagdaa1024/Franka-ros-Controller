#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState

def move_to_start():
    # 初始化节点
    rospy.init_node('move_to_start')

    # 1. 读取要控制的关节名称
    joint_names = [
        'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
        'panda_joint5', 'panda_joint6', 'panda_joint7'
    ]

    # 2. 从参数服务器读取目标姿态 (由 launch 文件中的 yaml 加载)
    target_pose = []
    for name in joint_names:
        param_name = '~joint_pose/' + name
        if not rospy.has_param(param_name):
            rospy.logerr("未找到目标姿态参数: %s" % param_name)
            return
        target_pose.append(rospy.get_param(param_name))
        
    # ==================== 新增：打印读取到的初始姿态 ====================
    rospy.loginfo("成功读取初始姿态参数！")
    for name, pos in zip(joint_names, target_pose):
        rospy.loginfo("  %s: %.4f rad", name, pos)
    rospy.loginfo("--------------------------------------------------")
    # =================================================================

    # 读取速度限制参数
    max_dq = rospy.get_param('~max_dq', 0.2)

    # 3. 连接到 Action Server (对应 launch 文件中的 remap)
    action_topic = rospy.resolve_name('~follow_joint_trajectory')
    rospy.loginfo("正在连接到轨迹控制器 Action Server: %s", action_topic)
    client = actionlib.SimpleActionClient(action_topic, FollowJointTrajectoryAction)
    
    # 设置超时时间，防止无限卡死
    if not client.wait_for_server(rospy.Duration(5.0)):
        rospy.logerr("无法连接到 Action Server，请检查控制器是否已启动。")
        return
    rospy.loginfo("成功连接 Action Server！")

    # 4. 获取当前的关节状态，用于计算移动时间
    rospy.loginfo("正在获取当前关节状态...")
    joint_state_topic = rospy.resolve_name('~joint_states')
    try:
        current_state = rospy.wait_for_message(joint_state_topic, JointState, timeout=3.0)
    except rospy.ROSException:
        rospy.logerr("无法获取当前关节状态，请检查状态发布话题。")
        return

    # 对齐当前关节位置的顺序
    current_pos = [0.0] * 7
    for i, name in enumerate(joint_names):
        if name in current_state.name:
            idx = current_state.name.index(name)
            current_pos[i] = current_state.position[idx]
        else:
            rospy.logerr("在当前状态中找不到关节 %s" % name)
            return

    # 5. 根据最大关节偏差和限制速度，计算所需的最短安全时间
    max_diff = max([abs(t - c) for t, c in zip(target_pose, current_pos)])
    time_from_start = max_diff / max_dq
    # 保底限制：如果距离极短，也至少给 1 秒的时间，防止底层加速度过大报错
    if time_from_start < 1.0:
        time_from_start = 1.0

    rospy.loginfo("计算出的安全运动时间为: %.2f 秒", time_from_start)
    
    # 6. 构建并发送目标轨迹
    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = joint_names

    point = JointTrajectoryPoint()
    point.positions = target_pose
    
    # 🌟 新增这两行：明确告诉控制器，到达终点时所有关节的速度和加速度必须降为 0 🌟
    point.velocities = [0.0] * 7
    point.accelerations = [0.0] * 7
    
    point.time_from_start = rospy.Duration(time_from_start)
    goal.trajectory.points.append(point)

    rospy.loginfo("开始移动机械臂！")
    client.send_goal(goal)
    
    # 等待执行结果
    client.wait_for_result()
    rospy.loginfo("动作执行完毕！")

if __name__ == '__main__':
    try:
        move_to_start()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点被意外终止。")
