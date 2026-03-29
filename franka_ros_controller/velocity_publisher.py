#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist


def main():
    rospy.init_node('my_velocity_commander_node')
    
    # 注意话题名字要和你的实际控制器对应
    pub = rospy.Publisher('/cartesian_velocity_example_controller/target_velocity', Twist, queue_size=1)
    
    # 【频率调高】：200Hz 足够满足平滑控制了
    rate = rospy.Rate(200) 
    vel_msg = Twist()

    # 目标速度和当前速度
    target_v = 0.05
    current_v = 0.0
    
    # 【核心魔法：平滑步长】
    # 频率是 200Hz，假设 step=0.0002，那么 1 秒钟速度会增加 0.0002 * 200 = 0.04 m/s
    # 这个加速度非常温柔，绝对不会触发 Franka 的急停
    step = 0.0002 

    rospy.loginfo("开始执行平滑加速 (Soft Start)...")

    try:
        while not rospy.is_shutdown():
            # 1. 缓慢踩油门逻辑
            if current_v < target_v:
                current_v += step
                # 如果超过了目标速度，就卡在目标速度
                if current_v > target_v:
                    current_v = target_v
            
            # 2. 赋值并发送
            vel_msg.linear.x = current_v 
            vel_msg.linear.y = 0.0
            vel_msg.linear.z = 0.0
            vel_msg.angular.x = 0.0
            vel_msg.angular.y = 0.0
            vel_msg.angular.z = 0.0
            
            pub.publish(vel_msg)
            rate.sleep()
            
    except rospy.ROSInterruptException:
        # 当按下 Ctrl+C 时，我们本来应该写一个平滑减速的过程。
        # 但因为 ROS 会直接杀掉循环，所以通常需要依靠 C++ 底层的控制器来做低通滤波。
        # 这里为了测试，我们什么都不做，看看看门狗的瞬间归零会不会再次触发急停。
        rospy.loginfo("停止发送。")







if __name__ == '__main__':
    
    main()
