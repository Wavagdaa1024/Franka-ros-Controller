#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist


class RemoteCartesianVelocitySender:
    def __init__(self,
                 topic='/cartesian_velocity_example_controller/target_velocity',
                 rate_hz=10):
        """
        控制端专用：
        只负责向 franka 端控制器发布末端笛卡尔速度指令。
        不依赖 franka_ros / libfranka / franka_msgs。
        """
        if not rospy.core.is_initialized():
            rospy.init_node('remote_cartesian_velocity_sender', anonymous=True)

        self.pub = rospy.Publisher(topic, Twist, queue_size=1)
        self.rate = rospy.Rate(rate_hz)
        self.topic = topic

        rospy.loginfo("RemoteCartesianVelocitySender 已启动")
        rospy.loginfo("发布话题: %s", self.topic)
        rospy.loginfo("发布频率: %d Hz", rate_hz)

    def send_twist(self, linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0)):
        msg = Twist()
        msg.linear.x = linear[0]
        msg.linear.y = linear[1]
        msg.linear.z = linear[2]
        msg.angular.x = angular[0]
        msg.angular.y = angular[1]
        msg.angular.z = angular[2]
        self.pub.publish(msg)

    def stop(self):
        self.send_twist((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def send_constant_for_duration(self,
                                   linear=(0.0, 0.0, 0.0),
                                   angular=(0.0, 0.0, 0.0),
                                   duration=2.0):
        """
        以固定频率持续发送一段时间，避免底层控制器因超时自动停下。
        """
        start_t = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            if now - start_t >= duration:
                break
            self.send_twist(linear, angular)
            self.rate.sleep()

        self.stop()

    def demo_move_x(self, velocity=0.02, duration=2.0):
        """
        示例：沿 x 方向移动
        """
        rospy.loginfo("开始沿 X 方向发送速度: %.4f m/s, 持续 %.2f s", velocity, duration)
        self.send_constant_for_duration(
            linear=(velocity, 0.0, 0.0),
            angular=(0.0, 0.0, 0.0),
            duration=duration
        )
        rospy.loginfo("运动结束，已停止")

    def keyboard_loop(self):
        """
        简单键盘控制：
        w/s: x+ / x-
        a/d: y+ / y-
        r/f: z+ / z-
        j/l: wz+ / wz-
        q: 退出
        其它键：停止
        """
        rospy.loginfo("键盘控制已启动")
        rospy.loginfo("w/s x轴, a/d y轴, r/f z轴, j/l 末端绕z转动, q退出, 其它键停止")

        while not rospy.is_shutdown():
            cmd = input("请输入指令: ").strip().lower()

            if cmd == 'q':
                self.stop()
                rospy.loginfo("退出控制")
                break
            elif cmd == 'w':
                self.send_constant_for_duration((0.02, 0.0, 0.0), (0.0, 0.0, 0.0), 0.2)
            elif cmd == 's':
                self.send_constant_for_duration((-0.02, 0.0, 0.0), (0.0, 0.0, 0.0), 0.2)
            elif cmd == 'a':
                self.send_constant_for_duration((0.0, 0.02, 0.0), (0.0, 0.0, 0.0), 0.2)
            elif cmd == 'd':
                self.send_constant_for_duration((0.0, -0.02, 0.0), (0.0, 0.0, 0.0), 0.2)
            elif cmd == 'r':
                self.send_constant_for_duration((0.0, 0.0, 0.02), (0.0, 0.0, 0.0), 0.2)
            elif cmd == 'f':
                self.send_constant_for_duration((0.0, 0.0, -0.02), (0.0, 0.0, 0.0), 0.2)
            elif cmd == 'j':
                self.send_constant_for_duration((0.0, 0.0, 0.0), (0.0, 0.0, 0.2), 0.2)
            elif cmd == 'l':
                self.send_constant_for_duration((0.0, 0.0, 0.0), (0.0, 0.0, -0.2), 0.2)
            else:
                self.stop()
                rospy.loginfo("停止")


if __name__ == '__main__':
    try:
        sender = RemoteCartesianVelocitySender(
            topic='/cartesian_velocity_example_controller/target_velocity',
            rate_hz=10
        )

        rospy.sleep(1.0)

        # 二选一：
        # 1) 直接测试一段固定运动
        # sender.demo_move_x(velocity=0.02, duration=2.0)

        # 2) 键盘控制
        sender.keyboard_loop()

    except rospy.ROSInterruptException:
        pass
