#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import rospy
from geometry_msgs.msg import Twist


class RemoteCartesianVelocitySender:
    def __init__(self,
                 topic='/cartesian_velocity_example_controller/target_velocity',
                 rate_hz=20):
        if not rospy.core.is_initialized():
            rospy.init_node('remote_cartesian_velocity_sender', anonymous=True)

        self.pub = rospy.Publisher(topic, Twist, queue_size=1)
        self.rate = rospy.Rate(rate_hz)

        self.linear_cmd = [0.0, 0.0, 0.0]
        self.angular_cmd = [0.0, 0.0, 0.0]
        self.lock = threading.Lock()
        self.running = True

        rospy.loginfo("RemoteCartesianVelocitySender 已启动")
        rospy.loginfo("发布话题: %s", topic)
        rospy.loginfo("发布频率: %d Hz", rate_hz)

    def set_twist(self, linear=None, angular=None):
        with self.lock:
            if linear is not None:
                self.linear_cmd = list(linear)
            if angular is not None:
                self.angular_cmd = list(angular)

    def stop(self):
        self.set_twist([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def publish_loop(self):
        while not rospy.is_shutdown() and self.running:
            msg = Twist()
            with self.lock:
                msg.linear.x = self.linear_cmd[0]
                msg.linear.y = self.linear_cmd[1]
                msg.linear.z = self.linear_cmd[2]
                msg.angular.x = self.angular_cmd[0]
                msg.angular.y = self.angular_cmd[1]
                msg.angular.z = self.angular_cmd[2]

            self.pub.publish(msg)
            self.rate.sleep()

        self.pub.publish(Twist())

    def keyboard_loop(self):
        rospy.loginfo("键盘控制已启动")
        rospy.loginfo("w/s x轴, a/d y轴, r/f z轴, j/l 绕z转, x停止, q退出")

        linear_speed = 0.02
        angular_speed = 0.20

        while not rospy.is_shutdown():
            cmd = input("请输入指令: ").strip().lower()

            if cmd == 'q':
                self.stop()
                self.running = False
                rospy.loginfo("退出控制")
                break
            elif cmd == 'x':
                self.stop()
                rospy.loginfo("停止")
            elif cmd == 'w':
                self.set_twist([ linear_speed, 0.0, 0.0], [0.0, 0.0, 0.0])
                rospy.loginfo("持续向 +X 运动")
            elif cmd == 's':
                self.set_twist([-linear_speed, 0.0, 0.0], [0.0, 0.0, 0.0])
                rospy.loginfo("持续向 -X 运动")
            elif cmd == 'a':
                self.set_twist([0.0,  linear_speed, 0.0], [0.0, 0.0, 0.0])
                rospy.loginfo("持续向 +Y 运动")
            elif cmd == 'd':
                self.set_twist([0.0, -linear_speed, 0.0], [0.0, 0.0, 0.0])
                rospy.loginfo("持续向 -Y 运动")
            elif cmd == 'r':
                self.set_twist([0.0, 0.0,  linear_speed], [0.0, 0.0, 0.0])
                rospy.loginfo("持续向 +Z 运动")
            elif cmd == 'f':
                self.set_twist([0.0, 0.0, -linear_speed], [0.0, 0.0, 0.0])
                rospy.loginfo("持续向 -Z 运动")
            elif cmd == 'j':
                self.set_twist([0.0, 0.0, 0.0], [0.0, 0.0,  angular_speed])
                rospy.loginfo("持续绕 +Z 转动")
            elif cmd == 'l':
                self.set_twist([0.0, 0.0, 0.0], [0.0, 0.0, -angular_speed])
                rospy.loginfo("持续绕 -Z 转动")
            else:
                rospy.loginfo("未知指令，输入 x 停止，q 退出")

    def run(self):
        pub_thread = threading.Thread(target=self.publish_loop)
        pub_thread.daemon = True
        pub_thread.start()

        try:
            self.keyboard_loop()
        finally:
            self.running = False
            self.stop()
            pub_thread.join(timeout=1.0)


if __name__ == '__main__':
    try:
        sender = RemoteCartesianVelocitySender(
            topic='/cartesian_velocity_example_controller/target_velocity',
            rate_hz=20
        )
        rospy.sleep(1.0)
        sender.run()
    except rospy.ROSInterruptException:
        pass