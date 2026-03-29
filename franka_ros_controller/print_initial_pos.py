#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from franka_msgs.msg import FrankaState
from tf.transformations import quaternion_from_matrix


def main():
    rospy.init_node('get_initial_pose_once', anonymous=True)

    rospy.loginfo("等待 /franka_state_controller/franka_states ...")
    msg = rospy.wait_for_message(
        '/franka_state_controller/franka_states',
        FrankaState,
        timeout=10.0
    )

    # Franka 的 O_T_EE 需要按列优先重排
    matrix = np.array(msg.O_T_EE, dtype=np.float64).reshape((4, 4), order='F')
    pos = matrix[:3, 3].copy()
    quat = np.array(quaternion_from_matrix(matrix), dtype=np.float64)

    # 打印详细数据
    print("\n===== Current EE Pose =====")
    print("matrix =")
    print(np.array2string(matrix, precision=10, separator=', '))
    print("pos =")
    print(np.array2string(pos, precision=10, separator=', '))
    print("quat =")
    print(np.array2string(quat, precision=10, separator=', '))

    # 直接打印成你可以粘贴到类里的形式
    print("\n===== Paste Into Your Class =====")
    print("self.initial_pose_matrix = np.array({}, dtype=np.float64)".format(
        np.array2string(matrix, precision=10, separator=', ')
    ))
    print("self.initial_pos = np.array({}, dtype=np.float64)".format(
        np.array2string(pos, precision=10, separator=', ')
    ))
    print("self.initial_quat = np.array({}, dtype=np.float64)".format(
        np.array2string(quat, precision=10, separator=', ')
    ))
    print("self._initial_pose_saved = True")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSException as e:
        print("Failed to get Franka state: {}".format(e))

