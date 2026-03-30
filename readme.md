# Franka ROS Controller

基于 ROS 的 Franka 机械臂遥操作与数据采集项目，当前支持：

- Touch 设备主从控制
- Franka 末端笛卡尔速度控制
- RealSense 图像采集
- 轨迹数据记录
- 录制轨迹重放验证

### 文件目录

```text
franka_ros_controller/
├── Recoder_main.py                  # 主从控制与数据记录
├── replay_recorded_trajectory.py    # 读取录制轨迹并重放            
├── real_dir/
├── sample_test/
│   └── print_initial_pos.py
├── controller/
│   ├── touch_franka_teleop_controller.py
│   ├── teleop_dataset_recorder.py
│   └── base_controller/
│       ├── touch_controller.py
│       ├── franka_cartesian_vel_controller.py
│       └── RealSenseCamera.py
```

### 数据集结构

```text
episode_x.hdf5
├── attributes
│   ├── sim = False
│   └── frequency = 20
├── observations
│   ├── images
│   │   └── top
│   ├── robot_joint          # [q1, q2, q3, q4, q5, q6, q7]
│   ├── robot_eef_pos        # [x, y, z]
│   ├── robot_eef_quat       # [qx, qy, qz, qw]
│   ├── robot_gripper_width  # [gripper_width]
│   ├── touch_position       # [tx, ty, tz]
│   └── timestamp            # [timestamp]
└── action                   # [dx, dy, dz, gripper_cmd]

```


## 日志

- [260329] 创建仓库并上传 `/franka_ros_controller` 与 `/src`；完成 Touch、Franka、RealSense 底层类、主从控制、数据记录、轨迹重放与 `real_dir` 忽略规则配置

