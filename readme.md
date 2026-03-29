# Franka ROS Controller

基于 ROS 的 Franka 机械臂遥操作与数据采集项目，当前支持：

- Touch 设备主从控制
- Franka 末端笛卡尔速度控制
- RealSense 图像采集
- 轨迹数据记录
- 录制轨迹重放验证

## 目录结构

```text
franka_ros_controller/
├── Recoder_main.py
├── replay_recorded_trajectory.py
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

## 模块说明

- `Recoder_main.py`：统一调度 Touch、Franka、RealSense、主从控制与数据记录
- `replay_recorded_trajectory.py`：读取录制轨迹并重放，用于验证数据集正确性
- `controller/touch_franka_teleop_controller.py`：Touch 到 Franka 的主从控制逻辑
- `controller/teleop_dataset_recorder.py`：记录图像、关节角、末端位姿、Touch 位置等数据
- `controller/base_controller/touch_controller.py`：Touch 设备底层封装
- `controller/base_controller/franka_cartesian_vel_controller.py`：Franka 末端笛卡尔速度控制与夹爪控制封装
- `controller/base_controller/RealSenseCamera.py`：RealSense 彩色图像采集与显示

## 运行方式

启动机器人控制器后运行：

```bash
python Recoder_main.py
```

轨迹重放验证：

```bash
python replay_recorded_trajectory.py
```

## 控制说明

### Touch

- 上按钮：切换主从控制状态
- 下按钮：切换夹爪开合状态

### Keyboard

- `s`：开始记录
- `e`：结束并保存
- `p`：打印当前状态
- `q`：退出程序

### Replay

- `r`：开始轨迹重放
- `q`：停止并退出

## 数据说明

记录数据默认保存在 `real_dir/`，主要包含：

- 相机彩色图像
- 机械臂关节角
- 末端笛卡尔位姿
- Touch 末端位置
- 动作数据

### HDF5 结构

```text
episode_x.hdf5
├── attributes
│   └── sim = False
├── observations
│   ├── images
│   │   └── top
│   ├── qpos
│   ├── cartesian_pose
│   └── touch_position
└── action
```

### 数据格式

- `observations/qpos`
  - `[joint1, joint2, joint3, joint4, joint5, joint6, joint7, gripper_width]`
- `observations/cartesian_pose`
  - `[x, y, z, qx, qy, qz, qw, gripper_width]`
- `observations/touch_position`
  - `[tx, ty, tz]`

## 轨迹重放说明

轨迹重放主要用于验证：

- 末端位置轨迹是否合理
- 夹爪状态是否正确
- 数据集录制是否基本可信

当前版本重放内容包括：

- 末端位置轨迹
- 夹爪开合状态

当前版本不重放末端姿态控制。

## Git 说明

项目已忽略以下内容：

- `__pycache__/`
- `*.pyc`
- `*.pyo`
- `real_dir/` 中录制生成的轨迹文件
- `.catkin_tools/`
- `build/`
- `devel/`
- `logs/`

因此：

- `real_dir/` 目录会保留
- 录制生成的 `.hdf5` 文件不会上传到 GitHub

## 日志

- [260329] 创建仓库并上传 `/franka_ros_controller` 与 `/src`；完成 Touch、Franka、RealSense 底层类、主从控制、数据记录、轨迹重放与 `real_dir` 忽略规则配置

