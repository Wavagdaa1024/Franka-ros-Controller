#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

import cv2
import h5py
import numpy as np


EPISODE_PATTERN = re.compile(r"episode_(\d+)\.hdf5$")


class FrankaRealDataConverter:
    #============ 初始化与配置 ================

    def __init__(self,
                 input_dir="real_dir",
                 output="diffusion_policy_real/data/franka_image/image.hdf5",
                 camera_name="top",
                 camera_key="camera_0",
                 width=84,
                 height=84,
                 target_hz=None,
                 gripper_close_threshold=0.04,
                 keep_touch=False):
        """
        初始化 Franka 实机数据转换器。

        参数:
            input_dir: 原始 episode 文件所在目录
            output: 输出 robomimic 风格 HDF5 路径
            camera_name: 原始数据中的相机名称
            camera_key: 转换后数据中的相机键名
            width: 输出图像宽度
            height: 输出图像高度
            target_hz: 可选目标频率，用于下采样
            gripper_close_threshold: 夹爪闭合阈值
            keep_touch: 是否保留 touch_position
        """
        self.input_dir = Path(input_dir)
        self.output = Path(output)
        self.camera_name = camera_name
        self.camera_key = camera_key
        self.width = width
        self.height = height
        self.target_hz = target_hz
        self.gripper_close_threshold = gripper_close_threshold
        self.keep_touch = keep_touch

    #============ 文件与频率处理 ================

    def list_episode_files(self):
        """按 episode 编号顺序收集输入目录中的 episode 文件。"""
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        episode_files = []
        for path in self.input_dir.glob("episode_*.hdf5"):
            match = EPISODE_PATTERN.match(path.name)
            if match is None:
                continue
            episode_files.append((int(match.group(1)), path))

        episode_files.sort(key=lambda x: x[0])
        return [path for _, path in episode_files]

    def get_source_frequency(self, root, timestamps):
        """优先从文件属性读取频率，否则根据时间戳估计采样频率。"""
        if "frequency" in root.attrs:
            return float(root.attrs["frequency"])

        if timestamps is not None and len(timestamps) >= 2:
            dt = np.median(np.diff(timestamps))
            if dt > 0:
                return 1.0 / dt

        return None

    def build_sample_indices(self, timestamps, n_steps, source_hz=None):
        """根据目标频率生成下采样索引；若不需要下采样则返回全索引。"""
        if n_steps <= 0:
            return np.zeros((0,), dtype=np.int64)

        if self.target_hz is None or source_hz is None or self.target_hz >= source_hz:
            return np.arange(n_steps, dtype=np.int64)

        if timestamps is not None and len(timestamps) == n_steps:
            rel_time = timestamps - timestamps[0]
            duration = rel_time[-1]
            target_dt = 1.0 / self.target_hz
            target_times = np.arange(0.0, duration + 1e-9, target_dt, dtype=np.float64)
            indices = np.searchsorted(rel_time, target_times, side="left")
            indices = np.clip(indices, 0, n_steps - 1)
            indices = np.unique(indices)
            if indices[-1] != (n_steps - 1):
                indices = np.append(indices, n_steps - 1)
            return indices.astype(np.int64)

        stride = max(int(round(source_hz / self.target_hz)), 1)
        indices = np.arange(0, n_steps, stride, dtype=np.int64)
        if indices[-1] != (n_steps - 1):
            indices = np.append(indices, n_steps - 1)
        return indices

    #============ 数据处理 ================

    def resize_images(self, images):
        """将图像序列统一缩放到训练分辨率。"""
        resized = []
        for image in images:
            resized.append(
                cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
            )
        return np.asarray(resized, dtype=np.uint8)

    def build_actions(self, selected_pos, selected_gripper_width, raw_action, selected_indices):
        """生成训练动作，格式为 [dx, dy, dz, gripper_cmd]。"""
        n_steps = len(selected_indices)
        actions = np.zeros((n_steps, 4), dtype=np.float32)

        if n_steps > 1:
            actions[:-1, :3] = (selected_pos[1:] - selected_pos[:-1]).astype(np.float32)

        if raw_action is not None and raw_action.ndim == 2 and raw_action.shape[1] >= 4:
            actions[:, 3] = raw_action[selected_indices, -1].astype(np.float32)
        else:
            actions[:, 3] = (
                selected_gripper_width.reshape(-1) < self.gripper_close_threshold
            ).astype(np.float32)

        return actions

    def load_episode(self, episode_path):
        """读取单个原始 episode，并转换成统一的中间表示。"""
        with h5py.File(str(episode_path), "r") as root:
            obs = root["observations"]
            images = np.asarray(obs["images"][self.camera_name], dtype=np.uint8)
            robot_joint = np.asarray(obs["robot_joint"], dtype=np.float32)
            robot_eef_pos = np.asarray(obs["robot_eef_pos"], dtype=np.float32)
            robot_eef_quat = np.asarray(obs["robot_eef_quat"], dtype=np.float32)
            robot_gripper_width = np.asarray(obs["robot_gripper_width"], dtype=np.float32)
            touch_position = np.asarray(obs["touch_position"], dtype=np.float32) if "touch_position" in obs else None
            timestamps = np.asarray(obs["timestamp"], dtype=np.float64).reshape(-1) if "timestamp" in obs else None
            raw_action = np.asarray(root["action"], dtype=np.float32) if "action" in root else None

            source_hz = self.get_source_frequency(root, timestamps)
            sample_indices = self.build_sample_indices(
                timestamps=timestamps,
                n_steps=len(robot_eef_pos),
                source_hz=source_hz
            )

            episode = dict()
            episode["robot0_eef_pos"] = robot_eef_pos[sample_indices]
            episode["robot0_eef_quat"] = robot_eef_quat[sample_indices]
            episode["robot0_gripper_qpos"] = robot_gripper_width[sample_indices]
            episode["robot0_joint_qpos"] = robot_joint[sample_indices]
            episode["timestamp"] = timestamps[sample_indices].astype(np.float32) if timestamps is not None else None
            if self.keep_touch and touch_position is not None:
                episode["teacher_touch_pos"] = touch_position[sample_indices]

            episode["actions"] = self.build_actions(
                selected_pos=episode["robot0_eef_pos"],
                selected_gripper_width=episode["robot0_gripper_qpos"],
                raw_action=raw_action,
                selected_indices=sample_indices
            )
            episode["images"] = self.resize_images(images[sample_indices])
            episode["n_steps"] = len(sample_indices)
            episode["source_hz"] = source_hz
            episode["target_hz"] = float(self.target_hz) if self.target_hz is not None else source_hz

        return episode

    #============ 数据写出 ================

    def write_demo_group(self, data_group, demo_idx, episode):
        """将单个 episode 写入 robomimic 风格的 demo 组。"""
        demo = data_group.create_group(f"demo_{demo_idx}")
        demo.attrs["num_samples"] = int(episode["n_steps"])

        demo.create_dataset("actions", data=episode["actions"], dtype=np.float32)

        obs_group = demo.create_group("obs")
        obs_group.create_dataset(self.camera_key, data=episode["images"], dtype=np.uint8)
        obs_group.create_dataset("robot0_eef_pos", data=episode["robot0_eef_pos"], dtype=np.float32)
        obs_group.create_dataset("robot0_eef_quat", data=episode["robot0_eef_quat"], dtype=np.float32)
        obs_group.create_dataset("robot0_gripper_qpos", data=episode["robot0_gripper_qpos"], dtype=np.float32)
        obs_group.create_dataset("robot0_joint_qpos", data=episode["robot0_joint_qpos"], dtype=np.float32)

        if episode["timestamp"] is not None:
            obs_group.create_dataset("timestamp", data=episode["timestamp"], dtype=np.float32)
        if "teacher_touch_pos" in episode:
            obs_group.create_dataset("teacher_touch_pos", data=episode["teacher_touch_pos"], dtype=np.float32)

    def convert(self):
        """批量转换目录下所有 episode 文件并写出目标 HDF5 数据集。"""
        episode_files = self.list_episode_files()
        if len(episode_files) == 0:
            raise RuntimeError(f"No episode_*.hdf5 found in {self.input_dir}")

        self.output.parent.mkdir(parents=True, exist_ok=True)

        total_steps = 0
        source_hz_list = []

        with h5py.File(str(self.output), "w") as root:
            data_group = root.create_group("data")

            for demo_idx, episode_path in enumerate(episode_files):
                episode = self.load_episode(episode_path)
                self.write_demo_group(
                    data_group=data_group,
                    demo_idx=demo_idx,
                    episode=episode
                )
                total_steps += episode["n_steps"]
                if episode["source_hz"] is not None:
                    source_hz_list.append(float(episode["source_hz"]))

            data_group.attrs["total"] = int(total_steps)
            data_group.attrs["env_args"] = json.dumps({
                "type": "franka_real_world_dataset",
                "env_name": "FrankaRealWorld",
                "camera_key": self.camera_key,
                "source_camera_name": self.camera_name,
                "source_hz_mean": float(np.mean(source_hz_list)) if source_hz_list else None,
                "target_hz": self.target_hz
            })

        print("Converted episodes:", len(episode_files))
        print("Output:", str(self.output))
        print("Total steps:", total_steps)
        if self.target_hz is not None:
            print("Target frequency:", self.target_hz, "Hz")


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Convert Franka raw episode HDF5 files to a robomimic-style image HDF5 dataset."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="real_dir",
        help="Directory containing episode_*.hdf5"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="diffusion_policy_real/data/franka_image/image.hdf5",
        help="Output robomimic-style HDF5 path"
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="top",
        help="Source image key under observations/images"
    )
    parser.add_argument(
        "--camera_key",
        type=str,
        default="camera_0",
        help="Target image key in converted dataset"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=84,
        help="Target image width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=84,
        help="Target image height"
    )
    parser.add_argument(
        "--target_hz",
        type=float,
        default=None,
        help="Optional target frequency for downsampling, e.g. 10. Keep original if omitted."
    )
    parser.add_argument(
        "--gripper_close_threshold",
        type=float,
        default=0.04,
        help="Threshold used to derive gripper command when needed"
    )
    parser.add_argument(
        "--keep_touch",
        action="store_true",
        help="Also save touch_position into obs/teacher_touch_pos"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    converter = FrankaRealDataConverter(
        input_dir=args.input_dir,
        output=args.output,
        camera_name=args.camera_name,
        camera_key=args.camera_key,
        width=args.width,
        height=args.height,
        target_hz=args.target_hz,
        gripper_close_threshold=args.gripper_close_threshold,
        keep_touch=args.keep_touch
    )
    converter.convert()
