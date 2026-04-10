#!/usr/bin/env python
# -*- coding: utf-8 -*-

import base64
import json
import pathlib
from collections import deque

import cv2
import hydra
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from omegaconf import OmegaConf

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


class DiffusionWorkspacePolicyLoader:
    #============ 初始化与配置 ================

    def __init__(self,
                 config_dir,
                 config_name,
                 checkpoint_path,
                 device=None):
        """
        使用 diffusion workspace 加载训练好的策略。

        流程:
            1. 读取 Hydra 配置
            2. 实例化 workspace
            3. 加载 checkpoint
            4. 实例化 dataset 并恢复 normalizer
            5. 返回 eval 模式 policy
        """
        self.config_dir = str(config_dir)
        self.config_name = str(config_name)
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    def load_policy(self):
        """按 workspace 训练流程恢复 policy 和关键配置。"""
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        with hydra.initialize_config_dir(version_base=None, config_dir=self.config_dir):
            cfg = hydra.compose(config_name=self.config_name)
        OmegaConf.resolve(cfg)

        workspace_cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = workspace_cls(cfg)

        print(f"[INFO] Loading checkpoint: {self.checkpoint_path}")
        workspace.load_checkpoint(path=self.checkpoint_path)

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        if not isinstance(dataset, BaseImageDataset):
            raise TypeError("cfg.task.dataset did not instantiate to BaseImageDataset.")

        normalizer = dataset.get_normalizer()
        workspace.model.set_normalizer(normalizer)
        workspace.model.to(self.device)
        workspace.model.eval()

        return workspace.model, cfg


class DiffusionPolicyInferenceBackend:
    #============ 初始化与配置 ================

    def __init__(self,
                 policy,
                 device=None,
                 n_obs_steps=2,
                 image_size=(640, 480),
                 image_key="camera_0",
                 pos_key="robot0_eef_pos",
                 quat_key="robot0_eef_quat",
                 gripper_key="robot0_gripper_qpos"):
        """
        云端推理后端。

        参数:
            policy: 已加载好的 diffusion policy，需要提供 predict_action(obs_dict)
            n_obs_steps: 模型需要的观测步数
            image_size: 训练时图像分辨率，格式 (width, height)
        """
        self.policy = policy
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_obs_steps = n_obs_steps
        self.image_width = int(image_size[0])
        self.image_height = int(image_size[1])

        self.image_key = image_key
        self.pos_key = pos_key
        self.quat_key = quat_key
        self.gripper_key = gripper_key

        self.policy.to(self.device)
        self.policy.eval()

    #============ 观测缓存 ================

    def create_history(self):
        """为单个客户端会话创建观测缓存。"""
        return {
            self.image_key: deque(maxlen=self.n_obs_steps),
            self.pos_key: deque(maxlen=self.n_obs_steps),
            self.quat_key: deque(maxlen=self.n_obs_steps),
            self.gripper_key: deque(maxlen=self.n_obs_steps),
        }

    def reset_history(self, history):
        """清空客户端缓存。"""
        for buffer in history.values():
            buffer.clear()

    #============ 数据预处理 ================

    def preprocess_image(self, image_b64):
        """将 base64 JPEG 解码为模型输入图像。"""
        img_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("Failed to decode incoming image.")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if frame_rgb.shape[1] != self.image_width or frame_rgb.shape[0] != self.image_height:
            frame_rgb = cv2.resize(
                frame_rgb,
                (self.image_width, self.image_height),
                interpolation=cv2.INTER_AREA
            )

        image = frame_rgb.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return image

    def parse_payload(self, payload):
        """解析客户端 JSON 为单步观测。"""
        return {
            self.image_key: self.preprocess_image(payload["image_b64"]),
            self.pos_key: np.asarray(payload[self.pos_key], dtype=np.float32),
            self.quat_key: np.asarray(payload[self.quat_key], dtype=np.float32),
            self.gripper_key: np.asarray(payload[self.gripper_key], dtype=np.float32),
        }

    def append_frame_obs(self, history, frame_obs):
        """将单步观测写入缓存。"""
        for key, value in frame_obs.items():
            history[key].append(value)

    def build_model_obs(self, history):
        """
        将缓存整理成模型输入。

        若不足 n_obs_steps，则使用首帧左填充。
        """
        obs_dict = {}
        for key, buffer in history.items():
            values = list(buffer)
            if len(values) == 0:
                raise RuntimeError(f"Observation buffer for {key} is empty.")

            while len(values) < self.n_obs_steps:
                values.insert(0, values[0].copy())

            stacked = np.stack(values[-self.n_obs_steps:], axis=0)
            tensor = torch.from_numpy(stacked).unsqueeze(0).to(self.device)
            obs_dict[key] = tensor

        return obs_dict

    #============ 模型推理 ================

    @torch.inference_mode()
    def predict_actions(self, history):
        """根据当前缓存执行一次模型推理。"""
        obs_dict = self.build_model_obs(history)
        result = self.policy.predict_action(obs_dict)
        actions = result["action"].detach().cpu().numpy()
        return actions.tolist()


class CloudInferenceServer:
    #============ 初始化与配置 ================

    def __init__(self, backend):
        """
        WebSocket 云端推理服务。

        client -> server:
            image_b64
            robot0_eef_pos
            robot0_eef_quat
            robot0_gripper_qpos
            step
            timestamp
            reset_history

        server -> client:
            actions
            step
            buffered_obs
        """
        self.backend = backend
        self.app = FastAPI()
        self._register_routes()

    def _register_routes(self):
        """注册 HTTP 和 WebSocket 路由。"""

        @self.app.get("/healthz")
        async def healthz():
            return {"status": "ok"}

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            print("[INFO] Client connected.")

            history = self.backend.create_history()

            try:
                while True:
                    raw_msg = await websocket.receive_text()
                    payload = json.loads(raw_msg)

                    if payload.get("reset_history", False):
                        self.backend.reset_history(history)

                    frame_obs = self.backend.parse_payload(payload)
                    self.backend.append_frame_obs(history, frame_obs)
                    actions = self.backend.predict_actions(history)

                    await websocket.send_json({
                        "actions": actions,
                        "step": payload.get("step", 0),
                        "buffered_obs": len(history[self.backend.image_key]),
                    })

            except WebSocketDisconnect:
                print("[INFO] Client disconnected.")
            except Exception as exc:
                print(f"[ERROR] WebSocket session failed: {exc}")
                try:
                    await websocket.close(code=1011)
                except Exception:
                    pass

    def run(self, host="0.0.0.0", port=6006):
        """启动 uvicorn 服务。"""
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    loader = DiffusionWorkspacePolicyLoader(
        config_dir="/root/autodl-tmp/ACTfranka/diffusion_policy/config",
        config_name="012train_diffusion_transformer_hybrid_workspace",
        checkpoint_path="/root/autodl-tmp/ACTfranka/data/outputs/2026.03.10/15.27.36_train_diffusion_transformer_hybrid_square_image/checkpoints/latest.ckpt",
        device="cuda:0",
    )

    policy, cfg = loader.load_policy()

    backend = DiffusionPolicyInferenceBackend(
        policy=policy,
        device=cfg.training.device,
        n_obs_steps=cfg.n_obs_steps,
        image_size=(640, 480),
        image_key="camera_0",
        pos_key="robot0_eef_pos",
        quat_key="robot0_eef_quat",
        gripper_key="robot0_gripper_qpos",
    )

    server = CloudInferenceServer(backend)
    server.run(host="0.0.0.0", port=6006)

