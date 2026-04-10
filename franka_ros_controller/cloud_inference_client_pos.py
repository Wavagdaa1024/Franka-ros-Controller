#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import base64
import json
import os
import select
import sys
import termios
import time
import tty
from typing import Any, Dict, Optional

import cv2
import h5py
import numpy as np
import rospy
import websockets

from controller.base_controller.RealSenseCamera import RealSenseCamera
from controller.base_controller.franka_cartesian_vel_controller import FrankaCartesianVelocityController


class KeyboardController:
    def __init__(self):
        self._fd = None
        self._old_settings = None

    def setup(self):
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

    def restore(self):
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        self._fd = None
        self._old_settings = None

    def read_key(self) -> Optional[str]:
        if self._fd is None:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


class CloudInferenceTransport:
    def __init__(self, server_uri: str, jpeg_quality: int = 90, recv_timeout: float = 5.0):
        self.server_uri = server_uri
        self.jpeg_quality = jpeg_quality
        self.recv_timeout = recv_timeout

    def encode_image(self, image_bgr: np.ndarray) -> str:
        ok, buffer = cv2.imencode(
            ".jpg",
            image_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("Failed to encode image.")
        return base64.b64encode(buffer).decode("utf-8")

    async def request_action_from_payload(self, websocket, payload: Dict[str, Any]) -> np.ndarray:
        await websocket.send(json.dumps(payload))
        response = await asyncio.wait_for(websocket.recv(), timeout=self.recv_timeout)
        data = json.loads(response)
        actions = np.asarray(data["actions"], dtype=np.float32)

        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions.squeeze(0)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        print("\n================ Online Predicted Actions ================")
        print(actions)
        print("xyz min:", actions[:, :3].min(axis=0))
        print("xyz max:", actions[:, :3].max(axis=0))
        print("first action:", actions[0])
        return actions


class FrankaActionExecutor:
    def __init__(
            self,
            arm,
            action_dt: float = 0.1,
            control_hz: int = 100,
            kp: float = 2.0,
            ki: float = 0.03,
            max_linear_vel: float = 0.08,
            robot_deadband: float = 0.001,
            integral_limit: float = 0.01,
            gripper_close_threshold: float = 0.5):
        self.arm = arm
        self.action_dt = action_dt
        self.control_hz = control_hz
        self.kp = kp
        self.ki = ki
        self.max_linear_vel = max_linear_vel
        self.robot_deadband = robot_deadband
        self.integral_limit = integral_limit
        self.gripper_close_threshold = gripper_close_threshold

        self._integral_error = np.zeros(3, dtype=np.float64)
        self._gripper_closed = False

    def set_gripper_state_from_width(self, width: Optional[float]):
        if width is None:
            return
        self._gripper_closed = width < 0.04

    def reset_integral(self):
        self._integral_error[:] = 0.0

    def apply_gripper_command(self, gripper_cmd: float):
        should_close = float(gripper_cmd) >= self.gripper_close_threshold
        if should_close and not self._gripper_closed:
            self.arm.close_gripper()
            self._gripper_closed = True
            rospy.loginfo("Cloud action gripper: close")
        elif (not should_close) and self._gripper_closed:
            self.arm.open_gripper()
            self._gripper_closed = False
            rospy.loginfo("Cloud action gripper: open")

    def compute_velocity_command(self, target_pos: np.ndarray):
        current_pos, _ = self.arm.get_cartesian_pose()
        if current_pos is None:
            self.reset_integral()
            return np.zeros(3, dtype=np.float64), None

        error = target_pos - current_pos
        distance = np.linalg.norm(error)
        if distance < self.robot_deadband:
            self.reset_integral()
            return np.zeros(3, dtype=np.float64), distance

        dt = 1.0 / float(self.control_hz)
        self._integral_error += error * dt
        self._integral_error = np.clip(
            self._integral_error,
            -self.integral_limit,
            self.integral_limit
        )

        linear_cmd = self.kp * error + self.ki * self._integral_error
        norm = np.linalg.norm(linear_cmd)
        if norm > self.max_linear_vel:
            linear_cmd = linear_cmd / norm * self.max_linear_vel
        return linear_cmd, distance

    def execute_single_action(self, action: np.ndarray, running_check=None, active_check=None):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < 4:
            raise ValueError(f"Expected action dim >= 4, got {action.shape[0]}")

        target_pos = action[:3].astype(np.float64)
        self.apply_gripper_command(action[3])
        self.reset_integral()

        rate = rospy.Rate(self.control_hz)
        end_time = time.time() + self.action_dt

        while not rospy.is_shutdown():
            if running_check is not None and not running_check():
                break
            if active_check is not None and not active_check():
                break

            linear_cmd, distance = self.compute_velocity_command(target_pos)
            self.arm.set_cartesian_twist(
                linear=linear_cmd.tolist(),
                angular=[0.0, 0.0, 0.0]
            )

            if distance is not None and distance < self.robot_deadband:
                break
            if time.time() >= end_time:
                break
            rate.sleep()

        self.arm.stop_motion()


class CloudFrankaInferenceClient:
    def __init__(
            self,
            server_uri,
            arm=None,
            camera=None,
            init_hardware=True,
            execute_steps=8,
            action_dt=0.1,
            control_hz=100,
            jpeg_quality=90,
            recv_timeout=5.0,
            kp=2.0,
            ki=0.03,
            max_linear_vel=0.08,
            robot_deadband=0.001,
            integral_limit=0.01,
            gripper_close_threshold=0.5,
            save_frames=False,
            frame_dir="frames",
            prefetch_threshold=3,
            blend_steps=2,
            max_pos_step=0.005):
        self.server_uri = server_uri
        self.arm = arm
        self.camera = camera
        if init_hardware:
            if self.arm is None:
                self.arm = FrankaCartesianVelocityController()
            if self.camera is None:
                self.camera = RealSenseCamera()

        self.execute_steps = execute_steps
        self.save_frames = save_frames
        self.frame_dir = frame_dir
        if self.save_frames:
            os.makedirs(self.frame_dir, exist_ok=True)

        self.transport = CloudInferenceTransport(
            server_uri=server_uri,
            jpeg_quality=jpeg_quality,
            recv_timeout=recv_timeout
        )
        self.keyboard = KeyboardController()
        self.executor = FrankaActionExecutor(
            arm=self.arm,
            action_dt=action_dt,
            control_hz=control_hz,
            kp=kp,
            ki=ki,
            max_linear_vel=max_linear_vel,
            robot_deadband=robot_deadband,
            integral_limit=integral_limit,
            gripper_close_threshold=gripper_close_threshold
        )

        self.prefetch_threshold = prefetch_threshold
        self.blend_steps = blend_steps
        self.max_pos_step = max_pos_step

        self._running = False
        self._active = False
        self._step = 0
        self._need_reset_history = True

        self._active_plan = None
        self._active_plan_idx = 0
        self._pending_plan = None
        self._request_in_flight = False
        self._last_executed_action = None

    def _is_running(self):
        return self._running

    def _is_active(self):
        return self._active

    def handle_keyboard(self):
        key = self.keyboard.read_key()
        if key == "s":
            if not self._active:
                self._need_reset_history = True
            self._active = True
            print("[INFO] Inference started.")
        elif key == "e":
            self._active = False
            self._need_reset_history = True
            if self.arm is not None:
                self.arm.stop_motion()
            print("[INFO] Inference paused.")
        elif key == "q":
            self._running = False
            self._active = False
            self._need_reset_history = True
            if self.arm is not None:
                self.arm.stop_motion()
            print("[INFO] Quit requested.")

    def get_observation(self) -> Optional[Dict[str, Any]]:
        if self.camera is None or self.arm is None:
            raise RuntimeError("Hardware is not initialized.")

        color_bgr, _, _ = self.camera.get_frames()
        if color_bgr is None:
            return None

        eef_pos, eef_quat = self.arm.get_cartesian_pose()
        gripper_width = self.arm.get_gripper_width()
        if eef_pos is None or eef_quat is None or gripper_width is None:
            return None

        if self.save_frames:
            cv2.imwrite(
                os.path.join(self.frame_dir, f"frame_{self._step:06d}.jpg"),
                color_bgr
            )

        return {
            "image_bgr": color_bgr,
            "robot0_eef_pos": np.asarray(eef_pos, dtype=np.float32),
            "robot0_eef_quat": np.asarray(eef_quat, dtype=np.float32),
            "robot0_gripper_qpos": np.asarray([gripper_width], dtype=np.float32),
            "step": self._step,
            "timestamp": time.time(),
        }

    def build_payload(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "image_b64": self.transport.encode_image(obs["image_bgr"]),
            "robot0_eef_pos": obs["robot0_eef_pos"].tolist(),
            "robot0_eef_quat": obs["robot0_eef_quat"].tolist(),
            "robot0_gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
            "step": int(obs["step"]),
            "timestamp": float(obs["timestamp"]),
            "reset_history": self._need_reset_history,
        }

    async def request_action_once(self, websocket):
        obs = self.get_observation()
        if obs is None:
            return None

        payload = self.build_payload(obs)
        actions = await self.transport.request_action_from_payload(websocket, payload)
        self._need_reset_history = False
        return actions

    def build_payload_from_dataset_frame(self, demo, step_idx, reset_history=False):
        image_rgb = np.asarray(demo["obs"]["camera_0"][step_idx], dtype=np.uint8)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return {
            "image_b64": self.transport.encode_image(image_bgr),
            "robot0_eef_pos": np.asarray(demo["obs"]["robot0_eef_pos"][step_idx], dtype=np.float32).tolist(),
            "robot0_eef_quat": np.asarray(demo["obs"]["robot0_eef_quat"][step_idx], dtype=np.float32).tolist(),
            "robot0_gripper_qpos": np.asarray(demo["obs"]["robot0_gripper_qpos"][step_idx], dtype=np.float32).tolist(),
            "step": int(step_idx),
            "timestamp": float(step_idx),
            "reset_history": bool(reset_history),
        }

    async def evaluate_demo_action_error(self, dataset_path, demo_key="demo_0", start_idx=1, n_obs_steps=2):
        with h5py.File(dataset_path, "r") as root:
            demo = root["data"][demo_key]
            actions = np.asarray(demo["actions"], dtype=np.float32)
            total_steps = len(actions)
            if total_steps <= 0:
                raise RuntimeError(f"{demo_key} has no actions.")

            start_idx = int(start_idx)
            if start_idx < 0 or start_idx >= total_steps:
                raise IndexError(f"start_idx {start_idx} out of range for {demo_key} with {total_steps} steps.")

            hist_start = max(0, start_idx - n_obs_steps + 1)

            async with websockets.connect(self.server_uri, max_size=None) as websocket:
                pred_actions = None
                first_payload = True
                for t in range(hist_start, start_idx + 1):
                    payload = self.build_payload_from_dataset_frame(
                        demo=demo,
                        step_idx=t,
                        reset_history=first_payload,
                    )
                    first_payload = False
                    pred_actions = await self.transport.request_action_from_payload(websocket, payload)

            if pred_actions is None:
                raise RuntimeError("No predicted actions received from server.")

            horizon = len(pred_actions)
            gt_actions = actions[start_idx:start_idx + horizon]
            compare_len = min(len(pred_actions), len(gt_actions))
            if compare_len <= 0:
                raise RuntimeError("No overlapping action steps for comparison.")

            pred_actions = np.asarray(pred_actions[:compare_len], dtype=np.float32)
            gt_actions = np.asarray(gt_actions[:compare_len], dtype=np.float32)

            print("\n================ Predicted Actions ================")
            print(pred_actions)

            print("\n================ Ground Truth Actions ================")
            print(gt_actions)

            print("\n================ XYZ Range Comparison ================")
            print("pred xyz min:", pred_actions[:, :3].min(axis=0))
            print("pred xyz max:", pred_actions[:, :3].max(axis=0))
            print("gt   xyz min:", gt_actions[:, :3].min(axis=0))
            print("gt   xyz max:", gt_actions[:, :3].max(axis=0))

            print("\n================ First-Step Comparison ================")
            print("pred first:", pred_actions[0])
            print("gt   first:", gt_actions[0])

            mae = float(np.mean(np.abs(pred_actions - gt_actions)))
            mse = float(np.mean((pred_actions - gt_actions) ** 2))

            print("\n================ Error Metrics ================")
            print("mean_abs_error:", mae)
            print("mse:", mse)

            return {
                "demo_key": demo_key,
                "start_idx": start_idx,
                "pred_len": int(len(pred_actions)),
                "gt_len": int(len(gt_actions)),
                "mean_abs_error": mae,
                "mse": mse,
            }

    def execute_action_sequence(self, actions: np.ndarray):
        num_to_execute = min(self.execute_steps, len(actions))
        print(
            f"[Step {self._step}] received actions shape={actions.shape}, "
            f"executing first {num_to_execute} steps."
        )

        for idx in range(num_to_execute):
            if not self._running or not self._active:
                break
            self.executor.execute_single_action(
                actions[idx],
                running_check=self._is_running,
                active_check=self._is_active
            )
            self._last_executed_action = np.asarray(actions[idx], dtype=np.float32).copy()

    def _smooth_new_plan(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32).copy()
        if self._last_executed_action is None or len(actions) == 0:
            return actions

        prev = self._last_executed_action.copy()
        n = min(self.blend_steps, len(actions))

        for i in range(n):
            alpha = float(i + 1) / float(n + 1)
            blended_pos = (1.0 - alpha) * prev[:3] + alpha * actions[i, :3]

            delta = blended_pos - prev[:3]
            norm = np.linalg.norm(delta)
            if norm > self.max_pos_step:
                blended_pos = prev[:3] + delta / norm * self.max_pos_step

            actions[i, :3] = blended_pos
            prev[:3] = blended_pos

        return actions

    def initialize(self, timeout=3.0):
        if self.arm is None or self.camera is None:
            raise RuntimeError("Hardware is not initialized for online run().")

        start_time = time.time()
        while time.time() - start_time < timeout:
            pos, _ = self.arm.get_cartesian_pose()
            width = self.arm.get_gripper_width()
            if pos is not None and width is not None:
                self.executor.set_gripper_state_from_width(width)
                rospy.loginfo("CloudFrankaInferenceClient ready.")
                return True
            rospy.sleep(0.01)

        rospy.logerr("Franka state not ready.")
        return False

    async def communication_loop(self, websocket):
        while not rospy.is_shutdown() and self._running:
            self.handle_keyboard()

            if not self._active:
                await asyncio.sleep(0.02)
                continue

            need_request = (
                self._active_plan is None or
                (
                    self._active_plan is not None and
                    len(self._active_plan) - self._active_plan_idx <= self.prefetch_threshold
                )
            )

            if self._request_in_flight or not need_request:
                await asyncio.sleep(0.01)
                continue

            try:
                self._request_in_flight = True
                actions = await self.request_action_once(websocket)
                if actions is not None:
                    self._pending_plan = self._smooth_new_plan(actions)
            except asyncio.TimeoutError:
                print(f"[WARN] Step {self._step}: server response timeout.")
            except Exception as exc:
                print(f"[ERROR] async communication: {exc}")
                await asyncio.sleep(0.1)
            finally:
                self._request_in_flight = False

            await asyncio.sleep(0.001)

    async def execution_loop(self):
        while not rospy.is_shutdown() and self._running:
            self.handle_keyboard()

            if not self._active:
                await asyncio.sleep(0.02)
                continue

            if self._active_plan is None:
                if self._pending_plan is not None:
                    self._active_plan = self._pending_plan
                    self._pending_plan = None
                    self._active_plan_idx = 0
                    print(f"[Step {self._step}] activated new plan shape={self._active_plan.shape}")
                else:
                    await asyncio.sleep(0.005)
                    continue

            if self._active_plan_idx >= len(self._active_plan):
                if self._pending_plan is not None:
                    self._active_plan = self._pending_plan
                    self._pending_plan = None
                    self._active_plan_idx = 0
                    print(f"[Step {self._step}] switched to pending plan shape={self._active_plan.shape}")
                else:
                    self._active_plan = None
                    await asyncio.sleep(0.005)
                    continue

            action = self._active_plan[self._active_plan_idx]

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                self.executor.execute_single_action,
                action,
                self._is_running,
                self._is_active
            )

            self._last_executed_action = np.asarray(action, dtype=np.float32).copy()
            self._active_plan_idx += 1
            self._step += 1


    async def run(self):
        if not self.initialize():
            return

        self.keyboard.setup()
        self._running = True

        print(f"[INFO] Connecting to {self.server_uri}")
        print("[INFO] Keys: s=start, e=stop, q=quit")

        try:
            async with websockets.connect(self.server_uri, max_size=None) as websocket:
                while not rospy.is_shutdown() and self._running:
                    self.handle_keyboard()
                    if not self._active:
                        await asyncio.sleep(0.02)
                        continue

                    try:
                        actions = await self.request_action_once(websocket)
                        if actions is None:
                            print(f"[WARN] Step {self._step}: observation not ready, skip.")
                            await asyncio.sleep(self.executor.action_dt)
                            continue

                        current_pos, _ = self.arm.get_cartesian_pose()
                        if current_pos is not None:
                            print("current_pos:", np.asarray(current_pos, dtype=np.float32))

                        print(f"[Step {self._step}] received actions shape={actions.shape}")
                        self.execute_action_sequence(actions)
                        self._step += 1

                    except asyncio.TimeoutError:
                        print(f"[WARN] Step {self._step}: server response timeout.")
                    except Exception as exc:
                        print(f"[ERROR] Step {self._step}: {exc}")
                        await asyncio.sleep(0.1)
        finally:
            if self.arm is not None:
                self.arm.stop_motion()
            self.keyboard.restore()
            if self.camera is not None:
                self.camera.stop()

    async def run_async(self):
        if not self.initialize():
            return

        self.keyboard.setup()
        self._running = True

        print(f"[INFO] Connecting to {self.server_uri}")
        print("[INFO] Keys: s=start, e=stop, q=quit")

        try:
            async with websockets.connect(self.server_uri, max_size=None) as websocket:
                await asyncio.gather(
                    self.communication_loop(websocket),
                    self.execution_loop(),
                )
        finally:
            if self.arm is not None:
                self.arm.stop_motion()
            self.keyboard.restore()
            if self.camera is not None:
                self.camera.stop()


if __name__ == "__main__":
    # Online robot mode, synchronous baseline:
    client = CloudFrankaInferenceClient(
        server_uri="ws://127.0.0.1:6006/ws",
        execute_steps=12,
        action_dt=0.1,
        control_hz=100,
        jpeg_quality=90,
        recv_timeout=5.0,
        kp=2.0,
        ki=0.03,
        max_linear_vel=0.08,
        robot_deadband=0.001,
        integral_limit=0.01,
        gripper_close_threshold=0.5,
        save_frames=False,
    )
    asyncio.run(client.run())

    # Online robot mode, asynchronous communication + execution:
    # client = CloudFrankaInferenceClient(
    #     server_uri="ws://127.0.0.1:6006/ws",
    #     execute_steps=12,
    #     action_dt=0.1,
    #     control_hz=100,
    #     jpeg_quality=90,
    #     recv_timeout=5.0,
    #     kp=2.0,
    #     ki=0.03,
    #     max_linear_vel=0.08,
    #     robot_deadband=0.001,
    #     integral_limit=0.01,
    #     gripper_close_threshold=0.5,
    #     save_frames=False,
    #     prefetch_threshold=3,
    #     blend_steps=2,
    #     max_pos_step=0.005,
    # )
    # asyncio.run(client.run_async())

    # Offline dataset evaluation:
    # client = CloudFrankaInferenceClient(
    #     server_uri="ws://127.0.0.1:6006/ws",
    #     init_hardware=False,
    # )
    # result = asyncio.run(
    #     client.evaluate_demo_action_error(
    #         dataset_path="/home/ssui/catkin1_ws/franka_ros_controller/for_diffusion_policy/data/franka_image/image.hdf5",
    #         demo_key="demo_0",
    #         start_idx=20,
    #         n_obs_steps=1,
    #     )
    # )
    # print(result)
