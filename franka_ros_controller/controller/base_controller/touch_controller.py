import time
import threading
from dataclasses import dataclass, field

import numpy as np
import pyOpenHaptics.hd as hd
from pyOpenHaptics.hd_callback import hd_callback
from pyOpenHaptics.hd_device import HapticDevice


#============ 数据结构 ================

@dataclass
class TouchState:
    """保存 Touch 设备当前一次采样得到的状态。"""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    button_down: bool = False
    button_up: bool = False
    timestamp: float = 0.0


# 当前激活的 TouchController 实例，供底层回调函数访问
_ACTIVE_CONTROLLER = None


#============ 底层回调 ================

@hd_callback
def _touch_state_callback():
    """底层设备回调函数，用于持续刷新 Touch 当前状态。"""
    global _ACTIVE_CONTROLLER
    controller = _ACTIVE_CONTROLLER
    if controller is None:
        return

    transform = hd.get_transform()
    button_state = hd.get_buttons()
    position = np.array([transform[3][0], transform[3][1], transform[3][2]], dtype=np.float64)

    with controller._lock:
        controller._state.position = position
        controller._state.button_down = bool(button_state & 1)
        controller._state.button_up = bool(button_state & 2)
        controller._state.timestamp = time.time()
        controller._has_state = True


class TouchController:
    #============ 初始化与配置 ================

    def __init__(self, device_name="Default Device", auto_start=True):
        """
        初始化 Touch 控制器。

        参数:
            device_name: Touch 设备名称，默认使用 "Default Device"
            auto_start: 是否在创建对象后立刻启动设备连接
        """
        self.device_name = device_name
        self._lock = threading.Lock()
        self._state = TouchState()
        self._has_state = False
        self._zero_position = np.zeros(3, dtype=np.float64)
        self._prev_button_down = False
        self._prev_button_up = False
        self._device = None

        if auto_start:
            self.start()

    #============ 设备生命周期管理 ================

    def start(self):
        """启动 Touch 设备，并注册状态回调。"""
        global _ACTIVE_CONTROLLER
        if self._device is not None:
            return
        _ACTIVE_CONTROLLER = self
        self._device = HapticDevice(device_name=self.device_name, callback=_touch_state_callback)

    def close(self):
        """关闭 Touch 设备连接，并清理当前控制器引用。"""
        global _ACTIVE_CONTROLLER
        if self._device is not None:
            self._device.close()
            self._device = None
        if _ACTIVE_CONTROLLER is self:
            _ACTIVE_CONTROLLER = None

    def wait_until_ready(self, timeout=2.0):
        """
        等待设备收到第一帧有效状态。

        参数:
            timeout: 最长等待时间，单位秒

        返回:
            True 表示设备已准备好，False 表示超时
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.has_state():
                return True
            time.sleep(0.01)
        return False

    def has_state(self):
        """判断当前是否已经收到过至少一帧设备状态。"""
        with self._lock:
            return self._has_state

    #============ 状态读取 ================

    def get_state(self):
        """获取当前 Touch 状态快照。"""
        with self._lock:
            return TouchState(
                position=self._state.position.copy(),
                button_down=self._state.button_down,
                button_up=self._state.button_up,
                timestamp=self._state.timestamp
            )

    def get_position(self):
        """获取当前 Touch 三维位置。"""
        return self.get_state().position

    def get_relative_position(self):
        """获取相对于零点位置的位移。"""
        return self.get_position() - self._zero_position

    def get_buttons(self):
        """
        获取当前两个按键状态。

        返回:
            (button_down, button_up)
        """
        state = self.get_state()
        return state.button_down, state.button_up

    #============ 零点与坐标映射 ================

    def zero(self):
        """
        将当前 Touch 位置设为零点。

        返回:
            当前记录下来的零点位置
        """
        self._zero_position = self.get_position().copy()
        return self._zero_position.copy()

    def get_mapped_delta(self):
        """
        获取映射到 Franka 坐标系后的相对位移。

        当前映射规则:
            touch.z -> -franka.x
            touch.x -> -franka.y
            touch.y ->  franka.z
        """
        delta = self.get_relative_position()
        return np.array([-delta[2], -delta[0], delta[1]], dtype=np.float64)

    #============ 按键事件处理 ================

    def get_button_edges(self):
        """
        获取按键边沿事件。

        返回:
            字典，包含 down/up 两个按键的按下沿和释放沿
        """
        state = self.get_state()
        down_pressed = state.button_down and not self._prev_button_down
        down_released = (not state.button_down) and self._prev_button_down
        up_pressed = state.button_up and not self._prev_button_up
        up_released = (not state.button_up) and self._prev_button_up
        self._prev_button_down = state.button_down
        self._prev_button_up = state.button_up
        return {
            "down_pressed": down_pressed,
            "down_released": down_released,
            "up_pressed": up_pressed,
            "up_released": up_released,
        }

    def reset_button_edges(self):
        """重置按键边沿检测的历史状态，避免初始化后误触发。"""
        state = self.get_state()
        self._prev_button_down = state.button_down
        self._prev_button_up = state.button_up
        
        
   
    #============ 测试函数 ================

    def test_print_state(self, hz=20):
        """
        持续打印 Touch 当前原始状态，用于检查设备是否正常连接。

        参数:
            hz: 打印频率
        """
        dt = 1.0 / float(hz)
        print("Start test_print_state, press Ctrl+C to stop.")
        while True:
            state = self.get_state()
            print(
                "pos = [{:.3f}, {:.3f}, {:.3f}] | down = {} | up = {} | t = {:.3f}".format(
                    state.position[0],
                    state.position[1],
                    state.position[2],
                    state.button_down,
                    state.button_up,
                    state.timestamp
                )
            )
            time.sleep(dt)

    def test_print_mapped_delta(self, hz=20):
        """
        先记录当前零点，再持续打印相对位移和 Franka 坐标映射结果。

        参数:
            hz: 打印频率
        """
        if not self.has_state():
            print("Touch state not ready.")
            return
        self.zero()
        self.reset_button_edges()
        dt = 1.0 / float(hz)
        print("Zero position recorded. Start test_print_mapped_delta, press Ctrl+C to stop.")
        while True:
            raw_pos = self.get_position()
            rel_pos = self.get_relative_position()
            mapped = self.get_mapped_delta()
            edges = self.get_button_edges()
            print(
                "raw = [{:.3f}, {:.3f}, {:.3f}] | rel = [{:.3f}, {:.3f}, {:.3f}] | mapped = [{:.3f}, {:.3f}, {:.3f}] | up_pressed = {} | down_pressed = {}".format(
                    raw_pos[0], raw_pos[1], raw_pos[2],
                    rel_pos[0], rel_pos[1], rel_pos[2],
                    mapped[0], mapped[1], mapped[2],
                    edges["up_pressed"], edges["down_pressed"]
                )
            )
            time.sleep(dt)



if __name__ == "__main__":
    touch = TouchController()
    try:
        if touch.wait_until_ready(timeout=2.0):

            # =============测试函数写这里=============

            # touch.test_print_state(hz=10)
            # 或者改成:
            touch.test_print_mapped_delta(hz=10)
        else:
            print("Touch device not ready.")
    finally:
        touch.close()
