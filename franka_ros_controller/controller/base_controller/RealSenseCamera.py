#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import cv2
import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    #============ 初始化与配置 ================

    def __init__(self,
                 width=640,
                 height=480,
                 fps=30,
                 enable_color=True,
                 enable_depth=True,
                 align_depth_to_color=True,
                 warmup_frames=10):
        """
        初始化 RealSense 相机。

        参数:
            width: 图像宽度
            height: 图像高度
            fps: 采集帧率
            enable_color: 是否启用彩色流
            enable_depth: 是否启用深度流
            align_depth_to_color: 是否将深度图对齐到彩色图坐标系
            warmup_frames: 启动后预热帧数
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.align_depth_to_color = align_depth_to_color
        self.warmup_frames = warmup_frames

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.profile = None
        self.align = None
        self._running = False

        self._start_pipeline()

    #============ 设备生命周期管理 ================

    def _start_pipeline(self):
        """配置并启动 RealSense 数据流。"""
        if self.enable_color:
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.enable_depth:
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self.profile = self.pipeline.start(self.config)

        if self.enable_color and self.enable_depth and self.align_depth_to_color:
            self.align = rs.align(rs.stream.color)

        self._running = True

        for _ in range(self.warmup_frames):
            self.pipeline.wait_for_frames()

    def stop(self):
        """停止相机数据流并关闭窗口。"""
        if self._running:
            self.pipeline.stop()
            self._running = False
        cv2.destroyAllWindows()

    #============ 图像采集 ================

    def get_frames(self):
        """
        获取当前彩色图和深度图。

        返回:
            color_image: BGR 彩色图
            color_image_rgb: RGB 彩色图
            depth_image: 深度图
        """
        color_image = None
        color_image_rgb = None
        depth_image = None

        frames = self.pipeline.wait_for_frames()

        if self.align is not None:
            frames = self.align.process(frames)

        if self.enable_color:
            color_frame = frames.get_color_frame()
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, color_image_rgb, depth_image

    def get_color_frame(self):
        """仅获取当前 BGR 彩色图。"""
        color_image, _, _ = self.get_frames()
        return color_image

    def get_depth_frame(self):
        """仅获取当前深度图。"""
        _, _, depth_image = self.get_frames()
        return depth_image

    #============ 图像显示 ================

    def get_depth_colormap(self, depth_image, alpha=0.03):
        """将深度图转换为伪彩色图，便于显示。"""
        if depth_image is None:
            return None
        return cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=alpha), cv2.COLORMAP_JET)

    def show_frames(self, color_image=None, depth_image=None, alpha=0.03):
        """
        显示彩色图和深度图。

        参数:
            color_image: BGR 彩色图
            depth_image: 深度图
            alpha: 深度图显示缩放系数
        """
        if color_image is not None:
            cv2.imshow("Color", color_image)

        if depth_image is not None:
            depth_colormap = self.get_depth_colormap(depth_image, alpha=alpha)
            if depth_colormap is not None:
                cv2.imshow("Depth", depth_colormap)

        cv2.waitKey(1)

    #============ 测试函数 ================

    def test_stream(self):
        """持续显示相机画面，用于检查采集与显示是否正常。"""
        print("Start RealSense stream test, press Ctrl+C to stop.")
        try:
            while True:
                color_image, _, depth_image = self.get_frames()
                self.show_frames(color_image, depth_image)
        except KeyboardInterrupt:
            print("Stream test stopped.")

    def test_capture_interval(self, interval=0.5):
        """
        定时采集图像并打印计数，用于检查连续采集是否稳定。

        参数:
            interval: 采集时间间隔，单位秒
        """
        data = []
        print("Start RealSense capture interval test, press Ctrl+C to stop.")
        try:
            while True:
                color_image, _, depth_image = self.get_frames()
                self.show_frames(color_image, depth_image)
                data.append(color_image)
                print("Captured frames:", len(data))
                time.sleep(interval)
        except KeyboardInterrupt:
            print("Capture interval test stopped.")


if __name__ == "__main__":
    '''
    camera = RealSenseCamera()
    color_image, color_image_rgb, depth_image = camera.get_frames()
    print(color_image.shape if color_image is not None else None)
    print(depth_image.shape if depth_image is not None else None)
    camera.stop()

    '''
    camera = RealSenseCamera()
    try:
        #============ 测试函数写这里 ============

        camera.test_stream()
        # camera.test_capture_interval(interval=0.5)

    finally:
        camera.stop()
    
