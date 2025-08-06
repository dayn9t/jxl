#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import cv2
from loguru import logger
from ultralytics import NAS, YOLO


def detect_person_in_video(
    video_path: str,
    model_name: str = "yolo_nas_s",
    conf_thres: float = 0.25,
    output_path: Optional[str] = None,
    display: bool = False,
):
    """
    使用YOLO-NAS检测视频中的人员

    Args:
        video_path: 输入视频路径
        model_name: YOLO-NAS模型名称，可选 "yolo_nas_s", "yolo_nas_m", "yolo_nas_l"
        conf_thres: 置信度阈值
        output_path: 输出视频路径，如果为None则不保存
        display: 是否实时显示检测结果
    """
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        logger.error(f"视频文件不存在: {video_path}")
        return

    # 加载YOLO-NAS模型
    try:
        # 尝试加载NAS模型
        model = NAS(model_name)
        logger.info(f"已加载YOLO-NAS模型: {model_name}")
    except Exception as e:
        # 如果NAS模型加载失败，尝试加载普通YOLO模型
        logger.warning(f"加载YOLO-NAS模型失败，尝试加载普通YOLO模型: {e}")
        model = YOLO("yolov8m.pt")
        logger.info("已加载YOLOv8n模型")

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"视频信息: {width}x{height}, {fps}FPS, 共{frame_count}帧")

    # 设置输出视频
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"输出视频将保存到: {output_path}")

    frame_idx = 0

    # 处理视频帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 10 == 0:
            logger.info(
                f"正在处理第 {frame_idx}/{frame_count} 帧 ({frame_idx / frame_count * 100:.1f}%)"
            )

        # 使用YOLO-NAS进行检测
        results = model(frame, conf=conf_thres, classes=[0])  # 类别0通常是person

        # 获取检测到的人员数量
        person_count = 0
        for result in results:
            boxes = result.boxes
            person_count = len(boxes)

        # 在帧上绘制检测结果
        annotated_frame = results[0].plot()

        # 添加人员计数文本
        cv2.putText(
            annotated_frame,
            f"Persons: {person_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # 保存或显示帧
        if out:
            out.write(annotated_frame)

        if display:
            cv2.imshow("YOLO-NAS Person Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # 清理资源
    cap.release()
    if out:
        out.release()
    if display:
        cv2.destroyAllWindows()

    logger.info("视频处理完成")


def main():
    parser = argparse.ArgumentParser(description="使用YOLO-NAS检测视频中的人员")
    parser.add_argument("video", help="输入视频文件路径")
    parser.add_argument(
        "--model",
        default="yolo_nas_s",
        help="YOLO-NAS模型: yolo_nas_s, yolo_nas_m, yolo_nas_l",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--output", help="输出视频路径")
    parser.add_argument("--display", action="store_true", help="实时显示检测结果")

    args = parser.parse_args()

    detect_person_in_video(
        video_path=args.video,
        model_name=args.model,
        conf_thres=args.conf,
        output_path=args.output,
        display=args.display,
    )


if __name__ == "__main__":
    main()
