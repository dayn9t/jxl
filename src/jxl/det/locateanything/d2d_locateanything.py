"""LocateAnything 目标检测器(Detector2D 客户端实现).

与 D2dYoloE 的进程内加载不同: 推理跑在独立 la-venv 的常驻服务
(script/la-serve.sh 启动, transformers==4.57.1 与主环境隔离), 本类是纯
httpx 客户端, 构造即探活(服务未启动 fail-fast, 不静默).

NVIDIA License 非商用(research/evaluation only) — 不得用于商用标注管线.
"""

import base64
from pathlib import Path

import cv2
import httpx
from jvi.geo.rectangle import Rect
from jvi.image.image_nda import ImageNda

from jxl.det.box_utils import xyxy_iou
from jxl.det.d2d import D2dObject, D2dOpt, D2dResult, Detector2D
from jxl.det.hardmine import Box
from jxl.det.locateanything.boxes import DEDUP_IOU_THR
from jxl.det.locateanything.client import LA_DEFAULT_URL, LaClient


class D2dLocateAnything(Detector2D):
    """基于 LocateAnything-3B 本地服务的开放词汇检测器."""

    model_class = "D2dLocateAnything"
    """模型类型标识"""

    def __init__(
        self,
        model_path: Path,
        opt: D2dOpt,
        names: list[str],
        device_name: str = "",
        verbose: bool = False,
        base_url: str = LA_DEFAULT_URL,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """初始化检测器.

        Args:
            model_path: 仅为 Detector2D 接口一致保留 — 权重由服务端持有, 客户端不加载.
            opt: 检测器选项(conf_thr 不生效: 模型无 confidence, conf 恒 1.0).
            names: 开放词汇类别名(逐类单请求, 规避官方多类 label corruption).
            device_name: 不生效(设备由服务端启动参数决定).
            verbose: 未使用(服务端日志).
            base_url: 推理服务地址.
            transport: http transport 注入(单测 MockTransport 用), 生产传 None.

        """
        super().__init__(model_path, opt, device_name, verbose)
        self._names = names
        self._client = LaClient(base_url=base_url, transport=transport)
        self._client.health()  # fail-fast: 服务未启动立即报错

    def detect(self, image: ImageNda) -> D2dResult:
        """检测图像中的目标(逐类单请求 + 跨类去重)."""
        b64 = self._encode_jpeg_b64(image)
        kept: list[tuple[int, Box]] = []  # (cls, box)
        for cls, name in enumerate(self._names):
            for box in self._client.detect_bytes(b64, name):
                if any(xyxy_iou(box[:4], kb[:4]) >= DEDUP_IOU_THR for _, kb in kept):
                    continue  # 同一目标被多个类名检出, 先到的类保留
                kept.append((cls, box))
        objects = [
            D2dObject(
                id=0, cls=cls, conf=box[4], rect=Rect.from_ltrb_list(list(box[:4]))
            )
            for cls, box in kept
        ]
        return D2dResult(objects=objects)

    @staticmethod
    def _encode_jpeg_b64(image: ImageNda) -> str:
        """ImageNda(BGR) → JPEG base64.

        cv2.imencode 期望 BGR 输入(JPEG 无通道序, 服务端 PIL 解码即得正确色),
        勿先转 RGB——那是 PIL Image.save 的约定, 混用会通道反转.
        """
        ok, buf = cv2.imencode(
            ".jpg", image.data(), [int(cv2.IMWRITE_JPEG_QUALITY), 92]
        )
        if not ok:
            msg = "JPEG 编码失败"
            raise RuntimeError(msg)
        return base64.b64encode(buf.tobytes()).decode()
