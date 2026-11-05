"""jxl.label.service——ulabel-service 确认服务 REST 客户端。

薄方法 + 显式异常；DTO 与 rs/ulabel-service 的 wire 正典（OpenAPI 快照）
同步修订（契约登记见 ulabel docs/CONTRACT.md「service client」）。
"""

from jxl.label.service.client import ConfirmClient, ServiceError
from jxl.label.service.dto import (
    AnnotationSampleOut,
    AnnotatorInfo,
    Fingerprint,
    GroupInfo,
    ProjectInfo,
    Relations,
    SetKind,
    SetMedia,
    SetOut,
    SetRelation,
    TruthOut,
    UpstreamRef,
    VerdictOut,
    VerdictPost,
    VerdictSampleOut,
    VerdictWritten,
)

__all__ = [
    "AnnotationSampleOut",
    "AnnotatorInfo",
    "ConfirmClient",
    "Fingerprint",
    "GroupInfo",
    "ProjectInfo",
    "Relations",
    "ServiceError",
    "SetKind",
    "SetMedia",
    "SetOut",
    "SetRelation",
    "TruthOut",
    "UpstreamRef",
    "VerdictOut",
    "VerdictPost",
    "VerdictSampleOut",
    "VerdictWritten",
]
