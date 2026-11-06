"""确认服务 wire DTO——镜像 rs/ulabel-service/src/http.rs 的 serde 形状。

wire 正典 = ulabel 仓 `rs/ulabel-service/tests/openapi.snapshot.json`
（OpenAPI 快照，唯一裁定源）；服务端改 wire 时本模块同一 commit 同步
（契约登记见 ulabel docs/CONTRACT.md「service client」）。

标注域类型，勿与检测域类型（rect 矩形、无属性表）混淆。字段名与服务端
serde 输出逐字对齐；读侧 extra="ignore"（宽容读惯例），请求体模型仅由
客户端显式构造。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from jxl.label.label import Label
from jxl.label.meta import LabelMeta


class _Wire(BaseModel):
    """服务端 JSON 读侧基型：未知字段忽略。"""

    model_config = ConfigDict(extra="ignore")


SetKind = Literal["verdict", "annotation", "session"]

# 集级媒体声明（set.json5 / 建集请求体必填，无默认——镜像 rs media_kind::SetMedia）
SetMedia = Literal["image", "audio_clip", "av_session"]


class ProjectInfo(_Wire):
    id: int
    name: str
    dir: str
    groups: int


class GroupInfo(_Wire):
    slug: str
    name: str
    sets: int


class AnnotatorInfo(_Wire):
    name: str
    display: str
    kind: str
    level: int
    display_color: str


class SetOut(_Wire):
    slug: str
    name: str
    samples: int
    progress: dict[str, int]
    media: SetMedia


class VerdictOut(_Wire):
    annotator: str
    value: int
    note: str | None = None
    ts: str


class TruthOut(_Wire):
    annotator: str
    value: int


class MediaImage(_Wire):
    """样本媒体负载·图像臂（serde tag=kind 的镜像）。"""

    kind: Literal["image"]
    url: str


class ClipSpan(_Wire):
    """源 session 时轴数对 {start, end}（i64 毫秒，字段名不带单位规则）。

    window 与 interval 同形（rs builder::clips::ClipSpan 镜像——同一概念
    同一 schema，跨 wire/磁盘/校验三处）。
    """

    start: int
    end: int


class MediaAudioClip(_Wire):
    """样本媒体负载·音频片段臂（rs http/samples.rs::SampleMedia 镜像）。

    start/end = 当前 interval（可编辑真值边界）；window = 不可变物化窗
    （wav 本地 0 = window.start 的锚）。全部 i64 毫秒（字段名不带单位规则）。
    """

    kind: Literal["audio_clip"]
    url: str
    start: int
    end: int
    window: ClipSpan


class ClipSeed(BaseModel):
    """POST samples 的 `?clip=` 载荷（audio_clip 集必填）——rs builder::clips::ClipSeed 镜像。

    上传时的物化事实：不可变 window + 初始 interval（同一 all-or-nothing
    步骤落为 clips/<stem>.json5 sidecar；interval ⊆ window 服务端按区间律校验）。
    严格 schema（请求件仅由客户端显式构造，不带 _Wire 宽容读基型）。
    """

    window: ClipSpan
    interval: ClipSpan


SampleMedia = MediaImage | MediaAudioClip


class VerdictSampleOut(_Wire):
    """verdict 集的 GET samples 条目。"""

    stem: str
    media: SampleMedia
    verdicts: list[VerdictOut]
    truth: TruthOut | None = None


class AnnotationTruthOut(_Wire):
    by: str
    label: Label


class AnnotationSampleOut(_Wire):
    """annotation 集的 GET samples 条目：每标注者整套 Label + 等级真值。"""

    stem: str
    media: SampleMedia
    annotations: dict[str, Label]
    truth: AnnotationTruthOut | None = None


class Fingerprint(_Wire):
    file: str
    sha256: str


class SetRelation(_Wire):
    parent: str
    relation: str
    stem_pattern: str
    upstream_fingerprint: Fingerprint | None = None


class Relations(_Wire):
    sets: dict[str, SetRelation]


class UpstreamRef(BaseModel):
    """POST create_set 的 upstream 载荷（builder::UpstreamRef 镜像）。"""

    parent: str
    relation: str
    stem_pattern: str
    fingerprint: Fingerprint | None = None


class VerdictPost(BaseModel):
    """POST verdicts/{stem} 的 verdict 集载荷（annotator/ts 由服务端补）。"""

    value: int
    note: str | None = None


class VerdictWritten(_Wire):
    """POST verdicts/{stem} verdict 臂响应——verdict::Verdict 镜像。

    服务端回显写入的裁决（含服务端所打 ts），无 `annotator` 字段——
    `annotator` 只存在于 GET samples 的 `VerdictOut`。annotation 臂回显
    整个 `Label`（`last_modified` 服务端重打），由 client 直接以 `Label`
    解析，不设额外 DTO。
    """

    value: int
    note: str | None = None
    ts: str


__all__ = [
    "AnnotationSampleOut",
    "AnnotationTruthOut",
    "AnnotatorInfo",
    "ClipSeed",
    "ClipSpan",
    "Fingerprint",
    "GroupInfo",
    "Label",
    "LabelMeta",
    "ProjectInfo",
    "Relations",
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
