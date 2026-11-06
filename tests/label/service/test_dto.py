"""service.dto 读侧镜像单测——字段名与服务端 serde 逐字对齐。"""

import json

from jxl.label.service.dto import (
    ClipSeed,
    ClipSpan,
    GroupInfo,
    ProjectInfo,
    Relations,
    SetOut,
    VerdictOut,
    VerdictSampleOut,
)


def test_project_info_parses_wire() -> None:
    payload = json.loads('{"id":1,"name":"n001","dir":"sgcc_n001","groups":2}')
    assert ProjectInfo.model_validate(payload) == ProjectInfo(
        id=1, name="n001", dir="sgcc_n001", groups=2
    )


def test_unknown_fields_ignored() -> None:
    out = GroupInfo.model_validate({"slug": "r01", "name": "首轮", "sets": 5, "future_field": 1})
    assert out == GroupInfo(slug="r01", name="首轮", sets=5)


def test_set_progress_and_verdict_note_none() -> None:
    out = SetOut.model_validate({"slug": "s", "name": "n", "samples": 7, "progress": {"jiang": 3}, "media": "image"})
    verdict = VerdictOut.model_validate(
        {"annotator": "jiang", "value": 2, "note": None, "ts": "2026-09-23T00:00:00Z"}
    )
    assert out.progress == {"jiang": 3}
    assert verdict.note is None


def test_relations_compound_keys() -> None:
    payload = {
        "sets": {
            "r01/role_confirm": {
                "parent": "gencheck/rolepool_20260919",
                "relation": "attribute:role",
                "stem_pattern": "<uid>_<track_id>",
                "upstream_fingerprint": {"file": "excluded.csv", "sha256": "ab12"},
            }
        }
    }
    rel = Relations.model_validate(payload)
    entry = rel.sets["r01/role_confirm"]
    assert entry.upstream_fingerprint is not None
    assert entry.upstream_fingerprint.sha256 == "ab12"


def test_sample_union_members() -> None:
    verdict_sample = VerdictSampleOut.model_validate(
        {
            "stem": "a_1",
            "media": {"kind": "image", "url": "/api/img?p=x"},
            "verdicts": [],
            "truth": None,
        }
    )
    assert verdict_sample.media.kind == "image"
    assert verdict_sample.media.url == "/api/img?p=x"

    clip = VerdictSampleOut.model_validate(
        {
            "stem": "s_2",
            "media": {
                "kind": "audio_clip",
                "url": "/api/img?p=w",
                "start": 3000,
                "end": 7500,
                "window": {"start": 1000, "end": 9000},
            },
            "verdicts": [],
            "truth": None,
        }
    )
    assert clip.media.kind == "audio_clip"
    assert clip.media.url == "/api/img?p=w"
    assert (clip.media.start, clip.media.end) == (3000, 7500)
    assert clip.media.window == ClipSpan(start=1000, end=9000), "物化窗镜像（wav 本地 0 锚）"


def test_clip_seed_roundtrip_shape() -> None:
    """POST samples `?clip=` 请求件：window + interval 同形数对（ClipSeed 镜像）。"""
    seed = ClipSeed.model_validate(
        {
            "window": {"start": 1000, "end": 9000},
            "interval": {"start": 3000, "end": 7500},
        }
    )
    assert seed.window == ClipSpan(start=1000, end=9000)
    assert seed.interval == ClipSpan(start=3000, end=7500)
    # 序列化为 query 值的紧凑 JSON（client.add_sample 以 model_dump_json 携带）
    assert '"window"' in seed.model_dump_json() and '"interval"' in seed.model_dump_json()


def test_session_kind_and_av_session_media_are_in_domain() -> None:
    """av_session 建集 wire（2026-11-06 plan）：SetKind/SetMedia 镜像必须含
    "session" / "av_session"——一旦服务端存在 session 集，SetOut.media 是
    pydantic Literal，缺值会让每次 list_sets 解析即 ValidationError。"""
    import typing

    from jxl.label.service.dto import SetKind, SetMedia

    assert set(typing.get_args(SetKind)) == {"verdict", "annotation", "session"}
    assert set(typing.get_args(SetMedia)) == {"image", "audio_clip", "av_session"}
    out = SetOut.model_validate(
        {"slug": "s", "name": "n", "samples": 3, "progress": {}, "media": "av_session"}
    )
    assert out.media == "av_session"
