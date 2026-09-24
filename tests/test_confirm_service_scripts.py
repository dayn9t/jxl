"""Regression tests: the confirm-service consumer scripts (projects/sgcc/bin).

Review findings covered (2026-09-24, P2 round 1): the nine audit fixes
(commits e7dcb89 + dd31b27) shipped with no test file, and the core guards
are behaviorally invisible today — the six `except httpx.TransportError:
raise` clauses are no-ops until someone merges them back into the
ServiceError handler. Every fix gets a pinning test:

  1. TransportError injected into create/add/verdict reaches the FATAL
     handler (exit 2); the merged-except regression would crash with
     AttributeError (no `.status` on a transport error) and fail here.
  2. ServiceError 409 = "exists, continue"; 500 re-raises (exit 2, aborted
     before any upload).
  3. build_confirm_samples: missing `--as` and `--limit 0` both exit 2 at
     the boundary (identity never defaulted; batch size at least 1).
  4. build_confirm_samples `--apply` refuses fingerprint-less sets, having
     spoken nothing beyond login + relations.
  5. role_confirm_backflow `--apply`: a failing rewrite maps OSError to
     exit 2 and leaves the target file byte-identical.
  6. init_confirm_project: unreadable fingerprint source is fatal;
     build_role_confirm_set: missing/undecodable excluded file is fatal.

The service is faked behind httpx.MockTransport — the transport seam
ConfirmClient.login documents for tests — so the real client (paths, Bearer
auth, response parsing, ServiceError raising) is exercised, not a
hand-rolled double. The scripts build their own clients, so the mock is
pinned at login via a ConfirmClient subclass and the script module's
imported name is patched.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from vlabel.service import ConfirmClient
from vlabel.service.client import DEFAULT_TIMEOUT

BIN_DIR = Path(__file__).resolve().parents[1] / "projects/sgcc/bin"
SERVICE = "http://confirm.test"
TOKEN = "test-token"
TS = "2026-09-24T10:00:00+08:00"
Respond = Callable[[], httpx.Response]

if str(BIN_DIR) not in sys.path:
    sys.path.insert(0, str(BIN_DIR))

# The bin scripts use flat cross-imports (no package); load them by module
# name with BIN_DIR on sys.path. Module-typed access keeps mypy out of the
# scripts themselves (they sit outside the checked tree, like src/jxl/bin).
sgcc_metas = importlib.import_module("sgcc_confirm_metas")
samples_script = importlib.import_module("build_confirm_samples")
role_script = importlib.import_module("build_role_confirm_set")
init_script = importlib.import_module("init_confirm_project")
backflow_script = importlib.import_module("role_confirm_backflow")

# Script-owned contract constants (single source — the scripts define them).
PROJECT: str = role_script.PROJECT_SLUG
GROUP: str = role_script.GROUP_SLUG
SET: str = role_script.SET_SLUG
EXIT_FATAL: int = role_script.EXIT_FATAL
EXIT_OK: int = role_script.EXIT_OK
EXIT_REFUSED: int = backflow_script.EXIT_REFUSED

LOGIN = "/api/login"
PROJECTS = "/api/projects"
GROUPS = f"/api/projects/{PROJECT}/groups"
SETS = f"{GROUPS}/{GROUP}/sets"
RELATIONS = f"/api/projects/{PROJECT}/relations"
ROLE_SAMPLES = f"{SETS}/{SET}/samples"
ROLE_VERDICTS = f"{SETS}/{SET}/verdicts/*"


def samples_path(set_slug: str) -> str:
    return f"{SETS}/{set_slug}/samples"


def meta_path(set_slug: str) -> str:
    return f"{SETS}/{set_slug}/meta"


# ---- mock service -----------------------------------------------------------


class MockConfirmService:
    """(method, path)-routed confirm-service fake behind httpx.MockTransport.

    Routes are exact matches; templates ending in `*` match by prefix (the
    per-stem verdict posts). Unregistered requests answer 500 with the
    unexpected path — a loud default, so a test only passes when the flow it
    pins stays on its registered rails. Route handlers are zero-arg
    callables so scenarios can inject httpx.ConnectError alongside
    responses.
    """

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Respond] = {}
        self.requests: list[tuple[str, str]] = []

    def on(self, method: str, path: str, respond: Respond) -> MockConfirmService:
        self.routes[(method, path)] = respond
        return self

    def handle(self, request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        self.requests.append(key)
        respond = self.routes.get(key) or self._prefix_match(*key)
        if respond is None:
            return httpx.Response(500, json={"error": f"unexpected request {key[0]} {key[1]}"})
        return respond()

    def _prefix_match(self, method: str, path: str) -> Respond | None:
        for route_method, template in self.routes:
            if route_method == method and template.endswith("*") and path.startswith(template[:-1]):
                return self.routes[(route_method, template)]
        return None


def pinned_confirm_client(service: MockConfirmService) -> type[ConfirmClient]:
    """ConfirmClient whose login() always rides the mock transport."""
    mock_transport = httpx.MockTransport(service.handle)

    class Pinned(ConfirmClient):
        @classmethod
        def login(
            cls,
            base_url: str,
            name: str,
            passphrase: str | None = None,
            *,
            timeout: float = DEFAULT_TIMEOUT,
            transport: httpx.BaseTransport | None = None,
        ) -> ConfirmClient:
            return ConfirmClient.login(
                base_url,
                name,
                passphrase,
                timeout=timeout,
                transport=mock_transport if transport is None else transport,
            )

    return Pinned


# ---- response builders -------------------------------------------------------


def json_ok(data: object, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=data)


def login_ok() -> httpx.Response:
    return json_ok({"token": TOKEN})


def connect_refused() -> httpx.Response:
    raise httpx.ConnectError("connection refused")


def conflict() -> httpx.Response:
    return json_ok({"error": "already exists"}, status=409)


def created() -> httpx.Response:
    return httpx.Response(201)


def verdict_written() -> httpx.Response:
    return json_ok({"value": 1, "ts": TS})


def relation(
    parent: str, rel: str, fingerprint: dict[str, str] | None = None
) -> dict[str, object]:
    """One relations.sets entry (upstream_fingerprint absent = broken set)."""
    payload: dict[str, object] = {
        "parent": parent,
        "relation": rel,
        "stem_pattern": "<uid>_<track_id>",
    }
    if fingerprint is not None:
        payload["upstream_fingerprint"] = fingerprint
    return payload


def project_row() -> dict[str, object]:
    return {"id": 1, "name": "sgcc-n001", "dir": PROJECT, "groups": 1}


def group_row() -> dict[str, object]:
    return {"slug": GROUP, "name": "2026-09 首轮", "sets": 1}


def set_row() -> dict[str, object]:
    return {"slug": SET, "name": SET, "samples": 0, "progress": {}}


def register_all_present(service: MockConfirmService) -> None:
    """Login + project/group/set all listed: every create POST is skipped."""
    service.on("POST", LOGIN, login_ok)
    service.on("GET", PROJECTS, lambda: json_ok([project_row()]))
    service.on("GET", GROUPS, lambda: json_ok([group_row()]))
    service.on("GET", SETS, lambda: json_ok([set_row()]))


# ---- fixtures / runners ------------------------------------------------------

ROLE_ROW: dict[str, object] = {
    "uid": "u001",
    "track_id": 7,
    "exclude_reason": "conf low",
    "step5_roles": {"customer": 0.9},
    "step5_reason": "customer speaks",
    "cache": {"role_label": "customer", "role_conf": 0.97},
}


@pytest.fixture()
def excluded_world(tmp_path: Path) -> Path:
    """rolepool_excluded.jsonl (one row) + its materialized crop image."""
    excluded = tmp_path / "rolepool_excluded.jsonl"
    excluded.write_text(json.dumps(ROLE_ROW) + "\n", encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    (images / "u001.jpg").write_bytes(b"\xff\xd8fake-jpeg")
    return excluded


def run_role_builder(
    monkeypatch: pytest.MonkeyPatch, service: MockConfirmService, excluded: Path, mode: str
) -> int:
    monkeypatch.setattr(role_script, "ConfirmClient", pinned_confirm_client(service))
    # role builder's main(argv) feeds argv straight to parse_args — no prog name
    rc: int = role_script.main(["--service", SERVICE, "--excluded", str(excluded), mode])
    return rc


def run_init(monkeypatch: pytest.MonkeyPatch, service: MockConfirmService) -> int:
    monkeypatch.setattr(init_script, "ConfirmClient", pinned_confirm_client(service))
    monkeypatch.setattr(
        sys, "argv", ["init_confirm_project.py", "--service", SERVICE, "--as", "jiang"]
    )
    rc: int = init_script.main()
    return rc


def run_samples_import(
    monkeypatch: pytest.MonkeyPatch, service: MockConfirmService, *flags: str
) -> int:
    monkeypatch.setattr(samples_script, "ConfirmClient", pinned_confirm_client(service))
    monkeypatch.setattr(sys, "argv", ["build_confirm_samples.py", "--service", SERVICE, *flags])
    rc: int = samples_script.main()
    return rc


def run_backflow(
    monkeypatch: pytest.MonkeyPatch, service: MockConfirmService, excluded: Path, *flags: str
) -> int:
    monkeypatch.setattr(backflow_script, "ConfirmClient", pinned_confirm_client(service))
    monkeypatch.setattr(
        sys,
        "argv",
        ["role_confirm_backflow.py", "--service", SERVICE, "--excluded", str(excluded), *flags],
    )
    rc: int = backflow_script.main()
    return rc


def parse_samples_args(monkeypatch: pytest.MonkeyPatch, *flags: str) -> int:
    """build_confirm_samples.main reads sys.argv; argparse errors exit 2."""
    monkeypatch.setattr(sys, "argv", ["build_confirm_samples.py", *flags])
    with pytest.raises(SystemExit) as excinfo:
        samples_script.main()
    code = excinfo.value.code
    assert isinstance(code, int)
    return code


def patch_fingerprint_sources(monkeypatch: pytest.MonkeyPatch, root: Path, *, missing: bool) -> None:
    """Point init's FINGERPRINT_SOURCES at tmp files (or at missing paths)."""
    sources: dict[str, Path] = {}
    for slug, _meta_builder, _upstream in init_script.FUTURE_SETS:
        source = root / f"{slug}.jsonl"
        if not missing:
            source.write_text("upstream rows\n", encoding="utf-8")
        sources[str(slug)] = source
    monkeypatch.setattr(init_script, "FINGERPRINT_SOURCES", sources)


class ReplaceFailingOs:
    """os stand-in whose replace() always fails (simulated rewrite OSError)."""

    @staticmethod
    def replace(src: Path, dst: Path) -> None:
        raise OSError(f"simulated rewrite failure: {src} -> {dst}")


# ---- 1. TransportError propagates to the FATAL handler (exit 2) --------------


def test_role_builder_transport_error_on_create_exits_2(
    monkeypatch: pytest.MonkeyPatch, excluded_world: Path
) -> None:
    service = MockConfirmService()
    service.on("POST", LOGIN, login_ok)
    service.on("GET", PROJECTS, lambda: json_ok([]))
    service.on("POST", PROJECTS, connect_refused)
    rc = run_role_builder(monkeypatch, service, excluded_world, "--apply")
    assert rc == EXIT_FATAL
    # aborted at the first create POST — a merged-except regression would
    # die on AttributeError (no .status on TransportError) before returning
    assert ("POST", ROLE_SAMPLES) not in service.requests


def test_role_builder_transport_error_on_add_exits_2(
    monkeypatch: pytest.MonkeyPatch, excluded_world: Path
) -> None:
    service = MockConfirmService()
    register_all_present(service)
    service.on("POST", ROLE_SAMPLES, connect_refused)
    rc = run_role_builder(monkeypatch, service, excluded_world, "--apply")
    assert rc == EXIT_FATAL


def test_role_builder_transport_error_on_verdict_exits_2(
    monkeypatch: pytest.MonkeyPatch, excluded_world: Path
) -> None:
    service = MockConfirmService()
    register_all_present(service)
    service.on("POST", ROLE_SAMPLES, created)
    service.on("POST", ROLE_VERDICTS, connect_refused)
    rc = run_role_builder(monkeypatch, service, excluded_world, "--apply")
    assert rc == EXIT_FATAL


def test_init_transport_error_on_create_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    service = MockConfirmService()
    service.on("POST", LOGIN, login_ok)
    service.on("GET", PROJECTS, lambda: json_ok([]))
    service.on("POST", PROJECTS, connect_refused)
    rc = run_init(monkeypatch, service)
    assert rc == EXIT_FATAL


# ---- 2. ServiceError: 409 tolerated, 500 re-raised ----------------------------


def test_role_builder_409_counts_as_exists(
    monkeypatch: pytest.MonkeyPatch, excluded_world: Path
) -> None:
    service = MockConfirmService()
    service.on("POST", LOGIN, login_ok)
    service.on("GET", PROJECTS, lambda: json_ok([]))
    service.on("POST", PROJECTS, conflict)
    service.on("GET", GROUPS, lambda: json_ok([]))
    service.on("POST", GROUPS, conflict)
    service.on("GET", SETS, lambda: json_ok([]))
    service.on("POST", SETS, conflict)
    service.on("POST", ROLE_SAMPLES, created)
    service.on("POST", ROLE_VERDICTS, verdict_written)
    rc = run_role_builder(monkeypatch, service, excluded_world, "--apply")
    assert rc == EXIT_OK
    # each create POST was attempted and its 409 absorbed as "exists"
    assert service.requests.count(("POST", PROJECTS)) == 1
    assert service.requests.count(("POST", SETS)) == 1


def test_role_builder_500_reraises_and_aborts(
    monkeypatch: pytest.MonkeyPatch, excluded_world: Path
) -> None:
    service = MockConfirmService()
    service.on("POST", LOGIN, login_ok)
    service.on("GET", PROJECTS, lambda: json_ok([]))
    service.on("POST", PROJECTS, lambda: json_ok({"error": "boom"}, status=500))
    rc = run_role_builder(monkeypatch, service, excluded_world, "--apply")
    assert rc == EXIT_FATAL
    # re-raised: the flow aborted at create instead of plowing on
    assert ("POST", ROLE_SAMPLES) not in service.requests


def test_init_409_counts_as_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    service = MockConfirmService()
    service.on("POST", LOGIN, login_ok)
    service.on("GET", PROJECTS, lambda: json_ok([]))
    service.on("POST", PROJECTS, conflict)
    service.on("GET", GROUPS, lambda: json_ok([]))
    service.on("POST", GROUPS, conflict)
    service.on("GET", SETS, lambda: json_ok([]))
    service.on("POST", SETS, conflict)
    patch_fingerprint_sources(monkeypatch, tmp_path, missing=False)
    rc = run_init(monkeypatch, service)
    assert rc == EXIT_OK
    assert service.requests.count(("POST", SETS)) == len(init_script.FUTURE_SETS)


# ---- 3. build_confirm_samples boundary args -----------------------------------


def test_samples_import_requires_explicit_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    assert parse_samples_args(monkeypatch, "--service", SERVICE) == EXIT_FATAL


def test_samples_import_rejects_nonpositive_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    flags = ("--as", "jiang", "--limit", "0")
    assert parse_samples_args(monkeypatch, "--service", SERVICE, *flags) == EXIT_FATAL


# ---- 4. fingerprint gate refuses --apply --------------------------------------


def test_samples_import_refuses_apply_without_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MockConfirmService()
    service.on("POST", LOGIN, login_ok)
    service.on(
        "GET",
        RELATIONS,
        lambda: json_ok(
            {"sets": {f"{GROUP}/bad_fit_review": relation("pospool_20260919", "attribute:box_quality")}}
        ),
    )
    rc = run_samples_import(
        monkeypatch, service, "--as", "jiang", "--set", "bad_fit_review", "--apply"
    )
    assert rc == EXIT_FATAL
    # refusal at the gate: login + relations only, nothing imported
    assert service.requests == [("POST", LOGIN), ("GET", RELATIONS)]


# ---- 5. backflow rewrite failure ----------------------------------------------


def test_backflow_apply_rewrite_oserror_exits_2_and_keeps_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    excluded = tmp_path / "rolepool_excluded.jsonl"
    original = '{"uid": "u001", "track_id": 7, "exclude_reason": "conf 0.62"}\n'
    excluded.write_text(original, encoding="utf-8")
    digest = hashlib.sha256(excluded.read_bytes()).hexdigest()

    service = MockConfirmService()
    service.on("POST", LOGIN, login_ok)
    service.on(
        "GET",
        RELATIONS,
        lambda: json_ok(
            {
                "sets": {
                    backflow_script.RELATIONS_KEY: relation(
                        "rolepool_20260919",
                        "attribute:role",
                        fingerprint={"file": excluded.name, "sha256": digest},
                    )
                }
            }
        ),
    )
    meta_payload: object = sgcc_metas.role_meta().model_dump(mode="json", exclude_none=True)
    service.on("GET", meta_path(SET), lambda: json_ok(meta_payload))
    samples_payload: object = [
        {
            "stem": "u001_7",
            "image_url": "/api/img?p=x",
            "verdicts": [],
            "truth": {"annotator": "jiang", "value": sgcc_metas.CUSTOMER_ID},
        }
    ]
    service.on("GET", samples_path(SET), lambda: json_ok(samples_payload))

    monkeypatch.setattr(backflow_script, "os", ReplaceFailingOs())
    rc = run_backflow(monkeypatch, service, excluded, "--as", "jiang", "--apply")
    assert rc == EXIT_REFUSED
    # the failed rewrite left the owned file byte-identical
    assert excluded.read_text(encoding="utf-8") == original


# ---- 6. local input guards -----------------------------------------------------


def test_role_builder_unreadable_excluded_exits_2(tmp_path: Path) -> None:
    missing = tmp_path / "absent.jsonl"
    rc_missing: int = role_script.main(["--excluded", str(missing), "--dry-run"])
    assert rc_missing == EXIT_FATAL
    undecodable = tmp_path / "rolepool_excluded.jsonl"
    undecodable.write_bytes(b"\xff\xfe\x00not-utf8")
    rc_bad: int = role_script.main(["--excluded", str(undecodable), "--dry-run"])
    assert rc_bad == EXIT_FATAL


def test_init_unreadable_fingerprint_source_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    service = MockConfirmService()
    service.on("POST", LOGIN, login_ok)
    service.on("GET", PROJECTS, lambda: json_ok([]))
    service.on("POST", PROJECTS, created)
    service.on("GET", GROUPS, lambda: json_ok([]))
    service.on("POST", GROUPS, created)
    service.on("GET", SETS, lambda: json_ok([]))
    patch_fingerprint_sources(monkeypatch, tmp_path / "nonexistent", missing=True)
    rc = run_init(monkeypatch, service)
    assert rc == EXIT_FATAL
