#!/usr/bin/env python3
"""裁决工作台服务端——看图点选标注的通用轻量工具（2026-09-20）。

设计契约（与 review_workbench.html 配套）：
  零依赖   标准库 http.server；无构建、无数据库、无前端框架
  轮次     tasks/ 下每轮一个子目录 r<序号>_<日期>[_<说明>]（如 r01_2026-09-19）；
           results/ 同名子目录承接该轮结果——各轮数据物理隔离，不混写
  任务包   <round>/<task>.json：
           {name, title, intro, hint, options: [{key, label, desc}],
            samples: [{id, images: [绝对路径…], meta: {…自由字段}}]}
  落盘     POST /api/verdict → append 到 results/<round>/<task>.jsonl
           （{ts, sample_id, choice, note}；同 id 重复裁决 append 新行，合并取最后）
  断点续审  进度即结果文件——重启后 GET /api/progress 读回已裁集
  图白名单  仅服务任务包 samples 声明过的绝对路径（GET /img?p=<urlencoded>）
用法：
  python -m jxl.bin.review_workbench --tasks <gencheck>/review/tasks \
      --results <gencheck>/review/results [--port 8787]
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HTML = Path(__file__).with_suffix(".html")
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
ROUND_DIR = re.compile(r"^r(\d+)_(\d{4}-\d{2}-\d{2})(?:_(.+))?$")


class Round:
    """一轮沉淀的任务包集合 + 结果落盘（与 HTTP 层解耦，可单测）。"""

    def __init__(self, dir_: Path, results_dir: Path) -> None:
        m = ROUND_DIR.match(dir_.name)
        if not m:
            raise ValueError(f"轮次目录须形如 r01_2026-09-19[_说明]: {dir_}")
        self.dir, self.results_dir = dir_, results_dir
        self.no, self.date, self.note = int(m[1]), m[2], m[3] or ""
        self.tasks: dict[str, dict] = {}
        self.allowed_images: set[Path] = set()
        for f in sorted(dir_.glob("*.json")):
            task = json.loads(f.read_text())
            task["_file"] = f.name
            self.tasks[task["name"]] = task
            for s in task["samples"]:
                self.allowed_images.update(Path(p) for p in s["images"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def result_path(self, task: str) -> Path:
        return self.results_dir / f"{task}.jsonl"

    def progress(self) -> dict[str, dict[str, dict]]:
        """每任务已裁决样本 → 最后一次裁决 {choice, note}（同 id 后写覆盖先写）。"""
        out: dict[str, dict[str, dict]] = {}
        for name in self.tasks:
            p = self.result_path(name)
            last: dict[str, dict] = {}
            if p.exists():
                for line in p.read_text().splitlines():
                    if line.strip():
                        r = json.loads(line)
                        last[r["sample_id"]] = {"choice": r["choice"], "note": r.get("note", "")}
            out[name] = last
        return out

    def append(self, task: str, sample_id: str, choice: str, note: str) -> None:
        if task not in self.tasks:
            raise KeyError(f"未知任务: {task}")
        options = {o["key"] for o in self.tasks[task]["options"]}
        if choice not in options:
            raise ValueError(f"非法选项 {choice}（任务 {task} 允许 {sorted(options)}）")
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "sample_id": sample_id,
               "choice": choice, "note": note}
        with self._lock:
            with self.result_path(task).open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


class Bench:
    """轮次容器：发现 tasks/ 下的轮次目录并持有各轮 Round。"""

    def __init__(self, tasks_root: Path, results_root: Path) -> None:
        self.rounds: dict[str, Round] = {}
        for d in sorted(tasks_root.iterdir()):
            if d.is_dir() and ROUND_DIR.match(d.name):
                self.rounds[d.name] = Round(d, results_root / d.name)
        if not self.rounds:
            raise SystemExit(
                f"未发现轮次目录（r01_2026-09-19 形式）: {tasks_root}")
        self.latest = max(self.rounds, key=lambda k: self.rounds[k].no)

    def get(self, name: str | None) -> Round:
        return self.rounds[name or self.latest]


class Handler(BaseHTTPRequestHandler):
    bench: Bench  # 由 serve() 注入

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")  # 任务包/进度须实时（fetch 缓存会吃掉更新）
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def _q(self, u, key: str) -> str | None:
        return urllib.parse.parse_qs(u.query).get(key, [None])[0]

    def do_GET(self) -> None:  # noqa: N802（http.server 接口名）
        u = urllib.parse.urlparse(self.path)
        if u.path == "/" or u.path == "/index.html":
            self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
        elif u.path == "/api/rounds":
            self._json([{"dir": k, "no": r.no, "date": r.date, "note": r.note,
                         "n_tasks": len(r.tasks),
                         "n_samples": sum(len(t["samples"]) for t in r.tasks.values())}
                        for k, r in sorted(self.bench.rounds.items(),
                                           key=lambda kv: kv[1].no)])
        elif u.path == "/api/tasks":
            b = self.bench.get(self._q(u, "round"))
            self._json({n: {"title": t.get("title", ""), "intro": t.get("intro", ""),
                            "hint": t.get("hint", ""), "options": t["options"],
                            "n_samples": len(t["samples"]), "file": t["_file"]}
                        for n, t in b.tasks.items()})
        elif u.path == "/api/task":
            b = self.bench.get(self._q(u, "round"))
            name = self._q(u, "name") or ""
            task = b.tasks.get(name)
            self._json(task if task else {"error": f"未知任务 {name}"}, 200 if task else 404)
        elif u.path == "/api/progress":
            self._json(self.bench.get(self._q(u, "round")).progress())
        elif u.path == "/img":
            p = Path(urllib.parse.unquote(self._q(u, "p") or ""))
            allowed = any(p in r.allowed_images for r in self.bench.rounds.values())
            if not allowed or not p.is_file():
                self._json({"error": "path not allowed"}, 403)
            else:
                self._send(200, p.read_bytes(), MIME.get(p.suffix.lower(), "application/octet-stream"))
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if urllib.parse.urlparse(self.path).path != "/api/verdict":
            self._json({"error": "not found"}, 404)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.bench.get(body.get("round")).append(
                body["task"], body["sample_id"], body["choice"], body.get("note", ""))
            self._json({"ok": True})
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self._json({"error": str(e)}, 400)

    def log_message(self, fmt: str, *args) -> None:  # 静默常规访问日志（错误仍抛）
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="裁决工作台（看图点选标注）")
    ap.add_argument("--tasks", type=Path, required=True,
                    help="任务包根目录（其下每轮一个 r01_2026-09-19 形式子目录）")
    ap.add_argument("--results", type=Path, required=True, help="结果根目录（按轮次建同名子目录）")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()

    bench = Bench(args.tasks, args.results)
    Handler.bench = bench
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    rounds = ", ".join(
        f"第{r.no}轮 {r.date}({sum(len(t['samples']) for t in r.tasks.values())}样本)"
        for r in sorted(bench.rounds.values(), key=lambda r: r.no))
    print(f"裁决工作台 http://127.0.0.1:{args.port}  轮次: {rounds}")
    print(f"结果目录: {args.results}")
    server.serve_forever()


if __name__ == "__main__":
    main()
