#!/usr/bin/env python3
"""裁决工作台服务端——看图点选标注的通用轻量工具（2026-09-20）。

设计契约（与 review_workbench.html 配套）：
  零依赖   标准库 http.server；无构建、无数据库、无前端框架
  任务包   gencheck …/review/tasks/<task>.json：
           {name, hint, options: [{key, label}], samples: [{id, images: [绝对路径…],
             meta: {…自由字段}}]}
  落盘     POST /api/verdict → append 到 …/review/results/<task>.jsonl
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
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HTML = Path(__file__).with_suffix(".html")
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


class Workbench:
    """任务包 + 结果落盘的线程安全核心（与 HTTP 层解耦，可单测）。"""

    def __init__(self, tasks_dir: Path, results_dir: Path) -> None:
        self.tasks: dict[str, dict] = {}
        self.allowed_images: set[Path] = set()
        for f in sorted(tasks_dir.glob("*.json")):
            task = json.loads(f.read_text())
            task["_file"] = f.name
            self.tasks[task["name"]] = task
            for s in task["samples"]:
                self.allowed_images.update(Path(p) for p in s["images"])
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def result_path(self, task: str) -> Path:
        return self.results_dir / f"{task}.jsonl"

    def progress(self) -> dict[str, list[str]]:
        """每任务已裁决的 sample_id 列表（同 id 去重，保留出现序）。"""
        out: dict[str, list[str]] = {}
        for name in self.tasks:
            p = self.result_path(name)
            seen: list[str] = []
            if p.exists():
                for line in p.read_text().splitlines():
                    if line.strip():
                        sid = json.loads(line)["sample_id"]
                        if sid not in seen:
                            seen.append(sid)
            out[name] = seen
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


class Handler(BaseHTTPRequestHandler):
    bench: Workbench  # 由 serve() 注入

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")  # 任务包/进度须实时（fetch 缓存会吃掉更新）
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802（http.server 接口名）
        u = urllib.parse.urlparse(self.path)
        if u.path == "/" or u.path == "/index.html":
            self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
        elif u.path == "/api/tasks":
            self._json({n: {"title": t.get("title", ""), "intro": t.get("intro", ""),
                            "hint": t.get("hint", ""), "options": t["options"],
                            "n_samples": len(t["samples"]), "file": t["_file"]}
                        for n, t in self.bench.tasks.items()})
        elif u.path == "/api/task":
            name = urllib.parse.parse_qs(u.query).get("name", [""])[0]
            task = self.bench.tasks.get(name)
            self._json(task if task else {"error": f"未知任务 {name}"}, 200 if task else 404)
        elif u.path == "/api/progress":
            self._json(self.bench.progress())
        elif u.path == "/img":
            p = Path(urllib.parse.unquote(urllib.parse.parse_qs(u.query).get("p", [""])[0]))
            if p not in self.bench.allowed_images or not p.is_file():
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
            self.bench.append(body["task"], body["sample_id"], body["choice"],
                              body.get("note", ""))
            self._json({"ok": True})
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self._json({"error": str(e)}, 400)

    def log_message(self, fmt: str, *args) -> None:  # 静默常规访问日志（错误仍抛）
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="裁决工作台（看图点选标注）")
    ap.add_argument("--tasks", type=Path, required=True, help="任务包目录（*.json）")
    ap.add_argument("--results", type=Path, required=True, help="结果 jsonl 落盘目录")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()

    bench = Workbench(args.tasks, args.results)
    Handler.bench = bench
    if not bench.tasks:
        raise SystemExit(f"任务包目录无 *.json: {args.tasks}")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    names = ", ".join(f"{n}({len(t['samples'])})" for n, t in bench.tasks.items())
    print(f"裁决工作台 http://127.0.0.1:{args.port}  任务: {names}")
    print(f"结果目录: {args.results}")
    server.serve_forever()


if __name__ == "__main__":
    main()
