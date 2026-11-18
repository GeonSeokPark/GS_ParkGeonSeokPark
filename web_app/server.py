"""Serve the DeepProtect test app on the loopback interface."""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
from threading import Lock, Thread
from time import time
from urllib.parse import parse_qs, urlsplit

from .settings import FIELDS, GROUPS, PRESETS, validate

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).with_name("static")
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
JOB_ID = re.compile(r"[0-9a-f]{24}")
ARTIFACTS = {"aligned.png", "protected.png", "metrics.json", "run.log"}
ERRORS = {
    "format": "JPG 또는 PNG 형식의 단일 사진을 선택해 주세요.",
    "dimensions": "사진의 픽셀 수가 2,400만을 초과합니다.",
    "invalid": "사진을 읽을 수 없습니다. 정상적으로 열리는 JPG 또는 PNG를 선택해 주세요.",
    "face": "사진에서 얼굴을 찾지 못했습니다. 정면 얼굴이 잘 보이는 사진을 선택해 주세요.",
}
MAX_UPLOAD = 25 * 1024 * 1024


class Jobs:
    def __init__(self, args):
        self.args = args
        self.root = args.jobs
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.active = None
        for path in self.root.glob("*/status.json"):
            if not JOB_ID.fullmatch(path.parent.name):
                continue
            state = self.read(path.parent.name)
            if state is None:
                continue
            if state["status"] in {"aligning", "running"}:
                self.update(path.parent.name, status="failed", error="서버가 재시작되어 작업이 중단됐습니다.")

    def update(self, job, **fields):
        with self.lock:
            self.write(job, {**(self.read(job) or {}), **fields, "updated": time()})

    def write(self, job, state):
        path = self.root / job / "status.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def read(self, job):
        if JOB_ID.fullmatch(job):
            try:
                return json.loads((self.root / job / "status.json").read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        return None

    def submit(self, body):
        with self.lock:
            if self.active:
                return None
            job = secrets.token_hex(12)
            directory = self.root / job
            directory.mkdir()
            (directory / "upload").write_bytes(body)
            self.write(job, {"id": job, "status": "aligning", "stage": "align", "updated": time()})
            self.active = job
        Thread(target=self.run, args=(job, True), daemon=True).start()
        return job

    def start(self, job, settings):
        with self.lock:
            state = self.read(job)
            if self.active or not state or state["status"] not in {"ready", "done", "failed"}:
                return None
            source = self.root / job / "aligned.png"
            if not source.is_file():
                return None
            if state["status"] != "ready":
                parent, job = job, secrets.token_hex(12)
                directory = self.root / job
                directory.mkdir()
                shutil.copyfile(source, directory / "aligned.png")
                state = {"id": job, "parent": parent}
            self.write(job, {**state, "status": "running", "stage": "reconstruct",
                             "current": 0, "total": settings["reconstruction_steps"],
                             "settings": settings, "started": time(), "updated": time()})
            self.active = job
        Thread(target=self.run, args=(job, False), daemon=True).start()
        return job

    def run(self, job, align):
        directory, args = self.root / job, self.args
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        if align:
            prefix = args.align_python.parent
            env["PATH"] = os.pathsep.join((str(prefix / "Library" / "bin"), str(prefix), env["PATH"]))
            command = [
                args.align_python, "-m", "web_app.align_upload", directory / "upload",
                directory, args.predictor,
            ]
        else:
            command = [sys.executable, "-m", "web_app.worker", directory,
                       "--models", args.models, "--bank", args.bank]
        outcome = {"status": "failed", "error": "처리 중 오류가 발생했습니다. 실행 로그를 확인해 주세요."}
        error_kind = None
        try:
            with (directory / "run.log").open("a", encoding="utf-8") as log, subprocess.Popen(
                list(map(str, command)), cwd=ROOT, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
            ) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith("WEB "):
                        self.update(job, **json.loads(line[4:]))
                    elif line.startswith("UPLOAD_ERROR:"):
                        error_kind = line.strip().split(":", 1)[1]
                if process.wait() == 0:
                    expected = "aligned.png" if align else "metrics.json"
                    if (directory / expected).is_file():
                        outcome = {"status": "ready" if align else "done",
                                   "stage": "aligned" if align else "complete"}
                elif error_kind:
                    outcome["error"] = ERRORS.get(error_kind, outcome["error"])
        except (OSError, ValueError) as exc:
            with (directory / "run.log").open("a", encoding="utf-8") as log:
                log.write(f"Web job error: {exc}\n")
        finally:
            with self.lock:
                try:
                    self.write(job, {**self.read(job), **outcome, "updated": time()})
                finally:
                    self.active = None


class Handler(BaseHTTPRequestHandler):
    def valid_host(self):
        return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

    def headers_for(self, status, content_type, length, attachment=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; img-src 'self' blob:; frame-ancestors 'none'; form-action 'self'")
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
        self.end_headers()

    def json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.headers_for(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def file(self, path, content_type, attachment=None):
        if not path.is_file():
            self.json(HTTPStatus.NOT_FOUND, {"error": "파일을 찾을 수 없습니다."})
            return
        with path.open("rb") as file:
            self.headers_for(HTTPStatus.OK, content_type, os.fstat(file.fileno()).st_size, attachment)
            shutil.copyfileobj(file, self.wfile)

    def do_GET(self):
        if not self.valid_host():
            self.json(HTTPStatus.FORBIDDEN, {"error": "로컬 주소에서 열어 주세요."})
            return
        url = urlsplit(self.path)
        if url.path in ASSETS:
            name, content_type = ASSETS[url.path]
            self.file(STATIC / name, content_type)
            return
        if url.path == "/api/status":
            self.json(HTTPStatus.OK, {"busy": self.server.jobs.active is not None})
            return
        if url.path == "/api/settings":
            self.json(HTTPStatus.OK, {"fields": FIELDS, "groups": GROUPS, "presets": PRESETS})
            return
        components = url.path.strip("/").split("/")
        if len(components) >= 3 and components[:2] == ["api", "jobs"]:
            job = components[2]
            with self.server.jobs.lock:
                state = self.server.jobs.read(job)
            if state:
                if len(components) == 3:
                    aligned = (self.server.jobs.root / job / "aligned.png").is_file()
                    self.json(HTTPStatus.OK, {**state, "aligned_available": aligned})
                    return
                name = "/".join(components[3:])
                if name in ARTIFACTS or re.fullmatch(r"candidates/(0[1-9]|1[0-9]|2[0-4])\.png", name):
                    kind = {"png": "image/png", "json": "application/json", "log": "text/plain; charset=utf-8"}
                    download = Path(name).name if parse_qs(url.query).get("download") else None
                    self.file(self.server.jobs.root / job / name, kind[name.rsplit(".", 1)[1]], download)
                    return
        self.json(HTTPStatus.NOT_FOUND, {"error": "요청한 항목을 찾을 수 없습니다."})

    def do_POST(self):
        origin = self.headers.get("Origin")
        if not self.valid_host() or (origin and origin != f"http://{self.headers['Host']}"):
            self.json(HTTPStatus.FORBIDDEN, {"error": "로컬 주소에서 요청해 주세요."})
            return
        if self.path == "/api/jobs":
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if not 0 < length <= MAX_UPLOAD:
                self.json(HTTPStatus.BAD_REQUEST, {"error": "25 MB 이하의 JPG 또는 PNG를 선택해 주세요."})
                return
            self.connection.settimeout(20)
            try:
                body = self.rfile.read(length)
            except TimeoutError:
                self.json(HTTPStatus.REQUEST_TIMEOUT, {"error": "사진 전송 시간이 초과되었습니다."})
                return
            if len(body) != length:
                self.json(HTTPStatus.BAD_REQUEST, {"error": "사진 전송이 중단됐습니다."})
                return
            job = self.server.jobs.submit(body)
            if job:
                self.json(HTTPStatus.ACCEPTED, {"id": job})
            else:
                self.json(HTTPStatus.CONFLICT, {"error": "다른 사진을 처리 중입니다."})
            return
        match = re.fullmatch(r"/api/jobs/([0-9a-f]{24})/start", self.path)
        if match:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 8192:
                    raise ValueError("설정 요청이 너무 큽니다.")
                self.connection.settimeout(10)
                settings = validate(json.loads(self.rfile.read(length)) if length else {})
            except (ValueError, UnicodeError, TimeoutError) as exc:
                message = str(exc) if type(exc) is ValueError else "설정 요청을 읽을 수 없습니다."
                self.json(HTTPStatus.BAD_REQUEST, {"error": message})
                return
            job = self.server.jobs.start(match[1], settings)
            if job:
                self.json(HTTPStatus.ACCEPTED, {"id": job})
                return
        self.json(HTTPStatus.CONFLICT, {"error": "정렬이 완료되고 다른 작업이 끝나면 실행할 수 있습니다."})


def main():
    models = ROOT / "checkpoints"
    align_python = ROOT / ".runtime" / "align" / "python.exe"
    if not (ROOT / ".venv" / "Scripts" / "python.exe").is_file() and len(ROOT.parents) > 1:
        work = ROOT.parents[1] / "work"
        if (work / "testenv" / "Scripts" / "python.exe").is_file():
            models = work / "checkpoints"
            align_python = work / "alignment-short" / "python.exe"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--models", type=Path, default=models)
    parser.add_argument("--bank", type=Path, default=ROOT / "data" / "banks" / "vggface2_hq_4605.h5")
    parser.add_argument("--align-python", type=Path, default=align_python)
    parser.add_argument("--predictor", type=Path)
    parser.add_argument("--jobs", type=Path, default=ROOT / "outputs" / "web-jobs")
    args = parser.parse_args()
    if args.predictor is None:
        args.predictor = args.models / "shape_predictor_68_face_landmarks.dat"
    for key in ("models", "bank", "align_python", "predictor", "jobs"):
        setattr(args, key, getattr(args, key).resolve())
    missing = [path for path in (args.bank, args.align_python, args.predictor, *(
        args.models / name for name in ("ffhq.pkl", "e4e.pt", "arcface.pth", "FaRL.pth")
    )) if not path.is_file()]
    if missing:
        parser.error("Missing files: " + ", ".join(map(str, missing)))
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.jobs = Jobs(args)
        print(f"DeepProtect is ready at http://127.0.0.1:{server.server_port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
