#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from contextlib import contextmanager


URL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

STATE = {
    "nodes": {},
    "task": None,
    "tasks": [],
    "acks": {},
    "next_master_port": 20000,
    "next_task_seq": 0,
}
LOCK = threading.Lock()

FINAL_STATUSES = {"completed", "failed", "cancelled"}


def task_set_status(task, status):
    task["status"] = status
    task["status_changed_at"] = time.time()


_TQDM_RE = re.compile(r'(\d+)%\|[^|]*\|\s*(\d+)/(\d+)')
_TQDM_RATE_RE = re.compile(r'([\d.]+)\s*(it/s|s/it)')
_TQDM_LOSS_RE = re.compile(r'(?:loss|Loss)[=:]?\s*(\d+\.?\d*(?:[eE][+-]?\d+)?)')


def _parse_tqdm_line(line):
    m = _TQDM_RE.search(line)
    if not m:
        return None
    pct, cur, total = int(m.group(1)), int(m.group(2)), int(m.group(3))
    info = {"cur_step": cur, "max_steps": total, "pct": pct}
    rm = _TQDM_RATE_RE.search(line)
    if rm:
        info["rate"] = float(rm.group(1))
    lm = _TQDM_LOSS_RE.search(line)
    if lm:
        try:
            info["loss"] = float(lm.group(1))
        except ValueError:
            pass
    return info


def _stderr_reader(proc, item):
    """Background thread: read stderr in real-time, parse tqdm progress."""
    buf = ""
    lines = item.setdefault("stderr_lines", [])
    try:
        while True:
            chunk = proc.stderr.read(256)
            if not chunk:
                break
            buf += chunk
            segments = re.split(r'[\r\n]', buf)
            buf = segments[-1]
            for seg in segments[:-1]:
                seg = seg.strip()
                if not seg:
                    continue
                lines.append(seg)
                parsed = _parse_tqdm_line(seg)
                if parsed:
                    item["progress"] = parsed
    except (OSError, ValueError):
        pass
    if buf.strip():
        lines.append(buf.strip())
        parsed = _parse_tqdm_line(buf)
        if parsed:
            item["progress"] = parsed


def parse_gpu_spec(spec):
    if not spec:
        return None
    gpus = []
    for part in str(spec).replace("+", " ").replace("|", " ").split():
        if "-" in part:
            start, end = part.split("-", 1)
            if start.isdigit() and end.isdigit():
                gpus.extend(range(int(start), int(end) + 1))
        elif part.isdigit():
            gpus.append(int(part))
    return sorted(set(gpus))


def parse_node_spec(spec):
    if ":" not in spec:
        return spec, None
    addr, gpu_spec = spec.split(":", 1)
    return addr, parse_gpu_spec(gpu_spec)


def task_specs_for_addr(task, addr):
    specs = task.get("node_addrs") or []
    if not specs:
        return [(addr, None)]
    matches = []
    for spec in specs:
        node_addr, gpus = parse_node_spec(spec)
        if node_addr == addr:
            matches.append((node_addr, gpus))
    return matches


def task_matches_addr(task, addr):
    return bool(task_specs_for_addr(task, addr))


def task_gpus_for_addr(task, addr):
    matches = task_specs_for_addr(task, addr)
    if not matches:
        return None
    _, gpus = matches[0]
    return gpus


def gpu_sets_overlap(left, right):
    if left is None or right is None:
        return True
    return bool(set(left) & set(right))


def node_can_accept_task(node, task, addr):
    requested = task_gpus_for_addr(task, addr)
    gpu_slots = node.get("gpu_slots_config", 1)
    gpu_slots_used = node.get("gpu_slots_used", {})
    if requested is None:
        total = node.get("gpus", 0)
        for gpu in range(total):
            if gpu_slots_used.get(str(gpu), 0) >= gpu_slots:
                return False
        return True
    for gpu in requested:
        if gpu_slots_used.get(str(gpu), 0) >= gpu_slots:
            return False
    return True


def load_cluster_config(path):
    data = {}
    node_addrs = []
    if not path:
        return data
    if not os.path.exists(path):
        return data

    in_nodes = False
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("NODE_ADDRS=("):
                in_nodes = True
                rest = line[len("NODE_ADDRS=("):].strip()
                if rest.endswith(")"):
                    rest = rest[:-1]
                    in_nodes = False
                node_addrs.extend(shlex.split(rest))
                continue
            if in_nodes:
                if line == ")":
                    in_nodes = False
                    continue
                if line.endswith(")"):
                    line = line[:-1].strip()
                    in_nodes = False
                node_addrs.extend(shlex.split(line))
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                data[key.strip()] = value.strip().strip('"').strip("'")

    data["NODE_ADDRS"] = node_addrs
    return data


def parse_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def local_ipv4_addrs():
    addrs = set()
    try:
        host = socket.gethostname()
        for item in socket.getaddrinfo(host, None, socket.AF_INET):
            addrs.add(item[4][0])
    except OSError:
        pass

    try:
        out = subprocess.check_output(["hostname", "-I"], text=True, stderr=subprocess.DEVNULL)
        for part in out.split():
            if part.count(".") == 3:
                addrs.add(part)
    except Exception:
        pass

    return sorted(addr for addr in addrs if not addr.startswith("127."))


def detect_node_addr(config):
    forced = os.environ.get("CTM_NODE_ADDR")
    if forced:
        return forced
    local = set(local_ipv4_addrs())
    for addr in config.get("NODE_ADDRS", []):
        if addr in local:
            return addr
    return next(iter(local), socket.gethostname())


def post_json(url, payload, timeout=10):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with URL_OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def get_json(url, timeout=10):
    with URL_OPENER.open(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def print_pool():
    now = time.time()
    with LOCK:
        nodes = dict(STATE["nodes"])
        task = STATE["task"]
        tasks = list(STATE.get("tasks", []))
        acks = dict(STATE["acks"])

    print("\n=== CTM Pool ===", flush=True)
    for addr, node in sorted(nodes.items()):
        age = now - node["last_seen"]
        gpu_summary = node.get("gpu_summary") or f"{node.get('gpus', '?')} GPU(s)"
        print(
            f"  {addr:15s} rank={node.get('rank', '?')} "
            f"host={node.get('hostname', '?')} status={node.get('status', '?')} "
            f"gpus={gpu_summary} busy={node.get('busy_gpus', [])} seen={age:.1f}s",
            flush=True,
        )
    if tasks:
        for t in tasks:
            assigned = ",".join(t.get("node_addrs") or [])
            st = t.get("status", "pending")
            age = now - t.get("created_at", now)
            line = (
                f"  task={t['task_id']} status={st} "
                f"nodes={assigned or 'all'} age={age:.0f}s "
                f"args={t.get('extra_args', '')}"
            )
            if t.get("return_code") is not None:
                line += f" rc={t['return_code']}"
            print(line, flush=True)
            for addr, ack in sorted(acks.get(t["task_id"], {}).items()):
                print(f"    ack {addr}: {ack.get('status')} {ack.get('message', '')}", flush=True)
    elif task:
        st = task.get("status", "pending")
        print(f"  task={task['task_id']} status={st} args={task.get('extra_args', '')}", flush=True)
        legacy_acks = acks.get(task["task_id"], acks)
        for addr, ack in sorted(legacy_acks.items()):
            print(f"    ack {addr}: {ack.get('status')} {ack.get('message', '')}", flush=True)
    print("================\n", flush=True)


class PoolHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        return json.loads(raw or "{}")

    def _write_json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/status":
            with LOCK:
                payload = {
                    "nodes": STATE["nodes"],
                    "task": STATE["task"],
                    "tasks": STATE.get("tasks", []),
                    "acks": STATE["acks"],
                }
            self._write_json(payload)
            return
        if parsed.path == "/task":
            query = urllib.parse.parse_qs(parsed.query)
            addr = query.get("node_addr", [""])[0]
            with LOCK:
                task = None
                acked = False
                for candidate in STATE.get("tasks", []):
                    if candidate.get("status") in FINAL_STATUSES:
                        continue
                    if not task_matches_addr(candidate, addr):
                        continue
                    node = STATE["nodes"].get(addr, {})
                    if not node_can_accept_task(node, candidate, addr):
                        continue
                    if STATE["acks"].get(candidate["task_id"], {}).get(addr):
                        acked_task = candidate
                        acked = True
                        continue
                    task = candidate
                    break
                if task is None and not acked:
                    legacy = STATE.get("task")
                    if (
                        legacy is not None
                        and legacy.get("status") not in FINAL_STATUSES
                        and task_matches_addr(legacy, addr)
                        and node_can_accept_task(node, legacy, addr)
                        and not STATE["acks"].get(legacy["task_id"], {}).get(addr)
                    ):
                        task = legacy
            self._write_json({"task": task})
            return
        self.send_error(404)

    def _handle_heartbeat(self, payload):
        addr = payload["node_addr"]
        announce = False
        task_progress = payload.get("task_progress", {})
        with LOCK:
            old = STATE["nodes"].get(addr)
            announce = old is None or old.get("status") != payload.get("status")
            payload["last_seen"] = time.time()
            STATE["nodes"][addr] = payload
            if task_progress:
                for tid, prog in task_progress.items():
                    for t in STATE.get("tasks", []):
                        if t["task_id"] == tid and t["status"] == "running":
                            t["progress"] = prog
                            break
        if announce:
            print(f"[pool] node online/update: {addr} status={payload.get('status')}", flush=True)
        self._write_json({"ok": True})

    def _handle_submit(self, payload):
        with LOCK:
            STATE["next_task_seq"] = int(STATE.get("next_task_seq", 0)) + 1
            task_id = (
                time.strftime("%Y%m%d_%H%M%S")
                + f"_{int(time.time() * 1000000) % 1000000:06d}"
                + f"_{STATE['next_task_seq']:04d}"
            )
            master_port = int(payload.get("master_port") or STATE.get("next_master_port", 20000))
            STATE["next_master_port"] = 20000 + ((master_port - 19999) % 30000)
        task = {
            "task_id": task_id,
            "config": payload["config"],
            "extra_args": payload.get("extra_args", ""),
            "node_addrs": payload.get("node_addrs") or [],
            "env": payload.get("env") or {},
            "master_port": master_port,
            "created_at": time.time(),
            "status": "pending",
            "status_changed_at": time.time(),
        }
        with LOCK:
            STATE["task"] = task
            STATE.setdefault("tasks", []).append(task)
            STATE["acks"].setdefault(task_id, {})
        nodes = ",".join(task["node_addrs"]) if task["node_addrs"] else "all"
        print(f"[pool] new task: {task_id} ({nodes})", flush=True)
        self._write_json({"ok": True, "task": task})

    def _handle_ack(self, payload):
        addr = payload["node_addr"]
        task_id = payload["task_id"]
        with LOCK:
            STATE["acks"].setdefault(task_id, {})[addr] = payload
            for t in STATE["tasks"]:
                if t["task_id"] == task_id and t["status"] == "pending":
                    task_set_status(t, "running")
                    break
        self._write_json({"ok": True})

    def _handle_complete(self, payload):
        task_id = payload["task_id"]
        addr = payload.get("node_addr", "?")
        rc = payload.get("return_code")
        status = "completed" if rc == 0 else "failed"
        with LOCK:
            for t in STATE["tasks"]:
                if t["task_id"] == task_id:
                    task_set_status(t, status)
                    t["return_code"] = rc
                    break
        print(f"[pool] task {task_id} {status} (rc={rc}) {addr}", flush=True)
        if rc != 0:
            stderr_tail = payload.get("stderr_tail", [])
            for line in stderr_tail:
                print(f"  {line}", flush=True)
        self._write_json({"ok": True})

    def _handle_cancel(self, payload):
        task_id = payload.get("task_id")
        with LOCK:
            cancelled = []
            for t in STATE["tasks"]:
                if t["task_id"] == task_id and t["status"] not in FINAL_STATUSES:
                    task_set_status(t, "cancelled")
                    cancelled.append(t)
            if not task_id:
                for t in STATE["tasks"]:
                    if t["status"] == "pending":
                        task_set_status(t, "cancelled")
                        cancelled.append(t)
        for t in cancelled:
            print(f"[pool] task {t['task_id']} cancelled", flush=True)
        self._write_json({"ok": True, "cancelled": [t["task_id"] for t in cancelled]})

    def do_POST(self):
        if self.path == "/heartbeat":
            self._handle_heartbeat(self._read_json())
            return

        if self.path == "/submit":
            self._handle_submit(self._read_json())
            return

        if self.path == "/ack":
            self._handle_ack(self._read_json())
            return

        if self.path == "/complete":
            self._handle_complete(self._read_json())
            return

        if self.path == "/cancel":
            self._handle_cancel(self._read_json())
            return

        if self.path == "/clear":
            with LOCK:
                before = len(STATE["tasks"])
                STATE["tasks"] = [t for t in STATE["tasks"] if t["status"] not in FINAL_STATUSES]
                cleared = before - len(STATE["tasks"])
            print(f"[pool] cleared {cleared} finished task(s)", flush=True)
            self._write_json({"ok": True, "cleared": cleared})
            return

        self.send_error(404)


def run_server(args):
    server = ThreadingHTTPServer((args.host, args.port), PoolHandler)
    print(f"CTM pool server listening on {args.host}:{args.port}", flush=True)
    print("Workers will appear here when online.", flush=True)
    server.serve_forever()


def gpu_inventory():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        devices = []
        for line in out.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            devices.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_mb": int(float(parts[2])),
            })
        return devices
    except Exception:
        return []


def summarize_gpus(devices):
    if not devices:
        return "0 GPU(s)"
    groups = {}
    for dev in devices:
        key = (dev["name"], dev["memory_mb"])
        groups[key] = groups.get(key, 0) + 1
    chunks = []
    for (name, memory_mb), count in sorted(groups.items()):
        memory_gb = int(round(memory_mb / 1024))
        chunks.append(f"{count}x {name} {memory_gb}GB")
    return " + ".join(chunks)


def format_gpu_lines(devices):
    if not devices:
        return "  GPUs: none detected"
    lines = ["  GPUs:"]
    for dev in devices:
        memory_gb = dev["memory_mb"] / 1024
        lines.append(f"    [{dev['index']}] {dev['name']} {memory_gb:.1f}GB")
    return "\n".join(lines)


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


@contextmanager
def repo_update_lock(config):
    if not parse_bool(config.get("SHARED_REPO"), default=False):
        yield
        return

    repo_dir = config.get("REPO_DIR") or os.getcwd()
    lock_dir = os.path.join(repo_dir, ".ctm_pool")
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, "git_update.lock")
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def git_update_ff_only(config, process_head=None):
    before = git_head()
    remote = config.get("GIT_REMOTE", "origin")
    branch = config.get("GIT_BRANCH")
    if not branch:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    env = os.environ.copy()
    if config.get("GIT_HTTP_PROXY"):
        env["http_proxy"] = config["GIT_HTTP_PROXY"]
        env["HTTP_PROXY"] = config["GIT_HTTP_PROXY"]
    if config.get("GIT_HTTPS_PROXY"):
        env["https_proxy"] = config["GIT_HTTPS_PROXY"]
        env["HTTPS_PROXY"] = config["GIT_HTTPS_PROXY"]
    with repo_update_lock(config):
        before = git_head()
        fetch_proc = subprocess.run(
            ["git", "fetch", remote, branch],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        if fetch_proc.returncode != 0:
            after = git_head()
            return {
                "ok": False,
                "before": before,
                "after": after,
                "restart_needed": process_head not in (None, after),
                "output": fetch_proc.stdout.strip(),
            }

        proc = subprocess.run(
            ["git", "merge", "--ff-only", f"{remote}/{branch}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        after = git_head()

    after = git_head()
    output = "\n".join(
        chunk for chunk in (fetch_proc.stdout.strip(), proc.stdout.strip()) if chunk
    )
    return {
        "ok": proc.returncode == 0,
        "before": before,
        "after": after,
        "restart_needed": process_head not in (None, after),
        "output": output,
    }


def restart_worker_process():
    print("[worker] restarting to load updated code", flush=True)
    os.execv(sys.executable, [sys.executable, *sys.argv])


def kill_process_group(pgid, timeout=3.0):
    """SIGTERM then SIGKILL an entire process group.

    Prevents torchrun's child python workers from surviving as orphans (which
    leak GPU memory) when a task exits via OOM kill / NCCL hang / segfault.
    No-op if the group is already gone.
    """
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.3)
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, OSError):
            return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def cleanup_all_tasks(procs, signum=None):
    """Kill every running task's process group (used on worker shutdown)."""
    n = 0
    for item in list(procs.values()):
        pgid = item.get("pgid")
        if pgid is not None:
            kill_process_group(pgid)
            n += 1
    if n:
        print(f"[worker] cleaned up {n} task(s) on exit", flush=True)
    if signum is not None:
        sys.exit(128 + signum)


def auto_gpu_slots(gpus, config):
    if not gpus:
        return 1
    total_mb = gpus[0]["memory_mb"]
    total_gb = total_mb / 1024
    train_args = config.get("TRAIN_ARGS", "")
    d_model = 512
    m = re.search(r'--d_model\s+(\d+)', train_args)
    if m:
        d_model = int(m.group(1))
    gb_per_task = max(2.0, d_model * 0.008)
    slots = max(1, int(total_gb / gb_per_task))
    return min(slots, 64)


def gpu_free_memory():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        result = {}
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                result[int(parts[0])] = int(float(parts[1]))
        return result
    except Exception:
        return {}


def estimate_task_gb(train_args):
    d_model = 512
    m = re.search(r'--d_model\s+(\d+)', train_args)
    if m:
        d_model = int(m.group(1))
    gb = max(2.0, d_model * 0.01)
    m2 = re.search(r'--cross_tick_jepa_weight\s+(\S+)', train_args)
    if m2 and float(m2.group(1)) > 0:
        gb *= 1.3
    return gb


def run_worker(args):
    config = load_cluster_config(getattr(args, 'config', None))
    node_addr = args.node_addr or detect_node_addr(config) or socket.gethostname()
    rank = "?"
    base = f"http://{args.master_addr}:{args.port}"
    hostname = socket.gethostname()
    gpus = gpu_inventory()
    gpu_summary = summarize_gpus(gpus)
    status = "idle"
    procs = {}
    process_head = git_head()

    def _on_exit_signal(signum, frame):
        print(f"[worker] signal {signum} received, killing {len(procs)} task(s)", flush=True)
        cleanup_all_tasks(procs, signum)

    for _sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, _on_exit_signal)

    if args.gpu_slots < 1:
        args.gpu_slots = auto_gpu_slots(gpus, config)
        print(f"[worker] auto-slots: {args.gpu_slots} per GPU (d_model from config)", flush=True)

    print(f"CTM worker online: addr={node_addr} rank={rank} host={hostname}", flush=True)
    print(gpu_summary, flush=True)
    print(format_gpu_lines(gpus), flush=True)
    print(f"Polling pool server: {base}", flush=True)

    while True:
        finished = []
        for task_id, item in list(procs.items()):
            if item["proc"].poll() is not None:
                finished.append(task_id)
        for task_id in finished:
            item = procs.pop(task_id)
            _pgid = item.get("pgid")
            if _pgid is not None:
                kill_process_group(_pgid)
            rc = item["proc"].returncode
            stderr_thread = item.get("stderr_thread")
            if stderr_thread and stderr_thread.is_alive():
                stderr_thread.join(timeout=3)
            stderr_tail = (item.get("stderr_lines") or [])[-30:]
            print(
                f"[worker] task {task_id} exited rc={rc} "
                f"gpus={item.get('gpus') or 'all'}",
                flush=True,
            )
            if rc != 0 and stderr_tail:
                for line in stderr_tail:
                    print(f"  {line}", flush=True)
            try:
                post_json(f"{base}/complete", {
                    "node_addr": node_addr,
                    "task_id": task_id,
                    "return_code": rc,
                    "stderr_tail": stderr_tail,
                }, timeout=5)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"[worker] failed to report completion for {task_id}: {exc}", flush=True)

        busy_gpus = sorted({
            gpu
            for item in procs.values()
            for gpu in (item.get("gpus") or [])
        })
        running_tasks = sorted(procs)
        status = "idle" if not procs else "running:" + ",".join(running_tasks)

        gpu_slots_config = args.gpu_slots
        gpu_slots_used = {}
        for item in procs.values():
            for gpu in (item.get("gpus") or []):
                gpu_slots_used[str(gpu)] = gpu_slots_used.get(str(gpu), 0) + 1

        heartbeat = {
            "node_addr": node_addr,
            "rank": rank,
            "hostname": hostname,
            "status": status,
            "gpus": len(gpus),
            "gpu_summary": gpu_summary,
            "gpu_devices": gpus,
            "busy_gpus": busy_gpus,
            "running_tasks": running_tasks,
            "pid": next(iter(procs.values()))["proc"].pid if procs else None,
            "gpu_slots_config": gpu_slots_config,
            "gpu_slots_used": gpu_slots_used,
            "task_progress": {
                tid: item.get("progress", {})
                for tid, item in procs.items()
                if item.get("progress")
            },
        }
        try:
            post_json(f"{base}/heartbeat", heartbeat, timeout=5)
            task_resp = get_json(
                f"{base}/task?node_addr={urllib.parse.quote(node_addr)}", timeout=5
            )
            task = task_resp.get("task")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"[worker] pool unavailable: {exc}", flush=True)
            time.sleep(args.interval)
            continue

        if task and task["task_id"] not in procs:
            requested_gpus = task_gpus_for_addr(task, node_addr)
            reject = False
            reject_msg = ""
            if requested_gpus is not None:
                for gpu in requested_gpus:
                    used = gpu_slots_used.get(str(gpu), 0)
                    if used >= gpu_slots_config:
                        reject = True
                        reject_msg = f"slot full gpu={gpu} used={used}/{gpu_slots_config}"
                        break
            elif busy_gpus:
                reject = True
                reject_msg = f"busy gpus={busy_gpus}, requested=all"
            if not reject:
                task_gb = estimate_task_gb(task.get("extra_args", ""))
                total_gpu_gb = gpus[0]["memory_mb"] / 1024 if gpus else 80.0
                mem_cap = total_gpu_gb * 0.85
                gpus_to_check = requested_gpus if requested_gpus is not None else list(range(len(gpus)))
                for gpu in gpus_to_check:
                    est_usage = sum(
                        item.get("est_gb", 0)
                        for item in procs.values()
                        if gpu in (item.get("gpus") or [])
                    )
                    if est_usage + task_gb > mem_cap:
                        reject = True
                        reject_msg = f"mem budget gpu={gpu} est={est_usage:.1f}+{task_gb:.1f}GB cap={mem_cap:.1f}GB"
                        break
            if reject:
                print(f"[worker] task {task['task_id']} ignored: {reject_msg}", flush=True)
                post_json(f"{base}/ack", {
                    "node_addr": node_addr,
                    "task_id": task["task_id"],
                    "status": "busy",
                    "message": reject_msg,
                })
            else:
                if args.auto_pull:
                    pull = git_update_ff_only(config, process_head=process_head)
                    if pull["ok"]:
                        print(
                            f"[worker] git update ok: {pull['before']} -> {pull['after']}",
                            flush=True,
                        )
                        if pull.get("restart_needed") and args.restart_on_update:
                            restart_worker_process()
                    else:
                        msg = f"git update failed: {pull['output']}"
                        print(f"[worker] task {task['task_id']} rejected: {msg}", flush=True)
                        post_json(f"{base}/ack", {
                            "node_addr": node_addr,
                            "task_id": task["task_id"],
                            "status": "pull_failed",
                            "message": msg,
                        })
                        time.sleep(args.interval)
                        continue

                extra = shlex.split(task.get("extra_args", ""))
                cmd = ["bash", "scripts/train_cluster.sh", "--config", task["config"], *extra]
                env = os.environ.copy()
                if "--train_module" in extra:
                    env["TRAIN_ENTRY"] = "scripts/run_via_pool.py"
                env["CTM_NODE_ADDR"] = node_addr
                env["CTM_POOL_MASTER_PORT"] = str(task.get("master_port") or 29500)
                if task.get("node_addrs"):
                    env["CTM_POOL_NODE_ADDRS"] = ",".join(task["node_addrs"])
                if requested_gpus is not None:
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in requested_gpus)
                    env["NPROC_PER_NODE"] = str(len(requested_gpus))
                for k, v in task.get("env", {}).items():
                    env[k] = v
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                print(f"[worker] received task {task['task_id']}: {' '.join(shlex.quote(x) for x in cmd)}", flush=True)
                proc = subprocess.Popen(
                    cmd, stderr=subprocess.PIPE, text=True,
                    env=env, start_new_session=True,
                )
                try:
                    _pgid = os.getpgid(proc.pid)
                except OSError:
                    _pgid = proc.pid
                item = {
                    "proc": proc,
                    "pgid": _pgid,
                    "gpus": requested_gpus,
                    "stderr_lines": [],
                    "est_gb": task_gb,
                    "progress": {},
                }
                procs[task["task_id"]] = item
                t = threading.Thread(target=_stderr_reader, args=(proc, item), daemon=True)
                item["stderr_thread"] = t
                t.start()
                post_json(f"{base}/ack", {
                    "node_addr": node_addr,
                    "task_id": task["task_id"],
                    "status": "started",
                    "message": f"pid={proc.pid} gpus={requested_gpus or 'all'} port={env['CTM_POOL_MASTER_PORT']}",
                })

        time.sleep(args.interval)


def run_submit(args):
    extra_items = list(args.extra_args)
    if extra_items and extra_items[0] == "--":
        extra_items = extra_items[1:]
    extra_args = " ".join(shlex.quote(item) for item in extra_items)
    node_addrs = []
    if args.nodes:
        raw_nodes = []
        for item in args.nodes:
            raw_nodes.extend(part for part in item.split(",") if part)
        node_addrs = [node.strip() for node in raw_nodes if node.strip()]
    config_path = getattr(args, 'config', None) or ""
    payload = {"config": config_path, "extra_args": extra_args, "node_addrs": node_addrs}
    resp = post_json(f"http://{args.master_addr}:{args.port}/submit", payload)
    task = resp["task"]
    print(f"submitted task {task['task_id']}: {task.get('extra_args', '')}")

    if args.wait <= 0:
        return

    expected = {parse_node_spec(spec)[0] for spec in node_addrs} if node_addrs else set()
    deadline = time.time() + args.wait
    seen = set()
    while time.time() < deadline:
        status = get_json(f"http://{args.master_addr}:{args.port}/status")
        acks = status.get("acks", {}).get(task["task_id"], {})
        for addr, ack in sorted(acks.items()):
            if ack.get("task_id") == task["task_id"] and addr not in seen:
                seen.add(addr)
                print(f"ack {addr}: {ack.get('status')} {ack.get('message', '')}")
        if expected and expected.issubset(seen):
            print("all expected nodes acknowledged")
            return
        time.sleep(1)
    missing = sorted(expected - seen)
    if missing:
        print(f"wait timeout, missing ack: {', '.join(missing)}")


def run_status(args):
    status = get_json(f"http://{args.master_addr}:{args.port}/status")
    print(json.dumps(status, indent=2, ensure_ascii=False))


def run_task(args):
    base = f"http://{args.master_addr}:{args.port}"

    if args.task_cmd == "history":
        return run_task_history(args, base)

    if args.task_cmd == "pending":
        return run_task_pending(args, base)

    if args.task_cmd == "list":
        status = get_json(f"{base}/status")
        tasks = status.get("tasks", [])
        if not tasks:
            print("no tasks")
            return
        now = time.time()
        running = [t for t in tasks if t.get("status") == "running"]
        pending = [t for t in tasks if t.get("status") == "pending"]
        completed = [t for t in tasks if t.get("status") == "completed"]
        failed = [t for t in tasks if t.get("status") == "failed"]
        print(f"\n  TASKS: {len(tasks)} total  ({len(running)} running, {len(pending)} pending, {len(completed)} completed, {len(failed)} failed)")
        print()
        if running:
            print(f"  RUNNING ({len(running)})")
            for t in running:
                name = _task_expname(t)
                age = int(now - t.get("created_at", now))
                print(f"    {name:50s}  running {age}s")
            print()
        if pending:
            print(f"  PENDING ({len(pending)})")
            for t in pending:
                name = _task_expname(t)
                print(f"    {name}")
            print()
        if completed or failed:
            print(f"  FINISHED ({len(completed)} ok, {len(failed)} failed)")
            for t in (completed + failed)[-10:]:
                name = _task_expname(t)
                print(f"    {name:50s}  {t.get('status')}")
        return
    elif args.task_cmd == "cancel":
        if not args.task_id:
            print("error: --task_id required", file=sys.stderr)
            sys.exit(1)
        resp = post_json(f"{base}/cancel", {"task_id": args.task_id})
        cancelled = resp.get("cancelled", [])
        if cancelled:
            print(f"cancelled: {', '.join(cancelled)}")
        else:
            print("nothing to cancel (task not found or already finished)")
    elif args.task_cmd == "cancel-pending":
        resp = post_json(f"{base}/cancel", {"task_id": None})
        cancelled = resp.get("cancelled", [])
        if cancelled:
            print(f"cancelled: {', '.join(cancelled)}")
        else:
            print("no pending tasks to cancel")
    elif args.task_cmd == "clear":
        resp = post_json(f"{base}/clear", {})
        cleared = resp.get("cleared", 0)
        if cleared:
            print(f"cleared {cleared} finished task(s)")
        else:
            print("no finished tasks to clear")
    elif args.task_cmd == "info":
        if not args.task_id:
            print("error: --task_id required", file=sys.stderr)
            sys.exit(1)
        status = get_json(f"{base}/status")
        tasks = status.get("tasks", [])
        found = [t for t in tasks if t["task_id"] == args.task_id]
        if not found:
            print(f"task {args.task_id} not found")
            return
        t = found[0]
        now = time.time()
        print(json.dumps(t, indent=2, ensure_ascii=False))
        acks = status.get("acks", {}).get(t["task_id"], {})
        if acks:
            print("acks:")
            for addr, ack in sorted(acks.items()):
                print(f"  {addr}: {ack.get('status')} {ack.get('message', '')}")
    elif args.task_cmd == "clean-fail":
        import glob
        pattern = os.path.join(args.metrics_dir, "*.fail.json")
        files = sorted(glob.glob(pattern))
        if not files:
            print("no .fail.json files found")
            return
        for f in files:
            os.remove(f)
        print(f"cleaned {len(files)} .fail.json file(s)")
    else:
        print(f"unknown task command: {args.task_cmd}", file=sys.stderr)
        sys.exit(1)


def run_task_history(args, base):
    status = get_json(f"{base}/status")
    tasks = status.get("tasks", [])
    duration = _parse_duration(args.duration)
    if duration is None:
        print(f"invalid duration: {args.duration}")
        return
    cutoff = time.time() - duration
    finished = [t for t in tasks if t.get("status") in ("completed", "failed") and t.get("status_changed_at", 0) >= cutoff]
    finished.sort(key=lambda t: t.get("status_changed_at", 0))

    completed = [t for t in finished if t["status"] == "completed"]
    failed = [t for t in finished if t["status"] == "failed"]
    span = _fmt_time(duration)
    print(f"\n  TASK HISTORY (last {span})")
    print(f"  Completed: {len(completed)}  Failed: {len(failed)}  Total: {len(finished)}")
    print()

    if completed:
        print(f"  COMPLETED")
        print(f"  {'NAME':40s}  {'DURATION':>8s}  {'RESULT':20s}")
        print(f"  {'-' * 40}  {'-' * 8}  {'-' * 20}")
        for t in completed:
            name = _task_expname(t)
            dur = t.get("status_changed_at", 0) - t.get("created_at", 0)
            info = "?"
            progress = t.get("progress", {})
            if progress.get("loss"):
                info = f"loss={progress['loss']:.4f}"
            else:
                m = _read_experiment_metrics(getattr(args, "metrics_dir", "runs/metrics"), name)
                if m:
                    info = f"acc={m.get('eval_accuracy','?')}" if m.get('eval_accuracy') else f"loss={m.get('loss','?')}"
            print(f"  {name:40s}  {_fmt_time(dur):>8s}  {info}")

    if failed:
        print(f"\n  FAILED")
        print(f"  {'NAME':40s}  {'DURATION':>8s}  {'REASON':30s}")
        print(f"  {'-' * 40}  {'-' * 8}  {'-' * 30}")
        for t in failed:
            name = _task_expname(t)
            dur = t.get("status_changed_at", 0) - t.get("created_at", 0)
            rc = t.get("return_code")
            fail_info = _read_failure(getattr(args, "metrics_dir", "runs/metrics"), name) if name else None
            reason = fail_info.get("error_type", f"rc={rc}") if fail_info else f"rc={rc}" if rc is not None else "?"
            print(f"  {name:40s}  {_fmt_time(dur):>8s}  {reason}")

    if not finished:
        print("  (no tasks finished in this window)")
    print()


def run_task_pending(args, base):
    status = get_json(f"{base}/status")
    tasks = status.get("tasks", [])
    pending = [t for t in tasks if t.get("status") == "pending"]
    if not pending:
        print("  no pending tasks")
        return
    print(f"\n  PENDING ({len(pending)})")
    for t in pending:
        extra = t.get("extra_args", "")
        name = _task_expname(t, extra)
        age = time.time() - t.get("created_at", time.time())
        print(f"  {name:50s}  queued {_fmt_time(age)}")
    print()


def _parse_duration(s):
    s = str(s).strip().lower()
    unit = 1
    if s.endswith("h"):
        unit = 3600
        s = s[:-1]
    elif s.endswith("m"):
        unit = 60
        s = s[:-1]
    elif s.endswith("s"):
        unit = 1
        s = s[:-1]
    try:
        return float(s) * unit
    except ValueError:
        return None


def _parse_gpu_list(ack_message):
    if not ack_message:
        return None
    m = re.search(r"gpus=\[([^\]]*)\]|gpus=all", ack_message)
    if not m:
        return None
    inner = m.group(1)
    if inner is None:
        return "all"
    return [int(x) for x in inner.split(",") if x.strip().isdigit()]


def _task_expname(task, extra=None):
    env = task.get("env", {})
    name = env.get("CTM_EXPERIMENT_NAME", "") if isinstance(env, dict) else ""
    if name:
        return name
    exp = extra or task.get("extra_args", "")
    for kw in ["--experiment_name", "--experiment-name"]:
        idx = exp.find(kw)
        if idx >= 0:
            rest = exp[idx + len(kw) :].strip()
            nrest = rest.split(None, 1)
            if nrest:
                return nrest[0]
    return task["task_id"]


def _fmt_time(seconds):
    if seconds < 0 or not seconds or seconds != seconds:
        return "    -"
    if seconds < 60:
        return f"{seconds:5.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:5.1f}m"
    return f"{seconds / 3600:5.1f}h"


def _fmt_progress_bar(ratio, width=20):
    if ratio < 0 or ratio != ratio:
        return "[" + " " * width + "]"
    filled = min(width, int(ratio * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _read_experiment_metrics(metrics_dir, experiment_name):
    path = os.path.join(metrics_dir, f"{experiment_name}.csv")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return rows[0] if rows else None
        return rows[-1]
    except Exception:
        return None


def _read_failure(metrics_dir, experiment_name):
    path = os.path.join(metrics_dir, f"{experiment_name}.fail.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def run_kanban(args):
    import csv as csv_mod
    global csv
    csv = csv_mod

    base = f"http://{args.master_addr}:{args.port}"
    metrics_dir = args.metrics_dir
    refresh = args.refresh
    width = args.width

    while True:
        try:
            status = get_json(f"{base}/status", timeout=5)
        except Exception as exc:
            if args.once:
                print(f"  Cannot reach pool server: {exc}")
                break
            print(f"\r  Waiting for pool server... {exc}    ", end="", flush=True)
            time.sleep(refresh)
            continue

        os.system("clear" if os.name != "nt" else "cls")

        nodes = status.get("nodes", {})
        tasks = status.get("tasks", [])
        acks = status.get("acks", {})
        now = time.time()

        sep = "=" * min(width, 72)
        thin = "-" * min(width, 72)

        lines = []
        lines.append(sep)
        lines.append(f"  CTM-LLM POOL KANBAN  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(sep)

        total_gpus = 0
        idle_gpus = 0
        node_lines = []

        # Build GPU→task mapping from acks
        gpu_task = {}
        for t in tasks:
            tid = t["task_id"]
            e = _task_expname(t)
            for addr, ack in acks.get(tid, {}).items():
                gpu_list = _parse_gpu_list(ack.get("message", ""))
                if gpu_list == "all":
                    gpu_task.setdefault(addr, {})["all"] = e
                elif gpu_list:
                    for g in gpu_list:
                        gpu_task.setdefault(addr, {})[str(g)] = e

        for addr, node in sorted(nodes.items()):
            host = node.get("hostname", "?")
            gpu_sum = node.get("gpu_summary", f"{node.get('gpus', '?')} GPU(s)")
            node_gpu_map = gpu_task.get(addr, {})

            total_gpus += node.get("gpus", 0)
            lines.append(f"  {addr}  {host}  {gpu_sum}")

            for gi in range(node.get("gpus", 0)):
                task_name = node_gpu_map.get(str(gi), "-")
                if task_name == "-":
                    idle_gpus += 1
                    lines.append(f"    GPU {gi:<2d}  idle")
                else:
                    lines.append(f"    GPU {gi:<2d}  {task_name}")

        n_nodes = len(nodes)
        lines.append(f"  NODES: {n_nodes}  TOTAL GPUS: {total_gpus}  IDLE GPUS: {idle_gpus}")

        lines.append(sep)
        lines.append("  TASK QUEUE")
        lines.append(thin)

        header = f"  {'#':>3s}  {'ID':22s}  {'STATUS':10s}  {'PROGRESS':24s}  {'ELAPSED':>7s}  {'ETA':>7s}  {'INFO'}"
        lines.append(header)

        active_tasks = []
        pending_tasks = []
        finished_tasks = []
        for t in tasks:
            st = t.get("status", "pending")
            if st in ("completed", "failed", "cancelled"):
                finished_tasks.append(t)
            elif st == "running":
                active_tasks.append(t)
            else:
                pending_tasks.append(t)

        row_idx = 0

        for t in active_tasks:
            row_idx += 1
            tid = t["task_id"]
            extra = t.get("extra_args", "")
            created = t.get("created_at", now)
            elapsed = now - created

            exp_name = _task_expname(t, extra)

            progress = t.get("progress", {})
            max_steps = progress.get("max_steps", 0)
            cur_step = progress.get("cur_step", 0)
            step_rate = progress.get("rate", 0)
            loss_val = progress.get("loss", "")

            ratio = cur_step / max_steps if max_steps > 0 else -1
            bar = _fmt_progress_bar(ratio)
            step_str = f"{int(cur_step):>6}/{int(max_steps):<6}" if max_steps > 0 else "   -/   - "

            if max_steps > 0 and cur_step > 0:
                if step_rate and step_rate > 0:
                    remaining = (max_steps - cur_step) / step_rate
                elif elapsed > 0:
                    avg_rate = cur_step / elapsed
                    remaining = (max_steps - cur_step) / avg_rate if avg_rate > 0 else -1
                else:
                    remaining = -1
            else:
                remaining = -1

            elapsed_s = _fmt_time(elapsed)
            eta_s = _fmt_time(remaining)

            info_parts = []
            if exp_name:
                info_parts.append(exp_name)
            if step_rate and step_rate > 0:
                info_parts.append(f"{step_rate:.1f}it/s")
            if loss_val and loss_val not in ("", "nan", "inf"):
                info_parts.append(f"loss={float(loss_val):.4f}")
            lines.append(
                f"  {row_idx:3d}  {tid:22s}  {'running':10s}  {bar} {step_str}  {elapsed_s:>7s}  {eta_s:>7s}  {' '.join(info_parts)}"
            )

        for t in pending_tasks:
            row_idx += 1
            tid = t["task_id"]
            created = t.get("created_at", now)
            elapsed = now - created
            extra = t.get("extra_args", "")
            short_extra = extra[:40] + "..." if len(extra) > 40 else extra
            lines.append(
                f"  {row_idx:3d}  {tid:22s}  {'pending':10s}  "
                f"{'[                      ]':24s}  {_fmt_time(elapsed):>7s}  {'    -':>7s}  {short_extra}"
            )

        if not active_tasks and not pending_tasks:
            lines.append("  (no active or pending tasks)")

        lines.append(thin)
        lines.append("  RECENTLY FINISHED (last 10)")
        lines.append(thin)

        recent = sorted(finished_tasks, key=lambda t: t.get("status_changed_at", 0), reverse=True)[:10]
        for t in recent:
            tid = t["task_id"]
            st = t.get("status", "?")
            rc = t.get("return_code")
            extra = t.get("extra_args", "")
            changed = t.get("status_changed_at", t.get("created_at", now))
            duration = changed - t.get("created_at", changed)

            extra_label = extra[:50] + "..." if len(extra) > 50 else extra

            exp_name = _task_expname(t, extra)

            fail_info = None
            if st == "failed" and exp_name:
                fail_info = _read_failure(metrics_dir, exp_name)

            status_label = st.upper()
            info_parts = []
            if st == "completed":
                progress = t.get("progress", {})
                lv = progress.get("loss", "")
                if lv and lv not in ("", "nan", "inf"):
                    info_parts.append(f"loss={float(lv):.4f}")
                elif exp_name:
                    metrics = _read_experiment_metrics(metrics_dir, exp_name)
                    if metrics:
                        lv = metrics.get("loss", "")
                        if lv and lv not in ("", "nan", "inf"):
                            info_parts.append(f"loss={float(lv):.4f}")
                if exp_name:
                    info_parts.append(exp_name)
            elif st == "failed":
                if fail_info:
                    err_type = fail_info.get("error_type", "")
                    info_parts.append(err_type)
                elif rc is not None:
                    info_parts.append(f"rc={rc}")

            info_str = " ".join(info_parts) if info_parts else extra_label
            lines.append(
                f"  {tid:22s}  {status_label:10s}  {_fmt_time(duration):>7s}  {info_str}"
            )

        if not recent:
            lines.append("  (no finished tasks)")

        lines.append(sep)
        print("\n".join(lines), flush=True)

        if args.once:
            break
        try:
            time.sleep(refresh)
        except KeyboardInterrupt:
            print("\n(kanban stopped)")
            break


def main():
    parser = argparse.ArgumentParser(description="CTM-LLM lightweight cluster pool")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=run_server)

    p = sub.add_parser("worker")
    p.add_argument("--config", default=None,
                   help="Cluster env file (optional, for git proxy/NCCL settings)")
    p.add_argument("--master_addr", default=__import__("_pool_config").MASTER_ADDR)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--node_addr", default=None)
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--gpu-slots", type=int, default=0,
                   help="Concurrent tasks per GPU (0=auto, 1=exclusive, N=custom)")
    p.add_argument("--no_auto_pull", action="store_false", dest="auto_pull")
    p.add_argument("--no_restart_on_update", action="store_false", dest="restart_on_update")
    p.set_defaults(auto_pull=True, restart_on_update=True)
    p.set_defaults(func=run_worker)

    p = sub.add_parser("submit")
    p.add_argument("--config", default="",
                   help="Task config env file (optional, e.g. infra/envs/h100_baseline.env)")
    p.add_argument("--master_addr", default=__import__("_pool_config").MASTER_ADDR)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--wait", type=float, default=30.0)
    p.add_argument("--nodes", nargs="+", default=None,
                   help="Restrict this task to a comma/space separated node subset.")
    p.add_argument("extra_args", nargs=argparse.REMAINDER)
    p.set_defaults(func=run_submit)

    p = sub.add_parser("status")
    p.add_argument("--master_addr", default=__import__("_pool_config").MASTER_ADDR)
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=run_status)

    p = sub.add_parser("task")
    p.add_argument("task_cmd", choices=["list", "cancel", "cancel-pending", "clear", "info", "history", "pending", "clean-fail"])
    p.add_argument("--task_id", default=None)
    p.add_argument("--duration", default="1h",
                   help="Time window for history: e.g. 1h, 2h, 4h, 8h, 16h")
    p.add_argument("--metrics_dir", default="runs/metrics",
                   help="Metrics directory for reading accuracy/loss")
    p.add_argument("--master_addr", default=__import__("_pool_config").MASTER_ADDR)
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=run_task)

    p = sub.add_parser("kanban")
    p.add_argument("--master_addr", default=__import__("_pool_config").MASTER_ADDR)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--metrics_dir", default="runs/metrics")
    p.add_argument("--refresh", type=float, default=5.0, help="Refresh interval in seconds")
    p.add_argument("--width", type=int, default=100, help="Display width")
    p.add_argument("--once", action="store_true", help="Print once and exit (no loop)")
    p.set_defaults(func=run_kanban)

    args, unknown = parser.parse_known_args()
    if args.cmd == "submit" and unknown:
        args.extra_args.extend(unknown)
    elif unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
