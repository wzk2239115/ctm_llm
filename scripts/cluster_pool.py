#!/usr/bin/env python3
import argparse
import json
import os
import shlex
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
}
LOCK = threading.Lock()

FINAL_STATUSES = {"completed", "failed", "cancelled"}


def task_set_status(task, status):
    task["status"] = status
    task["status_changed_at"] = time.time()


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
    busy = node.get("busy_gpus") or []
    if requested is None:
        return not busy and not node.get("running_tasks")
    return not set(requested) & set(busy)


def load_cluster_config(path):
    data = {}
    node_addrs = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)

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
        with LOCK:
            old = STATE["nodes"].get(addr)
            announce = old is None or old.get("status") != payload.get("status")
            payload["last_seen"] = time.time()
            STATE["nodes"][addr] = payload
        if announce:
            print(f"[pool] node online/update: {addr} status={payload.get('status')}", flush=True)
            print_pool()
        self._write_json({"ok": True})

    def _handle_submit(self, payload):
        task_id = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        with LOCK:
            master_port = int(payload.get("master_port") or STATE.get("next_master_port", 20000))
            STATE["next_master_port"] = 20000 + ((master_port - 19999) % 30000)
        task = {
            "task_id": task_id,
            "config": payload["config"],
            "extra_args": payload.get("extra_args", ""),
            "node_addrs": payload.get("node_addrs") or [],
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
        print(f"[pool] new task: {task_id} status=pending nodes={nodes} {task['extra_args']}", flush=True)
        print_pool()
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
        print(f"[pool] ack from {addr}: {payload.get('status')} {payload.get('message', '')}", flush=True)
        print_pool()
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
        print(f"[pool] task {task_id} {status} (rc={rc}) reported by {addr}", flush=True)
        print_pool()
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
        if cancelled:
            print_pool()
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


def run_worker(args):
    config = load_cluster_config(args.config)
    node_addr = args.node_addr or detect_node_addr(config)
    rank = config.get("NODE_ADDRS", []).index(node_addr) if node_addr in config.get("NODE_ADDRS", []) else "?"
    base = f"http://{args.master_addr}:{args.port}"
    hostname = socket.gethostname()
    gpus = gpu_inventory()
    gpu_summary = summarize_gpus(gpus)
    status = "idle"
    procs = {}
    process_head = git_head()

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
            rc = item["proc"].returncode
            print(
                f"[worker] task {task_id} exited rc={rc} "
                f"gpus={item.get('gpus') or 'all'}",
                flush=True,
            )
            try:
                post_json(f"{base}/complete", {
                    "node_addr": node_addr,
                    "task_id": task_id,
                    "return_code": rc,
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
            if any(gpu_sets_overlap(requested_gpus, item.get("gpus")) for item in procs.values()):
                msg = f"busy gpus={busy_gpus}, requested={requested_gpus or 'all'}"
                print(f"[worker] task {task['task_id']} ignored: {msg}", flush=True)
                post_json(f"{base}/ack", {
                    "node_addr": node_addr,
                    "task_id": task["task_id"],
                    "status": "busy",
                    "message": msg,
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
                env["CTM_NODE_ADDR"] = node_addr
                env["CTM_POOL_MASTER_PORT"] = str(task.get("master_port") or 29500)
                if task.get("node_addrs"):
                    env["CTM_POOL_NODE_ADDRS"] = ",".join(task["node_addrs"])
                if requested_gpus is not None:
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in requested_gpus)
                    env["NPROC_PER_NODE"] = str(len(requested_gpus))
                print(f"[worker] received task {task['task_id']}: {' '.join(shlex.quote(x) for x in cmd)}", flush=True)
                proc = subprocess.Popen(cmd, env=env)
                procs[task["task_id"]] = {
                    "proc": proc,
                    "gpus": requested_gpus,
                }
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
    payload = {"config": args.config, "extra_args": extra_args, "node_addrs": node_addrs}
    resp = post_json(f"http://{args.master_addr}:{args.port}/submit", payload)
    task = resp["task"]
    print(f"submitted task {task['task_id']}: {task.get('extra_args', '')}")

    if args.wait <= 0:
        return

    config = load_cluster_config(args.config)
    expected = {parse_node_spec(spec)[0] for spec in (node_addrs or config.get("NODE_ADDRS", []))}
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
    if args.task_cmd == "list":
        status = get_json(f"{base}/status")
        tasks = status.get("tasks", [])
        if not tasks:
            print("no tasks")
            return
        now = time.time()
        fmt = "{:<22s} {:<12s} {:<8s} {:<6s} {}"
        print(fmt.format("TASK_ID", "STATUS", "AGE(s)", "RC", "ARGS"))
        for t in tasks:
            age = int(now - t.get("created_at", now))
            rc = str(t.get("return_code", "")) if t.get("return_code") is not None else ""
            extra = t.get("extra_args", "")
            if len(extra) > 50:
                extra = extra[:47] + "..."
            print(fmt.format(t["task_id"], t.get("status", "?"), str(age), rc, extra))
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
    else:
        print(f"unknown task command: {args.task_cmd}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="CTM-LLM lightweight cluster pool")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=run_server)

    p = sub.add_parser("worker")
    p.add_argument("--config", default="infra/clusters/h100_2nodes.env")
    p.add_argument("--master_addr", default="11.131.210.78")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--node_addr", default=None)
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--no_auto_pull", action="store_false", dest="auto_pull")
    p.add_argument("--no_restart_on_update", action="store_false", dest="restart_on_update")
    p.set_defaults(auto_pull=True, restart_on_update=True)
    p.set_defaults(func=run_worker)

    p = sub.add_parser("submit")
    p.add_argument("--config", default="infra/clusters/h100_2nodes.env")
    p.add_argument("--master_addr", default="11.131.210.78")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--wait", type=float, default=30.0)
    p.add_argument("--nodes", nargs="+", default=None,
                   help="Restrict this task to a comma/space separated node subset.")
    p.add_argument("extra_args", nargs=argparse.REMAINDER)
    p.set_defaults(func=run_submit)

    p = sub.add_parser("status")
    p.add_argument("--master_addr", default="11.131.210.78")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=run_status)

    p = sub.add_parser("task")
    p.add_argument("task_cmd", choices=["list", "cancel", "cancel-pending", "clear", "info"])
    p.add_argument("--task_id", default=None)
    p.add_argument("--master_addr", default="11.131.210.78")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=run_task)

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
