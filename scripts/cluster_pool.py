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


URL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

STATE = {
    "nodes": {},
    "task": None,
    "acks": {},
}
LOCK = threading.Lock()


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
        acks = dict(STATE["acks"])

    print("\n=== CTM Pool ===", flush=True)
    for addr, node in sorted(nodes.items()):
        age = now - node["last_seen"]
        gpu_summary = node.get("gpu_summary") or f"{node.get('gpus', '?')} GPU(s)"
        print(
            f"  {addr:15s} rank={node.get('rank', '?')} "
            f"host={node.get('hostname', '?')} status={node.get('status', '?')} "
            f"gpus={gpu_summary} seen={age:.1f}s",
            flush=True,
        )
    if task:
        print(f"  task={task['task_id']} args={task.get('extra_args', '')}", flush=True)
        for addr, ack in sorted(acks.items()):
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
                    "acks": STATE["acks"],
                }
            self._write_json(payload)
            return
        if parsed.path == "/task":
            query = urllib.parse.parse_qs(parsed.query)
            addr = query.get("node_addr", [""])[0]
            with LOCK:
                task = STATE["task"]
                acked = task is not None and STATE["acks"].get(addr, {}).get("task_id") == task["task_id"]
            self._write_json({"task": None if acked else task})
            return
        self.send_error(404)

    def do_POST(self):
        if self.path == "/heartbeat":
            payload = self._read_json()
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
            return

        if self.path == "/submit":
            payload = self._read_json()
            task_id = time.strftime("%Y%m%d_%H%M%S")
            task = {
                "task_id": task_id,
                "config": payload["config"],
                "extra_args": payload.get("extra_args", ""),
                "created_at": time.time(),
            }
            with LOCK:
                STATE["task"] = task
                STATE["acks"] = {}
            print(f"[pool] new task: {task_id} {task['extra_args']}", flush=True)
            print_pool()
            self._write_json({"ok": True, "task": task})
            return

        if self.path == "/ack":
            payload = self._read_json()
            addr = payload["node_addr"]
            with LOCK:
                STATE["acks"][addr] = payload
            print(f"[pool] ack from {addr}: {payload.get('status')} {payload.get('message', '')}", flush=True)
            print_pool()
            self._write_json({"ok": True})
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


def git_pull_ff_only(config):
    before = git_head()
    env = os.environ.copy()
    if config.get("GIT_HTTP_PROXY"):
        env["http_proxy"] = config["GIT_HTTP_PROXY"]
        env["HTTP_PROXY"] = config["GIT_HTTP_PROXY"]
    if config.get("GIT_HTTPS_PROXY"):
        env["https_proxy"] = config["GIT_HTTPS_PROXY"]
        env["HTTPS_PROXY"] = config["GIT_HTTPS_PROXY"]
    proc = subprocess.run(
        ["git", "pull", "--ff-only"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    after = git_head()
    return {
        "ok": proc.returncode == 0,
        "before": before,
        "after": after,
        "output": proc.stdout.strip(),
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
    proc = None
    last_task_id = None

    print(f"CTM worker online: addr={node_addr} rank={rank} host={hostname}", flush=True)
    print(gpu_summary, flush=True)
    print(format_gpu_lines(gpus), flush=True)
    print(f"Polling pool server: {base}", flush=True)

    while True:
        if proc is not None and proc.poll() is not None:
            status = f"exited:{proc.returncode}"
            proc = None

        heartbeat = {
            "node_addr": node_addr,
            "rank": rank,
            "hostname": hostname,
            "status": status,
            "gpus": len(gpus),
            "gpu_summary": gpu_summary,
            "gpu_devices": gpus,
            "pid": proc.pid if proc else None,
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

        if task and task["task_id"] != last_task_id:
            if proc is not None and proc.poll() is None:
                msg = f"busy pid={proc.pid}, ignore task"
                print(f"[worker] task {task['task_id']} ignored: {msg}", flush=True)
                post_json(f"{base}/ack", {
                    "node_addr": node_addr,
                    "task_id": task["task_id"],
                    "status": "busy",
                    "message": msg,
                })
            else:
                if args.auto_pull:
                    pull = git_pull_ff_only(config)
                    if pull["ok"]:
                        print(
                            f"[worker] git pull ok: {pull['before']} -> {pull['after']}",
                            flush=True,
                        )
                        if pull["before"] != pull["after"] and args.restart_on_update:
                            restart_worker_process()
                    else:
                        msg = f"git pull failed: {pull['output']}"
                        print(f"[worker] task {task['task_id']} rejected: {msg}", flush=True)
                        post_json(f"{base}/ack", {
                            "node_addr": node_addr,
                            "task_id": task["task_id"],
                            "status": "pull_failed",
                            "message": msg,
                        })
                        last_task_id = task["task_id"]
                        time.sleep(args.interval)
                        continue

                extra = shlex.split(task.get("extra_args", ""))
                cmd = ["bash", "scripts/train_cluster.sh", "--config", task["config"], *extra]
                env = os.environ.copy()
                env["CTM_NODE_ADDR"] = node_addr
                print(f"[worker] received task {task['task_id']}: {' '.join(shlex.quote(x) for x in cmd)}", flush=True)
                proc = subprocess.Popen(cmd, env=env)
                status = f"running:{task['task_id']}"
                last_task_id = task["task_id"]
                post_json(f"{base}/ack", {
                    "node_addr": node_addr,
                    "task_id": task["task_id"],
                    "status": "started",
                    "message": f"pid={proc.pid}",
                })

        time.sleep(args.interval)


def run_submit(args):
    extra_items = list(args.extra_args)
    if extra_items and extra_items[0] == "--":
        extra_items = extra_items[1:]
    extra_args = " ".join(shlex.quote(item) for item in extra_items)
    payload = {"config": args.config, "extra_args": extra_args}
    resp = post_json(f"http://{args.master_addr}:{args.port}/submit", payload)
    task = resp["task"]
    print(f"submitted task {task['task_id']}: {task.get('extra_args', '')}")

    if args.wait <= 0:
        return

    config = load_cluster_config(args.config)
    expected = set(config.get("NODE_ADDRS", []))
    deadline = time.time() + args.wait
    seen = set()
    while time.time() < deadline:
        status = get_json(f"http://{args.master_addr}:{args.port}/status")
        acks = status.get("acks", {})
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
    p.add_argument("extra_args", nargs=argparse.REMAINDER)
    p.set_defaults(func=run_submit)

    p = sub.add_parser("status")
    p.add_argument("--master_addr", default="11.131.210.78")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=run_status)

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
