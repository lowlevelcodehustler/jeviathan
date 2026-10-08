#!/usr/bin/env python3
"""Jeviathan service manager — the daemon layer for both machines.

Services (ports are fixed by convention):
  shim   scripts/transformers_server.py   laptop NF4 Llama      :8200
  api    uvicorn jeviathan.main:app       System One API        :8100
  vllm   python -m vllm serve <model>     5090-box model server :8001

Cross-platform (Windows + Linux), stdlib only. PID files and logs live in the
repo root, keeping the legacy names: shim -> model.pid/model.log, api ->
api.pid/api.log, vllm -> vllm.pid/vllm.log. Logs are appended with a timestamp
header on each start so restarts stay traceable.

Examples:
  python scripts/manage.py status
  python scripts/manage.py start all --profile laptop-4050     # laptop stack
  python scripts/manage.py start vllm api --profile rtx5090    # 5090 box
  python scripts/manage.py restart api
  python scripts/manage.py stop shim
  python scripts/manage.py logs shim -n 40
  python scripts/manage.py fit --data evals/trinigard_eval.jsonl \
      --out calibration/calib-laptop-demo.json --profile laptop-4050-logprob
  python scripts/manage.py smoke rtx5090

Notes:
  * `start api` auto-points JEVIATHAN_BASE_URL/JEVIATHAN_MODEL at the local
    model server (shim :8200 or vllm :8001) unless you set them yourself.
  * `start shim` resolves the weights dir from --model-dir, then
    $JEVIATHAN_MODEL_DIR, then .jeviathan_model_dir at repo root (gitignored).
  * Model loads are slow on this hardware: shim ~40 s-6 min, vllm minutes to
    tens of minutes on first run (weight download). `start` waits and polls;
    use --wait 0 to detach immediately.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IS_WIN = os.name == "nt"
PY = sys.executable  # children run under the same interpreter (venv-aware)

VLLM_MODEL_DEFAULT = "Inferact/Qwen3.8-27B-NVFP4"
VLLM_ALIAS = "jeviathan-qwen3.8-27b"


def _reject_control_chars(p: str) -> None:
    """Fail fast if a resolved path carries control characters."""
    bad = sorted({c for c in p if ord(c) < 0x20 or ord(c) == 0x7F})
    if bad:
        raise SystemExit(
            f"model dir path contains control characters {bad!r}; refusing to use it "
            "(check --model-dir, $JEVIATHAN_MODEL_DIR and .jeviathan_model_dir)"
        )


def resolve_shim_model_dir(cli_value: str | None = None) -> str:
    """Local weights dir for the NF4 shim (no machine-specific defaults).

    Resolution order: --model-dir arg > $JEVIATHAN_MODEL_DIR env var >
    .jeviathan_model_dir file at repo root (gitignored, one line).
    Fails if the resolved path contains control characters.
    """
    p = cli_value or os.environ.get("JEVIATHAN_MODEL_DIR")
    if not p:
        local = REPO / ".jeviathan_model_dir"
        if local.is_file():
            p = local.read_text(encoding="utf-8").strip()
    if not p:
        raise SystemExit(
            "No model dir for the NF4 shim. Pass --model-dir, set $JEVIATHAN_MODEL_DIR, "
            f"or write your weights path to {REPO / '.jeviathan_model_dir'} (gitignored)."
        )
    _reject_control_chars(p)
    return p

# name -> (port, pidfile, logfile, health_path, default wait seconds)
SERVICES: dict[str, tuple[int, str, str, str, int]] = {
    "shim": (8200, "model.pid", "model.log", "/health", 900),
    "api": (8100, "api.pid", "api.log", "/health", 60),
    "vllm": (8001, "vllm.pid", "vllm.log", "/v1/models", 2400),
}


# --------------------------------------------------------------------------- helpers

def _log_path(name: str) -> Path:
    return REPO / SERVICES[name][2]


def _pid_path(name: str) -> Path:
    return REPO / SERVICES[name][1]


def read_pid(name: str) -> int | None:
    p = _pid_path(name)
    if not p.exists():
        return None
    try:
        pid = int(p.read_text(encoding="utf-8").strip() or 0)
        return pid or None
    except ValueError:
        return None


def write_pid(name: str, pid: int) -> None:
    _pid_path(name).write_text(str(pid), encoding="utf-8")


def clear_pid(name: str) -> None:
    p = _pid_path(name)
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass


def pid_alive(pid: int) -> bool:
    """True if a process with this PID is alive and openable.

    Windows note: os.kill(pid, 0) is NOT a reliable liveness probe here —
    CPython's kill() opens the target with PROCESS_ALL_ACCESS and raises
    OSError (e.g. errno 87/5) for processes it cannot fully open (elevated
    services, other sessions), which would make stop/restart/start-wait
    misreport live servers as dead. OpenProcess with
    PROCESS_QUERY_LIMITED_INFORMATION succeeds for any same-user process and
    is the standard "is this pid alive" check (what psutil does).
    """
    if IS_WIN:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        k32.CloseHandle(h)
        return True
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def health(name: str) -> tuple[bool, dict | None]:
    """(up, parsed-json-or-None). Shim requires model_loaded=true."""
    port, _, _, path, _ = SERVICES[name]
    url = f"http://localhost:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8")) if resp.status == 200 else None
    except Exception:
        return False, None
    if name == "shim" and isinstance(data, dict):
        ok = bool(data.get("model_loaded"))
        return ok, data
    return data is not None, data


def port_owner(port: int) -> int | None:
    """Best-effort PID of the process listening on `port` (diagnostics)."""
    try:
        if IS_WIN:
            out = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, timeout=10
            ).stdout
            for line in out.splitlines():
                parts = line.split()
                if (
                    len(parts) >= 5
                    and parts[3] == "LISTENING"
                    and f":{port}" in parts[1].split(")")[-1]
                ):
                    return int(parts[4])
        else:
            out = subprocess.run(
                ["ss", "-ltnp"], capture_output=True, text=True, timeout=10
            ).stdout
            m = re.search(rf":{port}\b.*?pid=(\d+)", out)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return None


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _open_log(name: str):
    path = _log_path(name)
    fh = open(path, "a", encoding="utf-8")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    fh.write(f"\n===== {stamp} manage.py start {name} =====\n")
    fh.flush()
    return fh


def _spawn(name: str, cmd: list[str], env: dict | None) -> int:
    fh = _open_log(name)
    kwargs: dict = dict(
        cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
    )
    if IS_WIN:
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, env=env or os.environ.copy(), **kwargs)
    write_pid(name, proc.pid)
    return proc.pid


def _kill_tree(pid: int) -> None:
    if IS_WIN:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, timeout=30)
    else:
        try:
            import signal

            os.killpg(os.getpgid(pid), 15)  # SIGTERM to the session we started
            for _ in range(10):
                if not pid_alive(pid):
                    return
                time.sleep(0.5)
            os.killpg(os.getpgid(pid), 9)
        except (OSError, ProcessLookupError):
            pass


def build_cmd(name: str, args) -> tuple[list[str], dict | None]:
    """Command + env for a service."""
    if name == "shim":
        cmd = [
            PY, "scripts/transformers_server.py",
            "--model-dir", resolve_shim_model_dir(args.model_dir),
            "--port", str(SERVICES["shim"][0]),
            "--served-name", "llama3.1-8b-local",
        ]
        return cmd, None
    if name == "api":
        port = SERVICES["api"][0]
        cmd = [PY, "-m", "uvicorn", "jeviathan.main:app", "--port", str(port)]
        env = os.environ.copy()
        profile = args.profile or env.get("JEVIATHAN_PROFILE") or "laptop-4050"
        env["JEVIATHAN_PROFILE"] = profile
        # Point the API at whichever local model server we manage, unless the
        # user already said otherwise.
        if not env.get("JEVIATHAN_BASE_URL"):
            if name == "api" and (port_open(8200) or args.model_dir):
                env["JEVIATHAN_BASE_URL"] = f"http://localhost:{SERVICES['shim'][0]}/v1"
                env.setdefault("JEVIATHAN_MODEL", "llama3.1-8b-local")
            elif port_open(8001):
                env["JEVIATHAN_BASE_URL"] = f"http://localhost:{SERVICES['vllm'][0]}/v1"
                env.setdefault("JEVIATHAN_MODEL", VLLM_ALIAS)
        return cmd, env
    # vllm
    model = os.environ.get("JEVIATHAN_5090_MODEL") or args.vllm_model or VLLM_MODEL_DEFAULT
    port = SERVICES["vllm"][0]
    cmd = [
        PY, "-m", "vllm", "serve", model,
        "--served-model-name", VLLM_ALIAS,
        "--tensor-parallel-size", "1",
        "--max-model-len", "32768",
        "--kv-cache-dtype", "fp8",
        "--gpu-memory-utilization", "0.92",
        "--port", str(port),
    ]
    return cmd, None


# --------------------------------------------------------------------------- commands

def cmd_status(args) -> int:
    print(f"Jeviathan status @ {REPO}")
    exit_code = 0
    for name in ("shim", "api", "vllm"):
        port, _, logname, _, _ = SERVICES[name]
        up, data = health(name)
        pid = read_pid(name)
        if up and pid and not pid_alive(pid):
            # Self-heal: Windows venv launchers can leave a wrapper PID in the
            # file while the real server runs as its child. Adopt port owner.
            owner = port_owner(port)
            if owner:
                write_pid(name, owner)
                pid = owner
        if up:
            extra = ""
            if name == "api" and isinstance(data, dict):
                extra = f"  profile={data.get('profile')} model={data.get('model')}"
            elif name == "shim":
                extra = "  model_loaded=true"
            state = f"UP (pid {pid or '?'}){extra}"
        else:
            owner = port_owner(port) if port_open(port) else None
            if owner and owner != pid:
                state = f"port busy (foreign pid {owner}, our pid file: {pid or 'none'})"
            elif pid and not pid_alive(pid):
                state = "stale pid file (process dead)"
                clear_pid(name)
            else:
                state = "down"
            if args.strict:
                exit_code = 1
        print(f"  {name:5s} :{port}  {state:<48s} log: {logname}")
    return exit_code


def cmd_start(args) -> int:
    targets = ["shim", "api"] if args.target == "all" else [args.target]
    rc = 0
    for name in targets:
        up, _ = health(name)
        if up:
            print(f"[{name}] already running on :{SERVICES[name][0]}")
            continue
        cmd, env = build_cmd(name, args)
        print(f"[{name}] starting: {' '.join(cmd[:6])} ...")
        pid = _spawn(name, cmd, env)
        wait_s = 0 if args.wait == 0 else (args.wait or SERVICES[name][4])
        t0 = time.time()
        while time.time() - t0 < wait_s:
            up, _ = health(name)
            if up:
                break
            if not pid_alive(pid):
                print(f"[{name}] process died during startup — see {_log_path(name)}")
                rc = 1
                break
            time.sleep(5 if name != "api" else 2)
        up, _ = health(name)
        if up:
            real = port_owner(SERVICES[name][0])
            if real and real != pid:  # Windows wrapper PID -> adopt the server
                write_pid(name, real)
                print(f"[{name}] UP on :{SERVICES[name][0]} (pid {real}, "
                      f"wrapper was {pid}, {time.time()-t0:.0f}s)")
            else:
                print(f"[{name}] UP on :{SERVICES[name][0]} (pid {pid}, {time.time()-t0:.0f}s)")
        elif args.wait == 0:
            print(f"[{name}] detached (pid {pid}); poll with `manage.py status`")
        else:
            print(
                f"[{name}] not healthy after {wait_s}s — still loading? "
                f"check `{PY} scripts/manage.py logs {name}` ({_log_path(name)})"
            )
            rc = 1
    return rc


def cmd_stop(args) -> int:
    targets = ["shim", "api"] if args.target == "all" else [args.target]
    rc = 0
    for name in targets:
        pid = read_pid(name)
        up, _ = health(name)
        if pid and pid_alive(pid):
            print(f"[{name}] killing pid {pid} (tree)")
            _kill_tree(pid)
            time.sleep(2)
            clear_pid(name)
        elif up:
            owner = port_owner(SERVICES[name][0])
            if owner and args.force:
                print(f"[{name}] no usable pid file; killing port owner {owner}")
                _kill_tree(owner)
                time.sleep(2)
                clear_pid(name)
            else:
                print(f"[{name}] running but pid unknown (port owner "
                      f"{owner or '?'}); re-run with --force to kill it")
                rc = 1
        elif port_open(SERVICES[name][0]):
            print(f"[{name}] port :{SERVICES[name][0]} busy by foreign process — "
                  f"see `manage.py status`")
            rc = 1
        else:
            clear_pid(name)
            print(f"[{name}] not running")
    return rc


def cmd_restart(args) -> int:
    args.force = True
    return cmd_stop(args) or cmd_start(args)


def cmd_logs(args) -> int:
    path = _log_path(args.service)
    if not path.exists():
        print(f"no log file yet: {path}")
        return 1
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-args.lines:]:
        print(line)
    return 0


def cmd_fit(args) -> int:
    env = os.environ.copy()
    if args.profile:
        env["JEVIATHAN_PROFILE"] = args.profile
    cmd = [PY, "-m", "jeviathan.calibration.cli", "fit"]
    if args.from_raw:
        cmd += ["--from-raw", args.from_raw]
    else:
        cmd += ["--data", args.data or "evals/trinigard_eval.jsonl"]
        if args.resume:
            cmd.append("--resume")
    if args.raw_out:
        cmd += ["--raw-out", args.raw_out]
    cmd += ["--out", args.out or "calibration/calib.json"]
    print(f"fitting: {' '.join(cmd)}")
    if args.background:
        log = REPO / "calib_fit.log"
        fh = open(log, "a", encoding="utf-8")
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} manage.py fit =====\n")
        fh.flush()
        kwargs: dict = dict(cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL)
        if IS_WIN:
            kwargs["creationflags"] = (subprocess.DETACHED_PROCESS
                                       | subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(cmd, env=env, **kwargs)
        print(f"fit running in background (pid {proc.pid}) -> {log}")
        return 0
    return subprocess.call(cmd, cwd=REPO, env=env)


def cmd_smoke(args) -> int:
    profile = args.profile or "rtx5090"
    print(f"logprob smoke test against profile {profile!r} ...")
    return subprocess.call([PY, str(REPO / "scripts" / "smoke_logprob.py"), profile], cwd=REPO)


# --------------------------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_target(sp, default):
        sp.add_argument("target", nargs="?", default=default,
                        choices=["all", "shim", "api", "vllm"])

    sp = sub.add_parser("status", help="Health-check all services")
    sp.add_argument("--strict", action="store_true",
                    help="Exit 1 if any service is down")
    sp.set_defaults(fn=cmd_status)

    for name, fn in (("start", cmd_start), ("stop", cmd_stop), ("restart", cmd_restart)):
        sp = sub.add_parser(name, help=f"{name} a service (or 'all' = shim+api)")
        add_target(sp, "all")
        if name != "stop":
            sp.add_argument("--profile", default=None,
                            help="JEVIATHAN_PROFILE for the api (default: laptop-4050)")
            sp.add_argument("--model-dir", default=None,
                            help="shim weights dir ($JEVIATHAN_MODEL_DIR or "
                                 ".jeviathan_model_dir if omitted)")
            sp.add_argument("--vllm-model", default=None,
                            help=f"vLLM model id (default: {VLLM_MODEL_DEFAULT})")
        else:
            sp.add_argument("--force", action="store_true",
                            help="Kill the port owner even without a usable pid file")
        if name in ("start", "restart"):
            sp.add_argument("--wait", type=int, default=None,
                            help="Seconds to wait for health (0 = detach immediately)")
        sp.set_defaults(fn=fn)

    sp = sub.add_parser("logs", help="Tail a service log")
    sp.add_argument("service", choices=["shim", "api", "vllm"])
    sp.add_argument("-n", "--lines", type=int, default=40)
    sp.set_defaults(fn=cmd_logs)

    sp = sub.add_parser("fit", help="Run the calibration fit (RLCD-lite)")
    sp.add_argument("--data", default=None, help="Eval JSONL (default: evals/trinigard_eval.jsonl)")
    sp.add_argument("--out", default=None, help="Artifact path (default: calibration/calib.json)")
    sp.add_argument("--profile", default=None, help="JEVIATHAN_PROFILE for the fit")
    sp.add_argument("--from-raw", default=None, help="Refit from a raw dump instead of model calls")
    sp.add_argument("--raw-out", default=None, help="Raw response dump path (append)")
    sp.add_argument("--resume", action="store_true", help="Skip rows already in the raw dump")
    sp.add_argument("--background", action="store_true", help="Detach; progress -> calib_fit.log")
    sp.set_defaults(fn=cmd_fit)

    sp = sub.add_parser("smoke", help="End-to-end logprob smoke test (scripts/smoke_logprob.py)")
    sp.add_argument("profile", nargs="?", default=None, help="Profile name (default: rtx5090)")
    sp.set_defaults(fn=cmd_smoke)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
