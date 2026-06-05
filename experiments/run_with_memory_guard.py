#!/usr/bin/env python3
"""
通用内存看门狗包装脚本。

用途：
1. 启动任意一条长时间运行的命令。
2. 轮询该命令所在进程组的总 RSS 内存。
3. 一旦超过阈值，先发送 SIGTERM，等待一段宽限时间。
4. 若宽限后仍未退出，再发送 SIGKILL，避免整机被 OOM 拖死。

设计约束：
- 只使用 Python 标准库，不额外依赖 psutil。
- 监控对象是“本次命令的整个进程组”，而不是单个 PID。
- 子进程 stdout/stderr 直接透传，方便沿用现有日志习惯。
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def build_arg_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(description="运行命令并为其加上 RSS 内存看门狗")
    parser.add_argument(
        "--rss-limit-gb",
        type=float,
        required=True,
        help="软阈值：进程组 RSS 总内存超过该值后开始计时",
    )
    parser.add_argument(
        "--rss-limit-duration-sec",
        type=float,
        default=30.0,
        help="软阈值持续超线时长，默认 30 秒；超过后触发终止",
    )
    parser.add_argument(
        "--rss-hard-limit-gb",
        type=float,
        default=None,
        help="硬阈值：RSS 一旦达到该值立即终止；默认关闭",
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=float,
        default=5.0,
        help="内存轮询间隔，默认 5 秒",
    )
    parser.add_argument(
        "--grace-sec",
        type=float,
        default=10.0,
        help="发送 SIGTERM 后的等待时间，默认 10 秒；超时再 SIGKILL",
    )
    parser.add_argument(
        "--heartbeat-sec",
        type=float,
        default=60.0,
        help="周期性状态日志间隔，默认 60 秒",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="子命令工作目录，默认继承当前目录",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="为子命令额外注入环境变量，格式 KEY=VALUE，可重复传入",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="看门狗日志文件路径，默认写入 logs/memory_guard_<时间>.log",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="要执行的命令；建议使用 '--' 之后传入",
    )
    return parser


class GuardLogger:
    """同时写 stdout 和文件的极简日志器。"""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.log_path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        """写一条带时间戳的日志，并立即 flush。"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line, flush=True)
        self._fp.write(line + "\n")
        self._fp.flush()

    def close(self) -> None:
        """关闭文件句柄。"""
        self._fp.close()


def _default_log_path() -> Path:
    """生成默认日志路径。"""
    project_root = Path(__file__).resolve().parent.parent
    log_dir = project_root / "logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"memory_guard_{timestamp}.log"


def _parse_env_overrides(env_items: List[str]) -> Dict[str, str]:
    """把 CLI 里的 KEY=VALUE 列表解析成字典。"""
    env_map: Dict[str, str] = {}
    for item in env_items:
        if "=" not in item:
            raise ValueError(f"--env 参数必须是 KEY=VALUE 格式，收到: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--env 的 KEY 不能为空，收到: {item}")
        env_map[key] = value
    return env_map


def _rss_kb_to_gb(rss_kb: int) -> float:
    """把 KB 转为 GB，便于日志展示。"""
    return rss_kb / 1024.0 / 1024.0


def _read_process_group_rss_kb(target_pgid: int) -> Tuple[int, List[Tuple[int, int]]]:
    """
    读取目标进程组的 RSS 总量。

    这里使用 `ps -ax -o pid=,pgid=,rss=`：
    - `pid`：进程号
    - `pgid`：进程组号
    - `rss`：常驻内存，单位 KB

    返回：
    - 总 RSS（KB）
    - 组内成员列表 [(pid, rss_kb), ...]
    """
    cmd = ["ps", "-ax", "-o", "pid=,pgid=,rss="]
    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    total_rss_kb = 0
    members: List[Tuple[int, int]] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            continue
        pid_str, pgid_str, rss_str = parts
        try:
            pid = int(pid_str)
            pgid = int(pgid_str)
            rss_kb = int(rss_str)
        except ValueError:
            continue
        if pgid != target_pgid:
            continue
        total_rss_kb += rss_kb
        members.append((pid, rss_kb))

    return total_rss_kb, members


def _signal_process_group(
    pgid: int,
    sig: signal.Signals,
    logger: GuardLogger,
    reason: str,
) -> None:
    """向整个进程组发信号；若进程组已不存在则静默跳过。"""
    try:
        os.killpg(pgid, sig)
        logger.log(f"{reason}：已向进程组 {pgid} 发送 {sig.name}")
    except ProcessLookupError:
        logger.log(f"{reason}：进程组 {pgid} 已不存在，跳过发送 {sig.name}")


def _terminate_process_group(
    pgid: int,
    proc: subprocess.Popen,
    grace_sec: float,
    logger: GuardLogger,
) -> None:
    """
    优先优雅终止，再在宽限期后强杀。

    - 先发 SIGTERM，给子进程清理和落盘机会。
    - 如果宽限期后主进程仍未退出，再发 SIGKILL。
    """
    _signal_process_group(pgid, signal.SIGTERM, logger, "达到内存阈值")
    deadline = time.time() + max(0.0, grace_sec)
    while time.time() < deadline:
        if proc.poll() is not None:
            logger.log(f"主进程已在 SIGTERM 后退出，returncode={proc.returncode}")
            return
        time.sleep(0.5)

    if proc.poll() is None:
        _signal_process_group(pgid, signal.SIGKILL, logger, "宽限期超时")
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.log("SIGKILL 后主进程仍未在 5 秒内退出，请人工检查")


def _normalize_command(raw_command: List[str]) -> List[str]:
    """去掉 argparse REMAINDER 里的前导 `--`。"""
    command = list(raw_command)
    if command and command[0] == "--":
        command = command[1:]
    return command


def main() -> int:
    """脚本主入口。"""
    parser = build_arg_parser()
    args = parser.parse_args()

    command = _normalize_command(args.command)
    if not command:
        parser.error("必须在 '--' 后提供要执行的命令")

    if args.rss_limit_gb <= 0:
        parser.error("--rss-limit-gb 必须大于 0")
    if args.rss_limit_duration_sec < 0:
        parser.error("--rss-limit-duration-sec 不能小于 0")
    if args.rss_hard_limit_gb is not None and args.rss_hard_limit_gb <= 0:
        parser.error("--rss-hard-limit-gb 必须大于 0")
    if args.rss_hard_limit_gb is not None and args.rss_hard_limit_gb < args.rss_limit_gb:
        parser.error("--rss-hard-limit-gb 不能小于 --rss-limit-gb")
    if args.poll_interval_sec <= 0:
        parser.error("--poll-interval-sec 必须大于 0")

    env = os.environ.copy()
    env.update(_parse_env_overrides(args.env))

    log_path = Path(args.log_file) if args.log_file else _default_log_path()
    logger = GuardLogger(log_path)
    rss_limit_kb = int(args.rss_limit_gb * 1024 * 1024)
    rss_hard_limit_kb = (
        int(args.rss_hard_limit_gb * 1024 * 1024)
        if args.rss_hard_limit_gb is not None
        else None
    )
    peak_rss_kb = 0
    last_heartbeat_ts = 0.0
    soft_breach_started_ts = None

    logger.log(
        "看门狗启动："
        f"soft_limit={args.rss_limit_gb:.2f} GB，"
        f"soft_duration={args.rss_limit_duration_sec:.1f}s，"
        f"hard_limit={f'{args.rss_hard_limit_gb:.2f} GB' if args.rss_hard_limit_gb is not None else 'disabled'}，"
        f"poll={args.poll_interval_sec:.1f}s，"
        f"grace={args.grace_sec:.1f}s"
    )
    logger.log(f"日志文件：{log_path}")
    logger.log(f"工作目录：{args.cwd or os.getcwd()}")
    logger.log(f"执行命令：{shlex.join(command)}")
    if args.env:
        logger.log(f"附加环境变量：{args.env}")

    proc = subprocess.Popen(
        command,
        cwd=args.cwd,
        env=env,
        start_new_session=True,
    )
    pgid = proc.pid
    logger.log(f"子进程已启动：pid={proc.pid}，pgid={pgid}")

    try:
        while True:
            returncode = proc.poll()
            rss_kb, members = _read_process_group_rss_kb(pgid)
            peak_rss_kb = max(peak_rss_kb, rss_kb)

            now_ts = time.time()
            if returncode is not None:
                logger.log(
                    "子命令正常结束："
                    f"returncode={returncode}，"
                    f"peak_rss={_rss_kb_to_gb(peak_rss_kb):.2f} GB"
                )
                return returncode

            if rss_hard_limit_kb is not None and rss_kb >= rss_hard_limit_kb:
                top_members = sorted(members, key=lambda x: x[1], reverse=True)[:5]
                top_desc = ", ".join(
                    f"pid={pid}:{_rss_kb_to_gb(mem_kb):.2f}GB"
                    for pid, mem_kb in top_members
                )
                logger.log(
                    "触发硬阈值："
                    f"current={_rss_kb_to_gb(rss_kb):.2f} GB，"
                    f"hard_limit={args.rss_hard_limit_gb:.2f} GB，"
                    f"members={len(members)}，top=[{top_desc}]"
                )
                _terminate_process_group(pgid, proc, args.grace_sec, logger)
                final_code = proc.poll()
                logger.log(
                    "看门狗结束目标命令："
                    f"returncode={final_code}，"
                    f"peak_rss={_rss_kb_to_gb(peak_rss_kb):.2f} GB"
                )
                return 137

            if rss_kb >= rss_limit_kb:
                if soft_breach_started_ts is None:
                    soft_breach_started_ts = now_ts
                    logger.log(
                        "进入软阈值区间："
                        f"current={_rss_kb_to_gb(rss_kb):.2f} GB，"
                        f"soft_limit={args.rss_limit_gb:.2f} GB，"
                        f"需连续维持 {args.rss_limit_duration_sec:.1f}s 才触发终止"
                    )
                breach_duration = now_ts - soft_breach_started_ts
                if breach_duration >= args.rss_limit_duration_sec:
                    top_members = sorted(members, key=lambda x: x[1], reverse=True)[:5]
                    top_desc = ", ".join(
                        f"pid={pid}:{_rss_kb_to_gb(mem_kb):.2f}GB"
                        for pid, mem_kb in top_members
                    )
                    logger.log(
                        "触发软阈值持续超线："
                        f"current={_rss_kb_to_gb(rss_kb):.2f} GB，"
                        f"soft_limit={args.rss_limit_gb:.2f} GB，"
                        f"duration={breach_duration:.1f}s，"
                        f"members={len(members)}，top=[{top_desc}]"
                    )
                    _terminate_process_group(pgid, proc, args.grace_sec, logger)
                    final_code = proc.poll()
                    logger.log(
                        "看门狗结束目标命令："
                        f"returncode={final_code}，"
                        f"peak_rss={_rss_kb_to_gb(peak_rss_kb):.2f} GB"
                    )
                    return 137
            else:
                if soft_breach_started_ts is not None:
                    logger.log(
                        "已回落到软阈值以下："
                        f"current={_rss_kb_to_gb(rss_kb):.2f} GB，"
                        f"本轮超线持续 {(now_ts - soft_breach_started_ts):.1f}s，已取消击杀计时"
                    )
                soft_breach_started_ts = None

            if now_ts - last_heartbeat_ts >= args.heartbeat_sec:
                logger.log(
                    "heartbeat："
                    f"current={_rss_kb_to_gb(rss_kb):.2f} GB，"
                    f"peak={_rss_kb_to_gb(peak_rss_kb):.2f} GB，"
                    f"members={len(members)}"
                )
                last_heartbeat_ts = now_ts

            time.sleep(args.poll_interval_sec)

    except KeyboardInterrupt:
        logger.log("收到 Ctrl-C，开始终止子进程组")
        _terminate_process_group(pgid, proc, args.grace_sec, logger)
        return 130
    finally:
        logger.close()


if __name__ == "__main__":
    sys.exit(main())
