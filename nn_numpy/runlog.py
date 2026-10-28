"""
Timestamped run logs, phase timings and progress lines for the long-running entry points.

``start_run_log(mode)`` puts a tee on the process's stdout and stderr file
descriptors. Every line written to either (print, logger output, Python
warnings, tracebacks, and text from the TensorFlow and PyTorch C++ backends)
is prefixed with ``YYYY-MM-DD HH:MM:SS | ``, shown on the terminal and
appended to ``logs/run_<YYYYmmdd_HHMMSS>_<mode>.log``. The root logger is
configured without a timestamp of its own, so logger lines get exactly one,
from the tee. The tee is closed at interpreter exit, after an uncaught
traceback has been printed, so the traceback is in the file too.

``begin_phase(name)`` / ``finish_run()`` time the stages of a run. A phase
starts when ``begin_phase`` is called and ends when the next one starts or
``finish_run`` runs. Each phase is printed with its duration, and
``finish_run`` prints a table of phases (seconds, share of the run).

``should_report`` and ``format_progress`` give the long loops one progress
line per unit of work (run, configuration, trial) in a fixed format, with a
bounded number of lines per loop.

Input: the mode name of the entry point that starts the log.

Output: logs/run_<YYYYmmdd_HHMMSS>_<mode>.log at the repository root (never
committed), plus the same lines on the terminal.
"""

import atexit
from datetime import datetime
import logging
import math
import os
import sys
import threading
import time
from typing import Any, TextIO

from nn_numpy.config import LOG_DIR

# Prefix format of every logged line.
LINE_STAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# A long loop prints at most about this many progress lines: every item when it
# has at most 50 items (a model's configurations, a study's 50 trials), and every
# ceil(total / 50)-th item otherwise (every 29th of the 1,440 NumPy runs), so a
# log stays readable whatever the loop length.
MAX_PROGRESS_LINES = 50

_TEE: "_Tee | None" = None
_LOG_HANDLER: logging.Handler | None = None
_PHASES: list[dict[str, Any]] = []
_CURRENT: dict[str, Any] | None = None


# ---------------------------------------------------------------------
# Run log
# ---------------------------------------------------------------------


class _Tee:
    """
    Timestamping tee on file descriptors 1 and 2.

    The original descriptors are duplicated, both are pointed at one pipe, and
    a daemon thread reads the pipe line by line, stamping each line before
    writing it to the original stdout and to the log file.

    Attributes
    ----------
    path : str
        The log file being written.
    closed : bool
        True once close() has run.
    """

    def __init__(self, path: str) -> None:
        """
        Open the log file, redirect descriptors 1 and 2 into a pipe and start the reader.

        Parameters
        ----------
        path : str
            Log file to append to (created if missing).

        Returns
        -------
        None
            Starts the reader thread; from here on, everything the process
            writes to stdout or stderr goes through the tee.

        Notes
        -----
        Processing:

        1. Open the log file (UTF-8, line-buffered) and flush the current
           Python streams, so nothing written before the tee is lost.
        2. Duplicate descriptors 1 and 2, and open a text stream on the saved
           stdout for the terminal copy of each line.
        3. Create a pipe and point descriptors 1 and 2 at its write end.
        4. Replace sys.stdout and sys.stderr with line-buffered UTF-8
           streams on the redirected descriptors.
        5. Start the daemon thread that reads the pipe.

        Why steps 2 and 4: on Windows a console-backed sys.stdout writes to
        the console behind descriptor 1 with WriteConsoleW, which fails once
        that descriptor is a pipe, and raw bytes written to a console are
        decoded with its code page, which garbles characters such as ±. Fresh
        streams on the pipe write plain UTF-8, and a text stream opened on
        the saved console descriptor writes Unicode correctly.
        """
        self.path = path
        self.closed = False
        self.file = open(path, "a", buffering=1, encoding="utf-8")
        for stream in (sys.stdout, sys.stderr):
            stream.flush()

        self.orig_out = os.dup(1)
        self.orig_err = os.dup(2)
        self.terminal = open(
            self.orig_out, "w", buffering=1, encoding="utf-8", errors="replace", closefd=False
        )

        self.read_fd, self.write_fd = os.pipe()
        os.dup2(self.write_fd, 1)
        os.dup2(self.write_fd, 2)

        self.saved_streams: tuple[TextIO, TextIO] = (sys.stdout, sys.stderr)
        sys.stdout = open(1, "w", buffering=1, encoding="utf-8", errors="replace", closefd=False)
        sys.stderr = open(2, "w", buffering=1, encoding="utf-8", errors="replace", closefd=False)

        self.thread = threading.Thread(target=self._pump, name="runlog-tee", daemon=True)
        self.thread.start()

    def _pump(self) -> None:
        """
        Read the pipe until it closes and emit each complete line.

        Returns
        -------
        None
            Runs in the reader thread until every write end of the pipe is
            closed.

        Notes
        -----
        Processing:

        1. Read chunks of up to 64 KiB; an empty chunk means end of file.
        2. Split the buffered bytes on newlines and emit each complete line.
        3. At end of file, emit any unterminated remainder and flush the file.
        """
        pending = b""
        while True:
            chunk = os.read(self.read_fd, 65536)
            if not chunk:
                break
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                self._emit(line)
        if pending:
            self._emit(pending)
        self.file.flush()

    def _emit(self, raw: bytes) -> None:
        """
        Stamp one line and write it to the terminal and to the log file.

        Parameters
        ----------
        raw : bytes
            One line without its newline, as read from the pipe.

        Returns
        -------
        None
            Writes "<YYYY-MM-DD HH:MM:SS> | <text>" to both destinations.

        Notes
        -----
        Processing:

        1. Decode as UTF-8 (undecodable bytes are replaced, never raised) and
           drop a trailing carriage return from Windows line endings.
        2. Prefix the current time and write the line to both destinations.
        """
        text = raw.decode("utf-8", errors="replace").rstrip("\r")
        line = f"{time.strftime(LINE_STAMP_FORMAT)} | {text}\n"
        self.terminal.write(line)
        self.file.write(line)

    def close(self) -> None:
        """
        Restore the descriptors and streams, let the reader drain, close the file.

        Returns
        -------
        None
            After this, stdout and stderr write to where they did before the
            tee. Calling it again does nothing.

        Notes
        -----
        Processing:

        1. Flush the tee's streams, so their last lines reach the pipe.
        2. Put back the original sys.stdout / sys.stderr and descriptors 1
           and 2, then close the pipe's write end, which ends the reader.
        3. Wait for the reader (at most 5 s), then close the file and the
           saved descriptors.

        The tee's own streams are flushed but not closed: they keep writing
        to descriptors 1 and 2, which are valid again after the restore, so a
        logging handler still holding one of them never writes to a closed
        file.
        """
        if self.closed:
            return
        self.closed = True
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (OSError, ValueError):
                pass
        sys.stdout, sys.stderr = self.saved_streams
        os.dup2(self.orig_out, 1)
        os.dup2(self.orig_err, 2)
        os.close(self.write_fd)
        # The join is bounded so shutdown never hangs on a stuck reader.
        self.thread.join(timeout=5)
        self.terminal.flush()
        self.file.close()
        for fd in (self.read_fd, self.orig_out, self.orig_err):
            os.close(fd)


def start_run_log(mode: str, log_dir: str | None = None) -> str:
    """
    Start the timestamped run log and route logging and warnings through it.

    Parameters
    ----------
    mode : str
        Tag for the file name, e.g. "train", "benchmarks" or "tune".
    log_dir : str or None, default=None
        Folder for the log; None uses logs/ at the repository root (LOG_DIR).

    Returns
    -------
    str
        Path of the log file being written. If a log is already running, its
        path is returned and nothing else happens.

    Notes
    -----
    Processing:

    1. Create the folder and the file name
       run_<YYYYmmdd_HHMMSS>_<mode>.log.
    2. Start the tee, and close it at interpreter exit.
    3. Configure the root logger to write "LEVEL | message" to the tee's
       stdout, without a timestamp (the tee adds one), and send Python
       warnings to it.
    4. Print the log's path as the first line.
    """
    global _TEE, _LOG_HANDLER
    if _TEE is not None:
        return _TEE.path
    folder = log_dir or LOG_DIR
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"run_{datetime.now():%Y%m%d_%H%M%S}_{mode}.log")

    _TEE = _Tee(path)
    atexit.register(close_run_log)

    _LOG_HANDLER = logging.StreamHandler(sys.stdout)
    _LOG_HANDLER.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    root = logging.getLogger()
    root.addHandler(_LOG_HANDLER)
    root.setLevel(logging.INFO)
    logging.captureWarnings(True)

    print(f"Run log: {path}")
    return path


def close_run_log() -> None:
    """
    Stop the run log, if one is running, and undo its logging setup.

    Returns
    -------
    None
        Restores stdout, stderr, the root logger's handlers and Python's
        warning display. Safe to call when no log is running.

    Notes
    -----
    Processing:

    1. Remove the handler start_run_log added and stop capturing warnings.
    2. Close the tee.

    It runs automatically at interpreter exit; tests call it to restore the
    file descriptors after each test.
    """
    global _TEE, _LOG_HANDLER
    if _LOG_HANDLER is not None:
        logging.getLogger().removeHandler(_LOG_HANDLER)
        logging.captureWarnings(False)
        _LOG_HANDLER = None
    if _TEE is not None:
        _TEE.close()
        _TEE = None


def log_path() -> str | None:
    """
    Return the current run log's path.

    Returns
    -------
    str or None
        The log file being written, or None when no log was started.
    """
    return _TEE.path if _TEE else None


# ---------------------------------------------------------------------
# Phase timings
# ---------------------------------------------------------------------


def _close_current() -> None:
    """
    End the running phase, if any, and record it.

    Returns
    -------
    None
        Appends the phase to the recorded phases and prints its duration.

    Notes
    -----
    Processing:

    1. Store the end time and the duration, rounded to 0.1 s.
    2. Print "◼ <phase> finished in <duration>" and record the phase.
    """
    global _CURRENT
    if _CURRENT is None:
        return
    _CURRENT["ended_at"] = datetime.now()
    _CURRENT["seconds"] = round(time.perf_counter() - _CURRENT["_start"], 1)
    print(f"◼ {_CURRENT['phase']} finished in {format_duration(_CURRENT['seconds'])}")
    _PHASES.append(_CURRENT)
    _CURRENT = None


def begin_phase(name: str) -> None:
    """
    Start timing a phase, ending the phase that was running.

    Parameters
    ----------
    name : str
        Phase label as it should appear in the log and the timings table.

    Returns
    -------
    None
        Prints "▶ <name>".
    """
    global _CURRENT
    _close_current()
    _CURRENT = {
        "phase_order": len(_PHASES) + 1,
        "phase": name,
        "started_at": datetime.now(),
        "_start": time.perf_counter(),
    }
    print(f"▶ {name}")


def finish_run() -> list[dict[str, Any]]:
    """
    Close the running phase and print the table of phase timings.

    Returns
    -------
    list of dict
        One row per phase, in order, with the keys "phase_order" (int),
        "phase" (str), "started_at" and "ended_at" (str, YYYY-MM-DD
        HH:MM:SS), "seconds" (float, rounded to 0.1) and "share_of_run"
        (float, the phase's share of the summed seconds, rounded to 4
        decimals). Empty if no phase was started.

    Notes
    -----
    Processing:

    1. End the running phase.
    2. Build the rows; the run total is the sum of the rounded phase
       seconds, so the rows add up to it exactly.
    3. Print the table and the run total.
    4. Clear the recorded phases, so a later run in the same process starts
       a fresh table.
    """
    _close_current()
    total = sum(p["seconds"] for p in _PHASES)
    rows = [
        {
            "phase_order": p["phase_order"],
            "phase": p["phase"],
            "started_at": p["started_at"].strftime(LINE_STAMP_FORMAT),
            "ended_at": p["ended_at"].strftime(LINE_STAMP_FORMAT),
            "seconds": p["seconds"],
            # A run of phases that all rounded to 0.0 s has no meaningful share.
            "share_of_run": round(p["seconds"] / total, 4) if total else 0.0,
        }
        for p in _PHASES
    ]
    _PHASES.clear()
    if not rows:
        return rows

    width = max(len("Phase"), *(len(r["phase"]) for r in rows))
    print("Phase timings:")
    print(f"  {'#':>2}  {'Phase':<{width}}  {'Seconds':>9}  {'Share':>7}")
    for r in rows:
        print(
            f"  {r['phase_order']:>2}  {r['phase']:<{width}}  "
            f"{r['seconds']:>9.1f}  {r['share_of_run']:>7.1%}"
        )
    print(f"Run total: {format_duration(total)}")
    return rows


def format_duration(seconds: float) -> str:
    """
    Format a duration for the log.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        "Hh MMm SSs" from one hour, "Mm SSs" from one minute, otherwise
        "S.Ss".
    """
    seconds = float(seconds)
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int(seconds % 3600 // 60):02d}m {int(seconds % 60):02d}s"
    if seconds >= 60:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    return f"{seconds:.1f}s"


# ---------------------------------------------------------------------
# Progress lines
# ---------------------------------------------------------------------


def should_report(done: int, total: int, max_lines: int = MAX_PROGRESS_LINES) -> bool:
    """
    Decide whether a loop prints a progress line after its done-th item.

    Parameters
    ----------
    done : int
        Number of items finished so far, 1-based.
    total : int
        Number of items in the loop.
    max_lines : int, default=MAX_PROGRESS_LINES
        Upper bound on the lines a loop prints.

    Returns
    -------
    bool
        True every ceil(total / max_lines)-th item and for the last item.

    Notes
    -----
    Processing:

    1. Compute the interval ceil(total / max_lines), at least 1.
    2. Report when done is a multiple of it, or done is the last item.

    A loop of n items therefore prints floor(n / k) lines, plus one when n is
    not a multiple of k, where k = ceil(n / max_lines): never more than
    max_lines + 1, and every item when n <= max_lines.
    """
    every = max(1, math.ceil(total / max_lines))
    return done % every == 0 or done == total


def format_progress(
    label: str,
    unit: str,
    done: int,
    total: int,
    metric_name: str,
    metric: float,
    seconds: float,
    params: dict[str, Any],
) -> str:
    """
    Format one progress line.

    Parameters
    ----------
    label : str
        What is being trained, usually the architecture or model, e.g. "A2".
    unit : str
        What is counted: "Run", "Config" or "Trial".
    done : int
        1-based number of the finished item.
    total : int
        Number of items in the loop.
    metric_name : str
        Name of the reported metric, e.g. "val BCE".
    metric : float
        Its value.
    seconds : float
        Wall-clock seconds the item took.
    params : dict
        The item's settings, printed as a dict.

    Returns
    -------
    str
        For example "A2 | Config 3/16 | val MSE: 0.012345 | 41s | Params: {...}".
    """
    return (
        f"{label} | {unit} {done}/{total} | {metric_name}: {metric:.6f} | "
        f"{seconds:.0f}s | Params: {params}"
    )
