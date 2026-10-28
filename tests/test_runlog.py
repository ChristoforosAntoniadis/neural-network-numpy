"""
Tests for the run log (nn_numpy.runlog): the timestamping tee, the phase timings and
the progress-line interval.
"""

from collections.abc import Iterator
import logging
import re
import time
from typing import TYPE_CHECKING
import warnings

import pytest

from nn_numpy import runlog

if TYPE_CHECKING:
    from pathlib import Path

# "YYYY-MM-DD HH:MM:SS | " at the start of every logged line.
STAMP = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| "


@pytest.fixture(autouse=True)
def closed_run_log() -> Iterator[None]:
    """
    Close any run log a test started, so file descriptors 1 and 2 are restored.

    Returns
    -------
    collections.abc.Iterator of None
        Yields once; the code after the yield runs as the test's teardown.

    Notes
    -----
    Processing:
    1. Let the test run.
    2. Close the tee (a no-op if none is running) and drop leftover phases.
    """
    yield
    runlog.close_run_log()
    runlog.finish_run()


def read_log(path: str) -> list[str]:
    """
    Read a run log's lines.

    Parameters
    ----------
    path : str
        The log file.

    Returns
    -------
    list of str
        The lines, without line endings.
    """
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def test_log_captures_print_logger_and_warning_lines_with_one_timestamp(
    tmp_path: "Path",
) -> None:
    """
    A print, a logger line and a warning all reach the file, each with the prefix.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Folder for the log (pytest's per-test temporary directory).

    Notes
    -----
    The logger line must read "<stamp> | INFO | ...": the tee adds the only
    timestamp, so a second one (from the logger's own format) would show up
    as another date after "INFO | ". The file name follows
    run_<YYYYmmdd_HHMMSS>_<mode>.log.
    """
    path = runlog.start_run_log("test", log_dir=str(tmp_path))
    print("a printed line ±")
    logging.getLogger("nn_numpy.test").info("a logger line")
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn("a warning line", UserWarning, stacklevel=1)
    runlog.close_run_log()

    assert re.fullmatch(r"run_\d{8}_\d{6}_test\.log", path.replace("\\", "/").rsplit("/", 1)[1])
    lines = read_log(path)
    assert all(re.match(STAMP, line) for line in lines)
    assert any(re.search(STAMP + r"a printed line ±$", line) for line in lines)
    assert any(re.search(STAMP + r"INFO \| a logger line$", line) for line in lines)
    assert any(
        re.search(STAMP + r"WARNING \| .*UserWarning: a warning line", line) for line in lines
    )


def test_phases_give_one_row_each_with_durations_that_add_up(tmp_path: "Path") -> None:
    """
    Two phases give two rows, in order, whose seconds and shares add up.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Folder for the log.

    Notes
    -----
    The phases sleep 0.2 s and 0.4 s, so after rounding to 0.1 s their
    seconds are about 0.2 and 0.4 (a busy machine can add a tenth), their
    shares about 1/3 and 2/3, and the shares sum to 1 within the 4-decimal
    rounding (2 × 0.00005 = 1e-4). The run total printed in the log is the
    sum of the rounded phase seconds.
    """
    path = runlog.start_run_log("phases", log_dir=str(tmp_path))
    runlog.begin_phase("First")
    time.sleep(0.2)
    runlog.begin_phase("Second")
    time.sleep(0.4)
    rows = runlog.finish_run()
    runlog.close_run_log()

    assert [r["phase"] for r in rows] == ["First", "Second"]
    assert [r["phase_order"] for r in rows] == [1, 2]
    assert rows[0]["seconds"] >= 0.2 and rows[1]["seconds"] >= 0.4
    assert sum(r["share_of_run"] for r in rows) == pytest.approx(1.0, abs=1e-4)
    total = runlog.format_duration(sum(r["seconds"] for r in rows))
    lines = read_log(path)
    assert any(line.endswith(f"Run total: {total}") for line in lines)
    assert sum("finished in" in line for line in lines) == 2


@pytest.mark.parametrize(
    "total, max_lines, expected",
    [
        # Every item when the loop is no longer than max_lines.
        (16, 50, 16),
        (50, 50, 50),
        # 51 items: every 2nd (ceil(51 / 50) = 2) gives 25 lines, plus the last.
        (51, 50, 26),
        # 95 items: every 2nd gives 47 lines, plus item 95.
        (95, 50, 48),
        # The NumPy grid: every 29th of 1,440 (ceil(1440 / 50) = 29) gives
        # floor(1440 / 29) = 49 lines, plus run 1,440.
        (1440, 50, 50),
        # Every 10%: 1,440 / 10 = 144, a multiple of the interval, so no extra line.
        (1440, 10, 10),
    ],
)
def test_should_report_prints_a_bounded_number_of_lines(
    total: int, max_lines: int, expected: int
) -> None:
    """
    should_report selects the hand-computed number of items for a known loop.

    Parameters
    ----------
    total : int
        Loop length.
    max_lines : int
        Upper bound passed to should_report.
    expected : int
        Hand-computed number of reported items (see the cases' comments).

    Notes
    -----
    The last item is always reported, so the final count is never missing.
    """
    reported = [
        done for done in range(1, total + 1) if runlog.should_report(done, total, max_lines)
    ]
    assert len(reported) == expected
    assert reported[-1] == total


def test_progress_lines_in_the_log_match_the_interval(tmp_path: "Path") -> None:
    """
    A 95-item loop writes 48 stamped progress lines to the log.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Folder for the log.

    Notes
    -----
    95 items with MAX_PROGRESS_LINES = 50: every 2nd item (47 lines) plus the
    last one, as in test_should_report_prints_a_bounded_number_of_lines.
    """
    path = runlog.start_run_log("progress", log_dir=str(tmp_path))
    for done in range(1, 96):
        if runlog.should_report(done, 95):
            print(runlog.format_progress("A2", "Run", done, 95, "val MSE", 0.5, 1.0, {"lr": 0.1}))
    runlog.close_run_log()

    progress = [line for line in read_log(path) if " | A2 | Run " in line]
    assert len(progress) == 48
    assert all(re.match(STAMP, line) for line in progress)


def test_format_progress_matches_the_documented_format() -> None:
    """
    One progress line has the label, the count, the metric, the seconds and the params.

    Notes
    -----
    0.0123451 is printed to 6 decimals (0.012345) and 41.4 s to whole seconds
    (41s).
    """
    line = runlog.format_progress("A2", "Config", 3, 16, "val MSE", 0.0123451, 41.4, {"lr": 0.1})
    assert line == "A2 | Config 3/16 | val MSE: 0.012345 | 41s | Params: {'lr': 0.1}"


def test_format_duration_switches_units_at_one_minute_and_one_hour() -> None:
    """
    Durations below a minute keep a decimal; longer ones use minutes and hours.

    Notes
    -----
    59.94 s rounds to "59.9s"; 125 s = 2 min 5 s; 3725 s = 1 h 2 min 5 s.
    """
    assert runlog.format_duration(59.94) == "59.9s"
    assert runlog.format_duration(125) == "2m 05s"
    assert runlog.format_duration(3725) == "1h 02m 05s"
