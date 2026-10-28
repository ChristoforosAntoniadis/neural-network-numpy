"""
Tests for the timestamped output-file helpers in nn_numpy.config: building a
stamped filename and finding the newest stamped file(s) in a folder.
"""

import os
from typing import TYPE_CHECKING

import pytest

from nn_numpy.config import latest_stamped_file, latest_stamped_outputs, stamped_filename

if TYPE_CHECKING:
    # Only for the tmp_path annotations; the tests themselves never import pathlib.
    from pathlib import Path


def test_stamped_filename_inserts_stamp_before_extension() -> None:
    """
    stamped_filename inserts "_<stamp>" just before the file extension.

    Notes
    -----
    "main_summary.csv" must become "main_summary_<stamp>.csv", and for
    "a.b.csv" only the last dot counts as the extension separator.
    """
    assert (
        stamped_filename("main_summary.csv", "20260101-000000")
        == "main_summary_20260101-000000.csv"
    )
    # Only the last dot is treated as the extension separator.
    assert stamped_filename("a.b.csv", "20260101-000000") == "a.b_20260101-000000.csv"


def test_latest_stamped_file_picks_newest_and_ignores_decoys(tmp_path: "Path") -> None:
    """
    latest_stamped_file returns the newest correctly stamped file and ignores decoys.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty temporary directory provided by pytest.

    Notes
    -----
    Three stamped .json files, one unstamped "backup" file and one newer
    stamped file with the wrong extension are created. The function must
    return the path of the newest stamped .json file
    (20260301-120000). The files are created out of time order, so the
    result cannot come from the creation order or the modification time.
    """
    for name in [
        "main_results_full_20260101-000000.json",
        "main_results_full_20260301-120000.json",
        "main_results_full_20260201-000000.json",
        "main_results_full_backup.json",  # not a stamp: must be ignored
        "main_results_full_20260401-000000.csv",  # wrong extension
    ]:
        (tmp_path / name).write_text("[]")

    latest = latest_stamped_file(str(tmp_path), "main_results_full", ".json")
    assert latest == str(tmp_path / "main_results_full_20260301-120000.json")


def test_latest_stamped_file_raises_when_nothing_matches(tmp_path: "Path") -> None:
    """
    latest_stamped_file raises FileNotFoundError when no file matches.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty temporary directory provided by pytest.

    Notes
    -----
    Searching the empty directory must raise FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        latest_stamped_file(str(tmp_path), "main_results_full", ".json")


def test_latest_stamped_outputs_keeps_newest_per_output(tmp_path: "Path") -> None:
    """
    latest_stamped_outputs keeps only the newest stamped file for each output, including subfolders.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty temporary directory provided by pytest.

    Notes
    -----
    Stamped files for three outputs (two of them with two versions, one
    inside a "comparisons" subfolder) and one unstamped file are created.
    The result must map each unstamped relative name to the path of its
    newest version, and leave out the unstamped "notes.txt".
    """
    (tmp_path / "comparisons").mkdir()
    for name in [
        "main_summary_20260101-000000.csv",
        "main_summary_20260201-000000.csv",
        "analysis_20260101-000000.txt",
        "comparisons/depth_analysis_20260301-000000.txt",
        "comparisons/depth_analysis_20260101-000000.txt",
        "notes.txt",  # not stamped: ignored
    ]:
        (tmp_path / name).write_text("x")

    latest = latest_stamped_outputs(str(tmp_path))

    assert latest == {
        "main_summary.csv": str(tmp_path / "main_summary_20260201-000000.csv"),
        "analysis.txt": str(tmp_path / "analysis_20260101-000000.txt"),
        os.path.join("comparisons", "depth_analysis.txt"): str(
            tmp_path / "comparisons" / "depth_analysis_20260301-000000.txt"
        ),
    }
