"""
Shared paths, the global random seed and run-stamp helpers.

Every other module imports its folder locations (project root, raw data,
reports and figures) and the default random seed from here. The folders
follow the Cookiecutter Data Science layout. This module also
creates the run stamp (a YYYYMMDD-HHMMSS time string) that is added to the
name of every output file, and provides helpers that find the newest
stamped output on disk so that downstream scripts (analysis, plots, the
git pre-commit tool) can read the most recent results.

Input: none. The paths are derived from this file's location and the run
stamp from the clock when the module is first imported.

Output: the path constants, RANDOM_SEED, the run stamp and the stamped-file
helpers below. Importing the module writes nothing and creates no folders.
"""

from datetime import datetime
import glob
import os
import re

# Paths are calculated from where this file lives: the package folder
# (nn_numpy/) sits directly inside the project root.
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PACKAGE_DIR)

# data/raw/: the original UCI datasets, as downloaded (never modified).
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATASETS_DIR = os.path.join(DATA_DIR, "raw")

# reports/: generated results (summary CSV, full results JSON, analysis text).
REPORT_DIR = os.path.join(ROOT_DIR, "reports")

# reports/comparisons/: the short text analyses written next to the comparison plots.
COMPARISONS_TEXT_DIR = os.path.join(REPORT_DIR, "comparisons")

# reports/figures/: every generated figure, split by kind.
FIGURES_DIR = os.path.join(REPORT_DIR, "figures")
COMPARISONS_FIGURES_DIR = os.path.join(FIGURES_DIR, "comparisons")
PREPROCESSING_GRAPHS_DIR = os.path.join(FIGURES_DIR, "eda")
BENCHMARK_FIGURES_DIR = os.path.join(FIGURES_DIR, "benchmarks")

# Global Configurations
# Default seed for the data split, weight initialization, dropout masks and
# shuffling, used whenever a config or results file does not give one.
# The value carries no meaning: any fixed seed makes reruns identical, and 42
# keeps the original single-seed results reproducible.
RANDOM_SEED = 42

# Run stamp used to give every output file of a run a unique name.
# It is computed once when this module is first imported, so every module
# in the same process shares the same stamp. Format: YYYYMMDD-HHMMSS.
STAMP_FORMAT = "%Y%m%d-%H%M%S"
RUN_STAMP = datetime.now().strftime(STAMP_FORMAT)

# Matches the "_YYYYMMDD-HHMMSS" suffix that stamped_filename() appends.
STAMP_SUFFIX_PATTERN = re.compile(r"_\d{8}-\d{6}$")


def stamped_filename(filename: str, stamp: str = RUN_STAMP) -> str:
    """
    Insert a run stamp into a file name, just before the extension.

    Parameters
    ----------
    filename : str
        A bare file name such as "main_summary.csv". It may also contain a
        folder part; only the text after the last dot is treated as the
        extension.
    stamp : str, default=RUN_STAMP
        Stamp to insert, in YYYYMMDD-HHMMSS format. The default is the stamp
        created when this module was first imported.

    Returns
    -------
    str
        The stamped file name, for example "main_summary_20261007-153012.csv".

    Notes
    -----
    Processing:
    1. Split the name into its root and extension with os.path.splitext.
    2. Join them back as "<root>_<stamp><ext>".

    Why: putting a unique stamp in every output name means a new run never
    overwrites the files of an earlier run.
    """
    root, ext = os.path.splitext(filename)
    return f"{root}_{stamp}{ext}"


def latest_stamped_file(directory: str, basename: str, ext: str) -> str:
    """
    Find the most recent stamped file named "<basename>_<stamp><ext>".

    Parameters
    ----------
    directory : str
        Folder to search in (not searched recursively).
    basename : str
        File name without the stamp or extension, e.g. "main_results_full".
    ext : str
        Extension including the dot, e.g. ".json".

    Returns
    -------
    str
        Full path of the newest matching file.

    Raises
    ------
    FileNotFoundError
        If no file in the directory matches "<basename>_<YYYYMMDD-HHMMSS><ext>".

    Notes
    -----
    Processing:
    1. List the files matching the glob "<basename>_*<ext>" in the directory.
    2. Keep only those whose name (without extension) really ends in a
       "_YYYYMMDD-HHMMSS" stamp.
    3. Raise FileNotFoundError if none are left.
    4. Return the candidate whose file name sorts last alphabetically.

    Because the stamp has a fixed width (YYYYMMDD-HHMMSS), sorting the
    file names alphabetically also sorts them chronologically, so the
    alphabetically largest name is the newest file.
    """
    candidates = []
    for path in glob.glob(os.path.join(directory, f"{basename}_*{ext}")):
        stem = os.path.splitext(os.path.basename(path))[0]
        # Only accept names that really end in a stamp, so that an unrelated
        # file such as "main_results_full_backup.json" is never selected.
        if STAMP_SUFFIX_PATTERN.search(stem):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No file matching {basename}_<YYYYMMDD-HHMMSS>{ext} found in "
            f"{directory}. Run python -m nn_numpy.modeling.train first to produce one."
        )

    return max(candidates, key=os.path.basename)


def latest_stamped_outputs(directory: str) -> dict[str, str]:
    """
    Find the newest stamped version of every output under a directory.

    Parameters
    ----------
    directory : str
        Folder to search recursively, usually REPORT_DIR.

    Returns
    -------
    dict of str to str
        Maps each output's un-stamped name (path relative to `directory`,
        e.g. "comparisons/depth_analysis.txt") to the full path of its newest
        stamped file (e.g. ".../comparisons/depth_analysis_20261007-094438.txt").
        Empty if no stamped files exist.

    Notes
    -----
    Processing:
    1. Walk every file under `directory`, recursively, skipping folders.
    2. Skip files whose name (without extension) does not end in a
       "_YYYYMMDD-HHMMSS" stamp.
    3. Remove the stamp to get the output's stable name, prefixed with its
       sub-folder relative to `directory` (no prefix for files at the top).
    4. Keep, for each stable name, the file whose name sorts last
       alphabetically, which is the newest because the stamp has a fixed
       width.

    Each output is treated on its own, so the newest analysis report and
    the newest summary CSV may come from different runs.
    """
    latest: dict[str, str] = {}
    for path in glob.glob(os.path.join(directory, "**", "*"), recursive=True):
        if not os.path.isfile(path):
            continue
        stem, ext = os.path.splitext(os.path.basename(path))
        match = STAMP_SUFFIX_PATTERN.search(stem)
        if not match:
            continue

        # Strip the stamp to get the output's stable name.
        relative_dir = os.path.relpath(os.path.dirname(path), directory)
        name = stem[: match.start()] + ext
        key = name if relative_dir == "." else os.path.join(relative_dir, name)

        # Fixed-width stamps sort alphabetically in time order.
        if key not in latest or os.path.basename(path) > os.path.basename(latest[key]):
            latest[key] = path

    return latest


def resolve_results_path(explicit_path: str | None = None) -> str:
    """
    Decide which full-results JSON a downstream script should read.

    Parameters
    ----------
    explicit_path : str or None, default=None
        A path given by the user, usually from the command line.
        When None (or an empty string), the newest
        reports/main_results_full_<stamp>.json is used.

    Returns
    -------
    str
        Path of the results file to load. An explicit path is returned
        unchanged, without checking that it exists.

    Raises
    ------
    FileNotFoundError
        If no explicit path is given and reports/ contains no stamped
        main_results_full JSON file (raised by latest_stamped_file).

    Notes
    -----
    Processing:
    1. If an explicit path was given, return it.
    2. Otherwise return the newest stamped "main_results_full" JSON in
       REPORT_DIR, found with latest_stamped_file.
    """
    if explicit_path:
        return explicit_path
    return latest_stamped_file(REPORT_DIR, "main_results_full", ".json")
