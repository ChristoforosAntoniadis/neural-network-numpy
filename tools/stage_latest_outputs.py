"""
Stage only the newest stamped outputs in reports/ for the next commit.

Run automatically by .githooks/pre-commit. It can also be run by hand:

    python tools/stage_latest_outputs.py

Input: the stamped output files on disk under reports/ and the git index.
Output: an updated git index (and possibly updated README.md /
docs/DETAILS.md), plus a one-line summary printed to the console.

Processing:
1. Finds the newest stamped version of every output in reports/.
2. Removes older stamped outputs from the index. They stay on disk.
3. Force-adds the newest files (stamped outputs are ignored by .gitignore).
4. Rewrites stamped output names in the docs (README.md, docs/DETAILS.md)
   to the newest stamps, so links and images point at committed files,
   and stages the changed docs.
"""

import os
import re
import subprocess
import sys

# Make "src" importable when this file is run as a script.
# (The package is nn_numpy: adding the repository root to sys.path lets the
# import below work when the hook runs this file directly, not as a module.)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from nn_numpy.config import REPORT_DIR, STAMP_SUFFIX_PATTERN, latest_stamped_outputs

# Documentation files whose stamped output references are kept current.
DOC_FILES = ["README.md", os.path.join("docs", "DETAILS.md")]

# Matches a stamped output name such as "main_summary_20261007-094422.csv".
STAMPED_NAME_PATTERN = re.compile(r"([A-Za-z0-9_.-]+?)_(\d{8}-\d{6})(\.[A-Za-z0-9]+)")


def git(*args: str) -> str:
    """
    Run a git command from the repository root and return its output.

    Parameters
    ----------
    *args : str
        Arguments passed to git, e.g. "ls-files", "--", "reports".

    Returns
    -------
    str
        Everything git printed to standard output, as text.

    Raises
    ------
    subprocess.CalledProcessError
        If git exits with a non-zero status.

    Notes
    -----
    Processing:
    1. Run "git <args>" with ROOT_DIR as the working directory, capturing
       standard output and standard error as text.
    2. Return the captured standard output.
    """
    result = subprocess.run(
        ["git", *args], cwd=ROOT_DIR, check=True, capture_output=True, text=True
    )
    return result.stdout


def to_repo_path(path: str) -> str:
    """
    Convert a file path to a forward-slash path relative to the repo root.

    Parameters
    ----------
    path : str
        Absolute path (or path relative to the current directory) of a file
        inside the repository.

    Returns
    -------
    str
        The path relative to ROOT_DIR with "/" separators, e.g.
        "reports/analysis_20261007-094442.txt", which is the form git prints.

    Notes
    -----
    Processing:
    1. Make the path relative to ROOT_DIR with os.path.relpath.
    2. Replace the operating-system separator (a backslash on Windows)
       with "/".
    """
    return os.path.relpath(path, ROOT_DIR).replace(os.sep, "/")


def is_stamped(repo_path: str) -> bool:
    """
    Check whether a repo path names a stamped output file.

    Parameters
    ----------
    repo_path : str
        A file path, e.g. "reports/main_summary_20261007-094422.csv".

    Returns
    -------
    bool
        True if the file name, without its extension, ends in a
        "_YYYYMMDD-HHMMSS" stamp.

    Notes
    -----
    Processing:
    1. Take the file name and drop its extension.
    2. Search the result for STAMP_SUFFIX_PATTERN.
    """
    stem = os.path.splitext(os.path.basename(repo_path))[0]
    return bool(STAMP_SUFFIX_PATTERN.search(stem))


def update_doc_references(newest_by_name: dict[str, str]) -> list[str]:
    """
    Point every stamped output name in the docs at its newest stamp.

    Parameters
    ----------
    newest_by_name : dict of str to str
        Maps an un-stamped file name (e.g. "analysis.txt") to its newest
        stamped file name (e.g. "analysis_20261007-094442.txt").

    Returns
    -------
    list of str
        Repo paths of the documentation files that were changed (and
        rewritten on disk). Empty if nothing needed updating.

    Raises
    ------
    SystemExit
        If a doc needs updating but already has unstaged changes, because
        staging it would also commit those unrelated edits.

    Notes
    -----
    Processing:
    1. For each file in DOC_FILES that exists, read its text (keeping its
       original line endings).
    2. Replace every stamped output name with the newest stamped name for
       the same output. Names with no entry in `newest_by_name` are left
       unchanged.
    3. Skip the file if nothing changed.
    4. Stop with an error if the file has unstaged edits in git.
    5. Write the updated text back and record the file's repo path.
    """

    def replace(match: re.Match[str]) -> str:
        """
        Return the newest stamped name for one matched stamped name.

        Parameters
        ----------
        match : re.Match of str
            A STAMPED_NAME_PATTERN match: group 1 the name, group 2 the
            stamp, group 3 the extension.

        Returns
        -------
        str
            The newest stamped file name for the same output, or the
            matched text unchanged if the output has no newest version
            (e.g. a name in prose that is not a file in reports/).
        """
        # Rebuild the un-stamped name: group 1 is the name, group 3 the extension.
        name = match.group(1) + match.group(3)
        return newest_by_name.get(name, match.group(0))

    changed = []
    for doc in DOC_FILES:
        path = os.path.join(ROOT_DIR, doc)
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8", newline="") as f:
            text = f.read()
        updated = STAMPED_NAME_PATTERN.sub(replace, text)
        if updated == text:
            continue

        # Refuse to stage a doc that also has unrelated unstaged edits,
        # because "git add" would sweep those edits into the commit.
        if git("diff", "--name-only", "--", doc).strip():
            sys.exit(
                f"stage_latest_outputs: {doc} has unstaged changes. "
                f"Stage or stash them, then commit again."
            )

        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(updated)
        changed.append(to_repo_path(path))

    return changed


def main() -> None:
    """
    Stage the newest stamped outputs and the docs that reference them.

    Parameters
    ----------
    None
        Uses REPORT_DIR, ROOT_DIR, DOC_FILES and the current git index.

    Returns
    -------
    None
        Changes the git index (untracks old outputs, force-adds the newest
        ones, stages updated docs), may rewrite README.md / docs/DETAILS.md,
        and prints a one-line summary.

    Notes
    -----
    Processing:
    1. Find the newest stamped version of every output in reports/. If there
       are none, print a message and stop.
    2. List the stamped files under reports/ that git currently tracks; the
       ones that are not the newest version are "stale".
    3. Remove the stale files from the index ("git rm --cached"), leaving
       them on disk.
    4. Force-add the newest files, since .gitignore ignores stamped outputs.
    5. Update stamped names in the docs to the newest stamps and stage the
       docs that changed.
    6. Print how many files were staged, untracked and updated.
    """
    latest = latest_stamped_outputs(REPORT_DIR)
    if not latest:
        print("stage_latest_outputs: no stamped outputs found in reports/.")
        return

    # Repo paths of the newest outputs, and of the stamped outputs git
    # tracks now; tracked files that are not the newest are stale.
    keep = {to_repo_path(path) for path in latest.values()}
    tracked = [p for p in git("ls-files", "--", "reports").splitlines() if is_stamped(p)]
    stale = [p for p in tracked if p not in keep]

    # Stop tracking older outputs, keeping the files on disk.
    if stale:
        git("rm", "--cached", "--quiet", "--", *stale)

    # Stamped outputs are ignored, so the newest ones must be force-added.
    git("add", "--force", "--", *sorted(keep))

    # Docs refer to outputs by file name only, so map bare un-stamped names
    # to bare newest stamped names.
    newest_by_name = {
        os.path.basename(key): os.path.basename(path) for key, path in latest.items()
    }
    changed_docs = update_doc_references(newest_by_name)
    if changed_docs:
        git("add", "--", *changed_docs)

    print(
        f"stage_latest_outputs: {len(keep)} newest outputs staged, "
        f"{len(stale)} older outputs untracked, "
        f"{len(changed_docs)} doc file(s) updated."
    )


# Running this file (by hand or from the pre-commit hook) stages the newest
# stamped outputs in reports/ and updates the docs that link to them.
if __name__ == "__main__":
    main()
