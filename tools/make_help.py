"""
Print the list of Makefile commands with their descriptions (used by ``make help``).

Input: the path of a Makefile, given as the first command-line argument.
Output: one line per documented target, printed to the console.

A target is documented by a ``## description`` comment line above it. Lines
between the comment and the target (such as ``.PHONY: name``) are skipped.
Keeping this logic in Python, rather than inside the Makefile, means it works
the same whether make runs its recipes with sh or with Windows cmd.exe.
"""

import re
import sys

# A target line: a name made of letters, digits, "_" or "-", then a colon.
TARGET_PATTERN = re.compile(r"^([A-Za-z0-9_-]+):")


def documented_targets(makefile_text: str) -> list[tuple[str, str]]:
    """
    Find each documented target and its description in a Makefile.

    Parameters
    ----------
    makefile_text : str
        The full contents of the Makefile.

    Returns
    -------
    list of tuple of (str, str)
        (target name, description) pairs, in the order they appear.

    Notes
    -----
    Processing:
    1. Walk the lines in order.
    2. When a line starts with "## ", remember the rest of it as the
       pending description.
    3. When a later line matches TARGET_PATTERN while a description is
       pending, record (target, description) and clear the pending one.
       Lines in between, such as ``.PHONY: name``, are ignored.
    """
    targets = []
    description = None
    for line in makefile_text.splitlines():
        if line.startswith("## "):
            description = line[3:].strip()
            continue
        match = TARGET_PATTERN.match(line)
        if match and description is not None:
            targets.append((match.group(1), description))
            description = None
    return targets


def main(makefile_path: str) -> None:
    """
    Print the documented Makefile targets as an aligned list.

    Parameters
    ----------
    makefile_path : str
        Path of the Makefile to read.

    Returns
    -------
    None
        Prints "Available commands:" followed by one "make <target>
        <description>" line per target.

    Notes
    -----
    Processing:
    1. Read the Makefile as UTF-8.
    2. Extract the documented targets with documented_targets().
    3. Print them with the target names padded to a common width.
    """
    with open(makefile_path, encoding="utf-8") as f:
        targets = documented_targets(f.read())

    # default=0 keeps a Makefile without documented targets from raising ValueError.
    width = max((len(name) for name, _ in targets), default=0)
    print("Available commands:\n")
    for name, description in targets:
        print(f"  make {name:<{width}}  {description}")


# Running this file prints the documented targets of the Makefile given as
# the first argument (default: Makefile in the current folder).
if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Makefile")
