"""Import-path helpers used during QGIS plugin startup."""

import os
import sys


def _normalized_path(path):
    """Return a stable path value for cross-platform comparisons."""
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))


def ensure_import_path(path, prepend=False, search_path=None):
    """Add *path* to an import search path once.

    QGIS imports plugins as packages, but some bundled dependencies use
    top-level imports internally.  Putting the plugin directory on
    ``sys.path`` lets those dependencies resolve without requiring a separate
    installation.  ``search_path`` exists so this behavior can be tested
    without modifying the interpreter's real import path.
    """
    path = os.fspath(path)
    target = sys.path if search_path is None else search_path
    normalized = _normalized_path(path)

    matching_indexes = [
        index
        for index, existing_path in enumerate(target)
        if existing_path is not None
        and _normalized_path(existing_path) == normalized
    ]

    if matching_indexes:
        existing_path = target[matching_indexes[0]]
        if prepend:
            for index in reversed(matching_indexes):
                target.pop(index)
            target.insert(0, existing_path)
        return

    if prepend:
        target.insert(0, path)
    else:
        target.append(path)
