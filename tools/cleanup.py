"""
tools/cleanup.py — Workspace Expiry & Cleanup Utility

Provides a lightweight cleanup routine that is called at the start of
every POST /analyze request to delete stale per-job workspace directories.

Each analysis job produces a UUID-named workspace under workspace/ that
contains generated scripts, cleaned data, and trained model artifacts.
These are kept on disk so the download endpoints can serve them after the
/analyze response is returned. The cleanup function removes any workspace
that has not been modified within WORKSPACE_EXPIRY_SECONDS (default: 1 hour),
preventing unbounded disk growth on long-running servers.

Usage:
    from tools.cleanup import cleanup_expired_workspaces
    cleanup_expired_workspaces()
"""

import shutil
import time
from pathlib import Path


# Root directory that contains all per-job workspace folders
WORKSPACE_ROOT = Path("workspace")

# How long (in seconds) to keep a workspace after its last modification.
# Default: 1 hour — long enough for the client to download artifacts.
WORKSPACE_EXPIRY_SECONDS = 60 * 60  # 1 hour


def cleanup_expired_workspaces():
    """
    Scan WORKSPACE_ROOT and delete any workspace directory whose last
    modification time is older than WORKSPACE_EXPIRY_SECONDS.

    Called at the start of each /analyze request so cleanup happens
    opportunistically without a background scheduler.

    Behaviour:
    - Skips non-directory entries (e.g. stray files in workspace/).
    - Silently returns if WORKSPACE_ROOT does not exist yet.
    - Logs each deleted workspace and any deletion errors to stdout.
    - Errors on individual workspaces are caught and logged so a single
      locked or permission-denied directory cannot abort the whole sweep.
    """

    # Nothing to clean if the workspace root hasn't been created yet
    if not WORKSPACE_ROOT.exists():
        return

    current_time = time.time()

    for workspace in WORKSPACE_ROOT.iterdir():

        # Only process directories — ignore stray files
        if not workspace.is_dir():
            continue

        try:
            # Use the directory's last modification timestamp as its age proxy.
            # The graph writes files into the workspace during execution, so
            # st_mtime reflects roughly when the job last produced output.
            modified_time = workspace.stat().st_mtime

            age = current_time - modified_time

            if age > WORKSPACE_EXPIRY_SECONDS:

                shutil.rmtree(
                    workspace,
                    ignore_errors=True
                )

                print(
                    f"[CLEANUP] Deleted expired workspace: "
                    f"{workspace.name}"
                )

        except Exception as exc:

            # Log but do not re-raise — a single bad directory should not
            # prevent other workspaces from being cleaned up
            print(
                f"[CLEANUP] Could not delete "
                f"{workspace}: {exc}"
            )