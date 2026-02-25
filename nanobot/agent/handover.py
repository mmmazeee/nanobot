"""Handover system for cross-session work transfer."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session


async def generate_handover_report(
    session: Session,
    workspace: Path,
    provider: LLMProvider,
    model: str,
) -> str:
    """
    Generate a comprehensive handover report from the current session.

    The report is session-specific (e.g., per Discord channel).
    Includes both session messages and global memory.

    Args:
        session: The current session (all messages will be processed)
        workspace: Workspace path
        provider: LLM provider
        model: Model name

    Returns:
        Path to the generated handover report

    Raises:
        ValueError: If LLM returns empty response
        Exception: If generation fails
    """
    from nanobot.agent.memory import MemoryStore

    # 1. Format current session messages (≤50 messages)
    messages_text = _format_session_messages(session)

    # 2. Read global memory files
    memory_store = MemoryStore(workspace)
    current_memory = memory_store.read_long_term()
    current_history = ""
    if memory_store.history_file.exists():
        current_history = memory_store.history_file.read_text(encoding="utf-8")

    # 3. Build prompt for LLM
    prompt = f"""Generate a comprehensive handover report in markdown format based on the session data below.

## Current Session Data
**Session Key**: {session.key}
**Message Count**: {len(session.messages)}
**Created**: {session.created_at.isoformat()}
**Last Updated**: {session.updated_at.isoformat()}

### Conversation History
{messages_text}

## Global Memory Files (Shared across all sessions/channels)

### MEMORY.md
{current_memory or "(empty)"}

### HISTORY.md (Recent)
{current_history[:5000] if current_history else "(empty)"}

---

Please generate a comprehensive handover report with the following sections:

1. **Executive Summary** (2-3 paragraphs)
   - Overall session purpose and main topics
   - Key achievements
   - Current status

2. **Completed Work** (detailed)
   - Features implemented
   - Bugs fixed
   - Tests completed
   - Files modified/created

3. **In Progress / Next Steps**
   - Currently working on
   - Immediate next steps
   - Blocked items or dependencies

4. **Important Files Map**
   - List relevant files with brief descriptions
   - Include line numbers where applicable

5. **Key Decisions**
   - Important architectural decisions
   - Trade-offs considered
   - Rationales

6. **Testing Status**
   - Tests completed
   - Tests pending
   - Known issues

7. **Configuration Changes**
   - Config file changes (if any)
   - Environment setup notes

8. **Knowledge for Next Session**
   - Important context to preserve
   - User preferences
   - Project-specific notes

9. **Critical Path for Next Session**
   - Priority order of tasks
   - Estimated effort
   - Dependencies

Format the report in markdown with clear sections and subsections. Be detailed and specific.
"""

    try:
        # 4. Call LLM to generate report
        response = await provider.chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert technical writer generating handover reports for development sessions. Be thorough, accurate, and well-organized.",
                },
                {"role": "user", "content": prompt},
            ],
            model=model,
        )

        report_content = response.content
        if not report_content:
            raise ValueError("LLM returned empty response")

        # 5. Save to session-specific file
        handovers_dir = ensure_dir(workspace / "handovers")
        timestamp_str = datetime.now().strftime("%Y-%m-%d")

        # Convert session key to filename prefix
        # "discord:1234567890" -> "discord-1234567890"
        session_prefix = session.key.replace(":", "-")

        # Find today's sequence number for this session
        existing_today = list(handovers_dir.glob(f"HANDOVER_{session_prefix}_{timestamp_str}_*.md"))
        next_num = len(existing_today) + 1

        filename = f"HANDOVER_{session_prefix}_{timestamp_str}_{next_num}.md"
        report_path = handovers_dir / filename

        # Add header metadata
        header = f"""# Handover Report - Session Summary

**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Session**: {session.key}
**Messages**: {len(session.messages)}

---

"""

        full_report = header + report_content
        report_path.write_text(full_report, encoding="utf-8")

        logger.info("Handover report generated for session {}: {}", session.key, report_path)
        return str(report_path)

    except Exception as e:
        logger.error("Failed to generate handover report for session {}: {}", session.key, e)
        raise


def _format_session_messages(session: Session) -> str:
    """
    Format all session messages into readable text.

    Similar to memory consolidation format, but includes all messages.

    Args:
        session: The session to format

    Returns:
        Formatted messages as text
    """
    lines = []
    for msg in session.messages:
        if not msg.get("content"):
            continue

        timestamp = msg.get("timestamp", "?")[:19]  # YYYY-MM-DD HH:MM:SS
        role = msg["role"].upper()
        content = msg["content"]

        # Add tool usage info
        tools_info = ""
        if msg.get("tools_used"):
            tools_info = f" [tools: {', '.join(msg['tools_used'])}]"

        lines.append(f"[{timestamp}] {role}{tools_info}: {content}")

    if not lines:
        return "(No messages in session)"

    return "\n".join(lines)


def list_handovers(
    workspace: Path,
    session_key: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """
    List handover reports, optionally filtered by session.

    Args:
        workspace: Workspace path
        session_key: If provided, only list reports for this session (e.g., "discord:1234567890")
        limit: Maximum number of reports to return

    Returns:
        List of dicts with keys: path, filename, mtime, size, session_key
    """
    handovers_dir = workspace / "handovers"
    if not handovers_dir.exists():
        return []

    # If session_key specified, convert to filename prefix
    session_prefix = None
    if session_key:
        session_prefix = session_key.replace(":", "-")

    reports = []
    for path in sorted(handovers_dir.glob("HANDOVER_*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        if len(reports) >= limit:
            break

        # Filter by session if specified
        if session_prefix:
            if not path.name.startswith(f"HANDOVER_{session_prefix}_"):
                continue

        # Parse session key from filename
        # "HANDOVER_discord-1234567890_2026-02-24_1.md" -> "discord:1234567890"
        parts = path.stem.split("_")  # ["HANDOVER", "discord-1234567890", "2026-02-24", "1"]
        if len(parts) >= 2:
            file_session_key = parts[1].replace("-", ":", 1)  # Only replace first -
        else:
            file_session_key = "unknown"

        stat = path.stat()
        reports.append(
            {
                "path": str(path),
                "filename": path.name,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "session_key": file_session_key,
            }
        )

    return reports


async def load_latest_handover(
    workspace: Path,
    session_key: str | None = None,
) -> str | None:
    """
    Load the most recent handover report.

    Args:
        workspace: Workspace path
        session_key: If provided, load the latest report for this session only

    Returns:
        Report content or None if no reports found
    """
    reports = list_handovers(workspace, session_key=session_key, limit=1)
    if not reports:
        return None

    latest_path = Path(reports[0]["path"])
    return latest_path.read_text(encoding="utf-8")


async def load_handover_by_filename(workspace: Path, filename: str) -> str | None:
    """
    Load a specific handover report by filename.

    Args:
        workspace: Workspace path
        filename: Name of the handover file (e.g., "HANDOVER_2026-02-24_1.md")

    Returns:
        Report content or None if not found
    """
    handovers_dir = workspace / "handovers"
    if not handovers_dir.exists():
        return None

    report_path = handovers_dir / filename
    if not report_path.exists():
        return None

    return report_path.read_text(encoding="utf-8")
