"""
Hermes MCP Tools Server — expose self-improvement tools to Claude CLI via MCP.

When Hermes uses Claude CLI as a backend provider, Claude CLI handles its own
tool execution.  This means Hermes' internal self-improvement tools (memory,
skills, session_search, todo) never get called, breaking the self-improvement
loop.

This MCP server bridges that gap: it runs alongside the Claude CLI subprocess
and exposes these tools over stdio MCP, so Claude CLI can invoke them as
external MCP tools.

Tools exposed:
  memory          — persistent curated memory (add/replace/remove)
  skill_manage    — create, edit, patch, delete skills and supporting files
  skill_view      — view skill content and linked files
  session_search  — search past session transcripts with LLM summaries
  todo            — in-memory task list for session planning

Usage:
    hermes mcp serve-tools
    hermes mcp serve-tools --verbose

MCP client config (e.g. claude_desktop_config.json):
    {
        "mcpServers": {
            "hermes-tools": {
                "command": "hermes",
                "args": ["mcp", "serve-tools"]
            }
        }
    }
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hermes.mcp_tools_serve")

# ---------------------------------------------------------------------------
# Lazy MCP SDK import
# ---------------------------------------------------------------------------

_MCP_SERVER_AVAILABLE = False
try:
    from mcp.server.fastmcp import FastMCP

    _MCP_SERVER_AVAILABLE = True
except ImportError:
    FastMCP = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Lazy singletons — initialized once on first use
# ---------------------------------------------------------------------------

_memory_store = None
_todo_store = None
_session_db = None


def _get_memory_store():
    """Get or create the shared MemoryStore instance."""
    global _memory_store
    if _memory_store is None:
        try:
            from tools.memory_tool import MemoryStore

            _memory_store = MemoryStore()
            _memory_store.load_from_disk()
        except Exception as e:
            logger.warning("Failed to initialize MemoryStore: %s", e)
            return None
    return _memory_store


def _get_session_db():
    """Get or create the shared SessionDB instance."""
    global _session_db
    if _session_db is None:
        try:
            from hermes_state import SessionDB

            _session_db = SessionDB()
        except Exception as e:
            logger.warning("Failed to initialize SessionDB: %s", e)
            return None
    return _session_db


def _get_todo_store():
    """Get or create the shared TodoStore instance."""
    global _todo_store
    if _todo_store is None:
        try:
            from tools.todo_tool import TodoStore

            _todo_store = TodoStore()
        except Exception as e:
            logger.warning("Failed to initialize TodoStore: %s", e)
            return None
    return _todo_store


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

def create_mcp_tools_server() -> "FastMCP":
    """Create and return the Hermes tools MCP server with all tools registered."""
    if not _MCP_SERVER_AVAILABLE:
        raise ImportError(
            "MCP server requires the 'mcp' package. "
            f"Install with: {sys.executable} -m pip install 'mcp'"
        )

    mcp = FastMCP(
        "hermes-tools",
        instructions=(
            "Hermes self-improvement tools. Use these tools to manage persistent "
            "memory, skills (procedural knowledge), search past sessions, and "
            "track tasks within the current session."
        ),
    )

    # -- memory ---------------------------------------------------------------

    @mcp.tool()
    def memory(
        action: str,
        target: str = "memory",
        content: Optional[str] = None,
        old_text: Optional[str] = None,
    ) -> str:
        """Save durable information to persistent memory that survives across sessions.

        Memory is injected into future sessions, so keep entries compact and
        focused on facts that will still matter later.

        WHEN TO SAVE (do this proactively):
        - User corrects you or says 'remember this'
        - User shares a preference, habit, or personal detail
        - You discover something about the environment
        - You learn a convention or workflow specific to this user's setup

        TWO TARGETS:
        - 'user': who the user is (name, role, preferences, style, pet peeves)
        - 'memory': your notes (environment facts, conventions, tool quirks, lessons)

        Args:
            action: The action to perform: 'add', 'replace', or 'remove'.
            target: Which memory store: 'memory' for personal notes, 'user' for user profile.
            content: The entry content. Required for 'add' and 'replace'.
            old_text: Short unique substring identifying the entry to replace or remove.
        """
        store = _get_memory_store()
        if store is None:
            return json.dumps({
                "success": False,
                "error": "MemoryStore is not available. Check that the memories directory exists.",
            })

        try:
            from tools.memory_tool import memory_tool

            return memory_tool(
                action=action,
                target=target,
                content=content,
                old_text=old_text,
                store=store,
            )
        except Exception as e:
            return json.dumps({"success": False, "error": f"memory tool error: {e}"})

    # -- skill_manage ---------------------------------------------------------

    @mcp.tool()
    def skill_manage(
        action: str,
        name: str,
        content: Optional[str] = None,
        category: Optional[str] = None,
        file_path: Optional[str] = None,
        file_content: Optional[str] = None,
        old_string: Optional[str] = None,
        new_string: Optional[str] = None,
        replace_all: bool = False,
    ) -> str:
        """Manage skills (create, update, delete). Skills are reusable procedural
        knowledge for recurring task types.

        Actions:
        - create: Create a new skill (provide full SKILL.md content with YAML frontmatter + body)
        - edit: Full rewrite of an existing skill's SKILL.md
        - patch: Targeted find-and-replace within SKILL.md or a supporting file
        - delete: Remove a skill entirely
        - write_file: Add/overwrite a supporting file (under references/, templates/, scripts/, assets/)
        - remove_file: Remove a supporting file from a skill

        Args:
            action: The action to perform: 'create', 'edit', 'patch', 'delete', 'write_file', or 'remove_file'.
            name: Skill name (lowercase, hyphens/underscores, max 64 chars). Must match existing skill for patch/edit/delete/write_file/remove_file.
            content: Full SKILL.md content (YAML frontmatter + markdown body). Required for 'create' and 'edit'.
            category: Optional category for organizing the skill (e.g. 'devops'). Only used with 'create'.
            file_path: Path to a supporting file within the skill directory. Required for 'write_file'/'remove_file'. Optional for 'patch' (defaults to SKILL.md).
            file_content: Content for the file. Required for 'write_file'.
            old_string: Text to find in the file. Required for 'patch'.
            new_string: Replacement text. Required for 'patch'. Can be empty string to delete matched text.
            replace_all: For 'patch': replace all occurrences instead of requiring a unique match.
        """
        try:
            from tools.skill_manager_tool import skill_manage as _skill_manage

            return _skill_manage(
                action=action,
                name=name,
                content=content,
                category=category,
                file_path=file_path,
                file_content=file_content,
                old_string=old_string,
                new_string=new_string,
                replace_all=replace_all,
            )
        except Exception as e:
            return json.dumps({"success": False, "error": f"skill_manage error: {e}"})

    # -- skill_view -----------------------------------------------------------

    @mcp.tool()
    def skill_view(
        name: str,
        file_path: Optional[str] = None,
    ) -> str:
        """View the content of a skill or a specific file within a skill directory.

        Load a skill's full content or access its linked files (references,
        templates, scripts). The first call returns SKILL.md content plus a
        'linked_files' dict showing available references/templates/scripts.
        To access those, call again with the file_path parameter.

        Args:
            name: The skill name (e.g. 'axolotl'). For plugin-provided skills, use the qualified form 'plugin:skill'.
            file_path: Optional path to a linked file within the skill (e.g. 'references/api.md', 'templates/config.yaml').
        """
        try:
            from tools.skills_tool import skill_view as _skill_view

            return _skill_view(name=name, file_path=file_path)
        except Exception as e:
            return json.dumps({"success": False, "error": f"skill_view error: {e}"})

    # -- session_search -------------------------------------------------------

    @mcp.tool()
    def session_search(
        query: Optional[str] = None,
        role_filter: Optional[str] = None,
        limit: int = 3,
    ) -> str:
        """Search your long-term memory of past conversations, or browse recent sessions.

        TWO MODES:
        1. Recent sessions (no query): Call with no arguments to see what was
           worked on recently. Returns titles, previews, and timestamps.
        2. Keyword search (with query): Search for specific topics across all
           past sessions. Returns LLM-generated summaries of matching sessions.

        USE THIS PROACTIVELY when:
        - The user says 'we did this before', 'remember when', 'last time'
        - The user asks about a topic you worked on before
        - You want to check if you've solved a similar problem before

        Search syntax: keywords joined with OR for broad recall, phrases for
        exact match ("docker networking"), boolean (python NOT java), prefix (deploy*).

        Args:
            query: Search query (keywords, phrases, boolean expressions). Omit to browse recent sessions.
            role_filter: Optional: only search messages from specific roles (comma-separated, e.g. 'user,assistant').
            limit: Max sessions to summarize (default 3, max 5).
        """
        db = _get_session_db()
        if db is None:
            return json.dumps({
                "success": False,
                "error": "Session database is not available. Check that state.db exists.",
            })

        try:
            from tools.session_search_tool import session_search as _session_search

            return _session_search(
                query=query or "",
                role_filter=role_filter,
                limit=limit,
                db=db,
                current_session_id=None,
            )
        except Exception as e:
            return json.dumps({"success": False, "error": f"session_search error: {e}"})

    # -- todo -----------------------------------------------------------------

    @mcp.tool()
    def todo(
        todos: Optional[List[Dict[str, Any]]] = None,
        merge: bool = False,
    ) -> str:
        """Manage your task list for the current session.

        Use for complex tasks with 3+ steps or when the user provides multiple
        tasks. Call with no parameters to read the current list.

        Writing:
        - Provide 'todos' array to create/update items
        - merge=false (default): replace the entire list with a fresh plan
        - merge=true: update existing items by id, add any new ones

        Each item: {id: string, content: string, status: pending|in_progress|completed|cancelled}
        List order is priority. Only ONE item should be in_progress at a time.

        Args:
            todos: Task items to write. Each item needs 'id', 'content', and 'status' fields. Omit to read current list.
            merge: If true, update existing items by id and add new ones. If false (default), replace the entire list.
        """
        store = _get_todo_store()
        if store is None:
            return json.dumps({
                "success": False,
                "error": "TodoStore is not available.",
            })

        try:
            from tools.todo_tool import todo_tool

            return todo_tool(todos=todos, merge=merge, store=store)
        except Exception as e:
            return json.dumps({"success": False, "error": f"todo error: {e}"})

    return mcp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_mcp_tools_server(verbose: bool = False) -> None:
    """Start the Hermes tools MCP server on stdio."""
    if not _MCP_SERVER_AVAILABLE:
        print(
            "Error: MCP server requires the 'mcp' package.\n"
            f"Install with: {sys.executable} -m pip install 'mcp'",
            file=sys.stderr,
        )
        sys.exit(1)

    if verbose:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    server = create_mcp_tools_server()

    import asyncio

    async def _run():
        await server.run_stdio_async()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hermes tools MCP server")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()
    run_mcp_tools_server(verbose=args.verbose)
