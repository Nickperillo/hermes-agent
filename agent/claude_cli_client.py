"""OpenAI-compatible shim that forwards Hermes requests to `claude -p`.

This adapter lets Hermes treat Claude Code CLI as a chat-style backend without
relying on Anthropic's direct API billing path. Each request shells out to the
local Claude Code binary, streams JSONL events, and converts the result back
into the minimal OpenAI-shaped response/chunk objects Hermes expects.

Unlike ACP backends, Claude CLI owns its own tool execution. Hermes tool schemas
are not translated into OpenAI-style tool calls here; instead Claude Code uses
its native local tools / MCP integrations while Hermes treats the final answer
as the assistant response.
"""

from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

_DEFAULT_TIMEOUT_SECONDS = 900.0
# Interval between empty-choices "heartbeat" chunks emitted while the
# subprocess is working but hasn't produced text yet.  Keeps Hermes'
# stale-stream detector (default 180s) from firing during long Claude
# thinking/tool phases before the first text delta.
_HEARTBEAT_INTERVAL_SECONDS = 5.0
CLAUDE_CLI_MARKER_BASE_URL = "claude-cli://local"

# Clear inherited routing/auth overrides so spawned Claude runs use the host's
# own Claude login state rather than Hermes/OpenRouter/API env.
_CLAUDE_CLI_CLEAR_ENV = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_API_KEY_OLD",
    "ANTHROPIC_API_TOKEN",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_OAUTH_TOKEN",
    "ANTHROPIC_UNIX_SOCKET",
    "CLAUDE_CONFIG_DIR",
    "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_OAUTH_REFRESH_TOKEN",
    "CLAUDE_CODE_OAUTH_SCOPES",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
    "CLAUDE_CODE_PLUGIN_CACHE_DIR",
    "CLAUDE_CODE_PLUGIN_SEED_DIR",
    "CLAUDE_CODE_REMOTE",
    "CLAUDE_CODE_USE_COWORK_PLUGINS",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_VERTEX",
}


def _coerce_timeout_seconds(timeout: Any) -> float:
    if isinstance(timeout, (int, float)):
        try:
            return max(1.0, float(timeout))
        except Exception:
            pass
    return _DEFAULT_TIMEOUT_SECONDS


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"].strip()
        if isinstance(content.get("content"), str):
            return content["content"].strip()
        if content.get("type") == "image_url":
            image = content.get("image_url") or {}
            if isinstance(image, dict) and isinstance(image.get("url"), str):
                return f"[image] {image['url']}"
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            rendered = _render_message_content(item)
            if rendered:
                parts.append(rendered)
        return "\n".join(parts).strip()
    return str(content).strip()


def _format_messages_as_prompt(messages: list[dict[str, Any]], model: str | None = None) -> tuple[str, str]:
    system_parts: list[str] = []
    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").strip().lower()
        rendered = _render_message_content(message.get("content"))
        if not rendered:
            continue
        if role == "system":
            system_parts.append(rendered)
            continue
        label = {
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
            "function": "Tool",
        }.get(role, role.title())
        transcript.append(f"{label}:\n{rendered}")

    prompt_sections: list[str] = []
    if model:
        prompt_sections.append(f"Hermes requested model hint: {model}")
    prompt_sections.append(
        "Continue the conversation from the latest user request. "
        "You are running inside Hermes as a Claude CLI backend. "
        "Use native Claude Code capabilities if needed, but return only the final assistant answer."
    )
    prompt_sections.append(
        "You have Hermes self-improvement tools available via MCP (hermes-tools server): "
        "memory (persistent curated memory — add/replace/remove), "
        "skill_manage (create/edit/patch/delete reusable skills), "
        "skill_view (read skill content and linked files), "
        "session_search (search past conversation transcripts), "
        "and todo (session task list). "
        "Use these proactively: save important facts to memory, search past sessions for context, "
        "and create/update skills when you learn reusable procedures."
    )
    if transcript:
        prompt_sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))
    return "\n\n".join(prompt_sections).strip(), "\n\n".join(system_parts).strip()


def _make_delta_chunk(text: str, model: str | None = None):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content=text, tool_calls=None, reasoning_content=None),
            finish_reason=None,
        )],
        usage=None,
    )


def _make_finish_chunk(finish_reason: str = "stop", model: str | None = None):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content=None, tool_calls=None, reasoning_content=None),
            finish_reason=finish_reason,
        )],
        usage=None,
    )


def _make_usage_chunk(usage: Any, model: str | None = None):
    return SimpleNamespace(model=model, choices=[], usage=usage)


class _ClaudeCliChatCompletions:
    def __init__(self, client: "ClaudeCliClient"):
        self._client = client

    def create(self, **kwargs):
        return self._client._create_chat_completion(**kwargs)


class _ClaudeCliChatNamespace:
    def __init__(self, client: "ClaudeCliClient"):
        self.completions = _ClaudeCliChatCompletions(client)


class ClaudeCliClient:
    """Minimal OpenAI-client-compatible facade for `claude -p`."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        timeout: Any = None,
        default_headers: dict[str, Any] | None = None,
        mcp_config_path: str | None = None,
        **_: Any,
    ) -> None:
        self.api_key = api_key or "claude-cli"
        self.base_url = base_url or CLAUDE_CLI_MARKER_BASE_URL
        self.command = command or os.getenv("HERMES_CLAUDE_CLI_COMMAND", "").strip() or os.getenv("CLAUDE_CODE_CLI_PATH", "").strip() or "claude"
        self.args = list(args or [])
        self.timeout = _coerce_timeout_seconds(timeout)
        self.default_headers = dict(default_headers or {})
        self.mcp_config_path = mcp_config_path
        self.chat = _ClaudeCliChatNamespace(self)
        self._lock = threading.Lock()
        self._resume_session_id: str | None = None
        self.is_closed = False
        self._active_procs: list[subprocess.Popen] = []
        self._procs_lock = threading.Lock()
        self.on_tool_event: Any = None

    def _register_proc(self, proc: subprocess.Popen) -> None:
        with self._procs_lock:
            self._active_procs.append(proc)

    def _unregister_proc(self, proc: subprocess.Popen) -> None:
        with self._procs_lock:
            try:
                self._active_procs.remove(proc)
            except ValueError:
                pass

    def close(self) -> None:
        self.is_closed = True
        with self._procs_lock:
            procs = list(self._active_procs)
        for p in procs:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        for key in _CLAUDE_CLI_CLEAR_ENV:
            env.pop(key, None)
        return env

    def _build_command(self, *, model: str | None, system_prompt: str, stream: bool) -> list[str]:
        cmd = [self.command]
        cmd.extend(self.args or [
            "-p",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--verbose",
            "--setting-sources",
            "user",
            "--permission-mode",
            "bypassPermissions",
        ])
        if model:
            cmd.extend(["--model", str(model)])
        if system_prompt:
            cmd.extend(["--append-system-prompt", system_prompt])
        if self.mcp_config_path:
            cmd.extend(["--mcp-config", self.mcp_config_path])
        # MVP: stay stateless at the Hermes layer and let the full transcript be
        # the source of truth. Claude's returned session_id is still captured for
        # future evolution/debugging, but we intentionally do not pass --resume
        # yet because Hermes already re-sends full history each turn.
        return cmd

    def _run_claude_stream(
        self,
        *,
        prompt: str,
        model: str | None,
        system_prompt: str,
        timeout_seconds: float,
        on_text_delta=None,
    ) -> dict[str, Any]:
        cmd = self._build_command(model=model, system_prompt=system_prompt, stream=on_text_delta is not None)
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd(),
            env=self._build_env(),
            bufsize=1,
        )
        self._register_proc(proc)

        assert proc.stdin is not None
        assert proc.stdout is not None
        assert proc.stderr is not None

        session_id = None
        model_name = None
        text_parts: list[str] = []
        finish_reason = "stop"
        usage = None
        stderr_lines: list[str] = []
        watchdog_fired = {"killed": False}

        def _drain_stderr():
            for line in proc.stderr:
                stderr_lines.append(line.rstrip("\n"))

        def _write_stdin():
            # Large prompts can exceed the pipe buffer (~64KB on macOS)
            # and deadlock if written synchronously before stdout is drained.
            try:
                proc.stdin.write(prompt)
            except Exception:
                pass
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

        def _watchdog():
            try:
                if proc.poll() is None:
                    watchdog_fired["killed"] = True
                    proc.kill()
            except Exception:
                pass

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()
        stdin_thread = threading.Thread(target=_write_stdin, daemon=True)
        stdin_thread.start()
        watchdog = threading.Timer(max(1.0, timeout_seconds), _watchdog)
        watchdog.daemon = True
        watchdog.start()

        try:
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                event_type = event.get("type")
                if event.get("session_id"):
                    session_id = event.get("session_id")
                if event_type == "stream_event":
                    payload = event.get("event") or {}
                    inner_type = payload.get("type")
                    if inner_type == "message_start":
                        message = payload.get("message") or {}
                        model_name = message.get("model") or model_name
                    elif inner_type == "content_block_delta":
                        delta = payload.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            text = delta.get("text") or ""
                            if text:
                                text_parts.append(text)
                                if on_text_delta is not None:
                                    on_text_delta(text, model_name)
                    elif inner_type == "message_delta":
                        delta = payload.get("delta") or {}
                        finish_reason = delta.get("stop_reason") or finish_reason
                        usage = payload.get("usage") or usage
                elif event_type == "assistant":
                    message = event.get("message") or {}
                    model_name = message.get("model") or model_name
                    if self.on_tool_event:
                        for block in (message.get("content") or []):
                            if isinstance(block, dict) and block.get("type") == "tool_use":
                                tool_name = block.get("name", "")
                                tool_input = block.get("input") or {}
                                preview = json.dumps(tool_input, ensure_ascii=False)[:200]
                                try:
                                    self.on_tool_event("tool.started", tool_name, preview, tool_input)
                                except Exception:
                                    pass
                elif event_type == "user":
                    if self.on_tool_event:
                        for block in ((event.get("message") or {}).get("content") or []):
                            if isinstance(block, dict) and block.get("type") == "tool_result":
                                tool_id = block.get("tool_use_id", "")
                                try:
                                    self.on_tool_event("tool.completed", tool_id, None, None)
                                except Exception:
                                    pass
                elif event_type == "result":
                    model_usage = event.get("modelUsage") or {}
                    if not model_name and isinstance(model_usage, dict) and model_usage:
                        model_name = next(iter(model_usage.keys()), None)
                    session_id = event.get("session_id") or session_id
                    finish_reason = event.get("stop_reason") or finish_reason
                    usage = event.get("usage") or usage
                    result_text = event.get("result")
                    if isinstance(result_text, str):
                        joined = "".join(text_parts)
                        if result_text and result_text != joined:
                            # If the stream did not include visible text deltas
                            # (e.g. tool-heavy run), prefer the final result.
                            text_parts = [result_text]
                    break
            try:
                proc.wait(timeout=max(1.0, timeout_seconds))
            except subprocess.TimeoutExpired:
                watchdog_fired["killed"] = True
                proc.kill()
                raise TimeoutError(f"Claude CLI timed out after {int(timeout_seconds)}s")
        finally:
            watchdog.cancel()
            try:
                proc.stdout.close()
            except Exception:
                pass
            stderr_thread.join(timeout=1.0)
            stdin_thread.join(timeout=1.0)
            self._unregister_proc(proc)

        if watchdog_fired["killed"]:
            raise TimeoutError(f"Claude CLI timed out after {int(timeout_seconds)}s")
        if self.is_closed and proc.returncode not in (0, None):
            raise RuntimeError("Claude CLI aborted: client closed")
        if proc.returncode not in (0, None):
            err = "\n".join(stderr_lines[-20:]).strip() or f"exit code {proc.returncode}"
            raise RuntimeError(f"Claude CLI failed: {err}")

        self._resume_session_id = session_id or self._resume_session_id
        return {
            "text": "".join(text_parts).strip(),
            "session_id": session_id,
            "model": model_name or model,
            "finish_reason": finish_reason or "stop",
            "usage": usage,
        }

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        stream: bool = False,
        timeout: Any = None,
        **_: Any,
    ):
        prompt, system_prompt = _format_messages_as_prompt(messages or [], model=model)
        timeout_seconds = _coerce_timeout_seconds(timeout)

        if stream:
            def _generator() -> Iterable[Any]:
                # Run the subprocess in a worker thread and stream deltas
                # through a queue so the outer Hermes stream loop sees
                # chunks in real time.  Previously this collected everything
                # into a list before yielding, which made the 180s stale-
                # stream detector fire on long Claude turns.
                q: "queue.Queue[Any]" = queue.Queue()
                sentinel = object()
                holder: dict[str, Any] = {"result": None, "error": None}

                def _on_text_delta(text: str, model_name: str | None):
                    if text:
                        q.put(_make_delta_chunk(text, model_name or model))

                def _worker():
                    try:
                        holder["result"] = self._run_claude_stream(
                            prompt=prompt,
                            model=model,
                            system_prompt=system_prompt,
                            timeout_seconds=timeout_seconds,
                            on_text_delta=_on_text_delta,
                        )
                    except Exception as exc:
                        holder["error"] = exc
                    finally:
                        q.put(sentinel)

                worker = threading.Thread(target=_worker, daemon=True)
                worker.start()

                any_delta = False
                while True:
                    try:
                        item = q.get(timeout=_HEARTBEAT_INTERVAL_SECONDS)
                    except queue.Empty:
                        # Empty-choices chunk keeps the outer stale-stream
                        # timer fresh without surfacing anything to the user.
                        yield _make_usage_chunk(None, model)
                        continue
                    if item is sentinel:
                        break
                    any_delta = True
                    yield item

                if holder["error"] is not None:
                    raise holder["error"]

                result = holder["result"] or {}
                if not any_delta and result.get("text"):
                    yield _make_delta_chunk(result["text"], result.get("model") or model)
                if result.get("usage"):
                    yield _make_usage_chunk(result.get("usage"), result.get("model") or model)
                yield _make_finish_chunk(result.get("finish_reason") or "stop", result.get("model") or model)

            return _generator()

        result = self._run_claude_stream(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            timeout_seconds=timeout_seconds,
            on_text_delta=None,
        )
        message = SimpleNamespace(
            role="assistant",
            content=result.get("text") or None,
            tool_calls=None,
            reasoning_content=None,
        )
        choice = SimpleNamespace(index=0, message=message, finish_reason=result.get("finish_reason") or "stop")
        return SimpleNamespace(
            id="claude-cli-" + str(uuid.uuid4()),
            model=result.get("model") or model,
            choices=[choice],
            usage=result.get("usage"),
            session_id=result.get("session_id"),
        )
