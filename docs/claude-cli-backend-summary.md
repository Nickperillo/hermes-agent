# Making Claude CLI the Primary LLM in Hermes

## The Problem

Hermes was built around OpenAI-compatible API providers (GPT, OpenRouter, etc.). The goal was to use a Claude Max subscription as the primary backend instead of paying per-token Anthropic API costs. But Claude CLI isn't an API — it's a local binary with its own auth, session management, and tool system.

## Phase 1: Dead End — ACP Protocol

The first attempt tried to use `claude --acp --stdio` (the IDE integration protocol). Testing on the live machine showed ACP isn't exposed in the installed CLI version — `error: unknown option '--acp'`. That killed the "treat it like an API" approach.

## Phase 2: OpenClaw Pattern Discovery

We cloned and analysed OpenClaw (https://github.com/openclaw/openclaw), which had already solved this. Their approach: spawn `claude -p` as a subprocess, feed prompts via stdin, parse `stream-json` JSONL output. No ACP, no API keys -- just shell out to the CLI and let it use its own subscription auth.

Key details borrowed from OpenClaw:
- `--output-format stream-json` for structured streaming
- `--permission-mode bypassPermissions` for non-interactive use
- `--setting-sources user` to ignore repo-local config
- Clear all inherited Anthropic env vars so the subprocess uses its own login, not Hermes's API keys

## Phase 3: Building the Subprocess Provider

New files and changes:

1. **agent/claude_cli_client.py** -- The core adapter. Spawns `claude -p`, parses JSONL events (`message_start`, `content_block_delta`, `message_delta`, `result`), and presents an OpenAI-compatible `client.chat.completions.create()` interface so the rest of Hermes doesn't know the difference. Includes heartbeat chunks every 5s to prevent Hermes's stale-stream detector from killing long Claude thinking phases.

2. **hermes_cli/auth.py** -- Registered `claude-cli` as an `external_process` provider with default args (`-p`, `--output-format stream-json`, `--verbose`, `--permission-mode bypassPermissions`, etc.) and env var overrides.

3. **hermes_cli/providers.py** -- Added the provider overlay so Hermes's resolution system routes `claude-cli` requests correctly.

4. **run_agent.py** -- Added routing: if provider is `claude-cli` or base URL starts with `claude-cli://`, instantiate `ClaudeCliClient` instead of the normal OpenAI client. Also wires up the tool progress callback.

## Phase 4: The MCP Bridge -- Making Hermes Features Available

The hardest part. Claude CLI runs in its own sandbox -- it can't call Hermes tools (memory, skills, session search, todos) natively. Solution: expose Hermes tools as an MCP server that gets injected into the Claude CLI subprocess.

1. **mcp_tools_serve.py** -- A FastMCP server exposing 5 tools: `memory`, `skill_manage`, `skill_view`, `session_search`, `todo`. Each tool calls back into the real Hermes tool implementations.

2. **run_agent.py (`_build_claude_cli_mcp_config()`)** -- Builds a temporary JSON MCP config file containing the hermes-tools server definition (plus any user-configured MCP servers from `config.yaml`), then passes it to Claude CLI via `--mcp-config`.

This means when Claude CLI runs, it discovers hermes-tools as available MCP tools and can call them naturally.

## Phase 5: The Timing Bug Fix

The last piece: `tool_progress_callback` (which sends "tool started/completed" events to Discord/Telegram) was never reaching the Claude CLI client. The gateway assigns this callback after client creation, but the Claude CLI client was only checked at creation time. Fix in commit `e7b90159`: re-sync the callback in `_create_request_openai_client` so it's fresh on every request.

## Architecture Summary

```
User (Discord/Telegram)
  -> Hermes Gateway
    -> run_agent.py (detects claude-cli provider)
      -> ClaudeCliClient
        -> spawns `claude -p --output-format stream-json ...`
        -> injects MCP config (hermes-tools + shared-memory)
        -> clears Anthropic env vars (uses subscription auth)
        -> parses JSONL stream -> OpenAI-compatible chunks
        -> heartbeat keeps connection alive
        -> tool events forwarded to gateway for notifications
      -> response back through normal Hermes pipeline
```

## Net Result

Claude Opus 4.6 via Max subscription, with full access to Hermes memory, skills, session history, and task tracking -- zero API cost.
