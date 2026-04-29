# Claude CLI subprocess provider implementation plan

> **For Hermes:** implement an OpenClaw-style Claude CLI backend using `claude -p`, not ACP.

**Goal:** Add a Hermes provider that routes chat requests through the locally installed Claude Code CLI with subscription-backed login/session behavior.

**Architecture:** Register a new `claude-cli` provider as an external-process provider with base URL marker `claude-cli://local`. Add a `ClaudeCliClient` adapter that exposes `client.chat.completions.create(...)`, shells out to `claude -p --output-format stream-json`, parses JSONL events into OpenAI-like response/chunk objects, and preserves Claude session IDs across turns. Wire runtime resolution, model picker flow, and client creation in `run_agent.py`.

**Tech Stack:** Python, subprocess, Hermes provider registry/runtime resolver, Claude Code CLI `stream-json` output.

**Validation:** targeted pytest for provider registration, model persistence, client parsing, run_agent routing, then a live smoke test using `hermes chat` with `--provider claude-cli`.
