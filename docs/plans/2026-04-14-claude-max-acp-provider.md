# Claude Max via Claude Code ACP for Hermes — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Let Hermes run against the local Claude Code CLI on a Claude Max subscription, with no Anthropic API spend, using a fresh ACP subprocess per request.

**Architecture:** Hermes already has most of the machinery needed through the existing `copilot-acp` path: runtime provider resolution, CLI/gateway model persistence, ACP subprocess spawning, and an OpenAI-compatible shim client. The clean path is to generalize that path into a provider-agnostic ACP subprocess backend, then add a `claude-code-acp` provider wired to `claude --acp --stdio`.

**Tech Stack:** Python, Hermes CLI/gateway/runtime provider system, existing ACP shim client in `agent/copilot_acp_client.py`, Claude Code CLI, pytest.

---

## Current findings from the codebase

### What already exists
- Hermes already supports a subprocess-backed provider: `copilot-acp`.
- Runtime provider resolution already carries:
  - `provider`
  - `api_mode`
  - `base_url`
  - `command`
  - `args`
- `AIAgent` already accepts `acp_command` / `acp_args`.
- `run_agent.py` already swaps from `OpenAI(...)` to a special ACP client when provider/base URL matches the ACP path.
- CLI model flow already has a dedicated provider flow for `copilot-acp`.
- Tests already exist proving `AIAgent` can initialize an ACP subprocess client.

### Important constraint
The current ACP path is hard-coded to **Copilot naming and markers** in several places:
- provider id: `copilot-acp`
- base URL marker: `acp://copilot`
- env vars: `HERMES_COPILOT_ACP_COMMAND`, `HERMES_COPILOT_ACP_ARGS`
- class name: `CopilotACPClient`

### Important product insight
OpenClaw needed `sessionMode: none` because it tried to resume stale CLI sessions.
Hermes appears **not** to have that problem on the existing ACP path because it already starts a **fresh subprocess per request**. The right Hermes equivalent is to **preserve per-request fresh ACP sessions** and explicitly avoid any session reuse layer for Claude Code.

---

## Desired user experience

Target setup should look roughly like this:

```bash
claude --version
claude auth login

hermes model --provider claude-code-acp
# choose claude-opus-4-6 or another Claude Code-supported model
```

And/or via env/config:

```bash
export HERMES_CLAUDE_ACP_COMMAND="/Users/cypher/.local/bin/claude"
export HERMES_CLAUDE_ACP_ARGS="--acp --stdio"
```

Target persisted config shape:

```yaml
model:
  provider: claude-code-acp
  default: claude-opus-4-6
  base_url: acp://claude-code
  api_mode: chat_completions
```

Optional future sugar:
- `hermes auth add claude-code-acp`
- `hermes model` first-class menu entry for Claude Code ACP
- `hermes doctor` validation for Claude CLI presence/auth

---

## Non-goals
- Do **not** build session persistence/resume for Claude ACP.
- Do **not** route through Anthropic API keys for this mode.
- Do **not** make this Claude-only in the underlying architecture if a provider-agnostic ACP abstraction is easy.
- Do **not** break the current `copilot-acp` path while adding Claude support.

---

## Task 1: Document the exact integration seam before editing

**Objective:** Confirm the minimum file set and current hard-coded assumptions for ACP subprocess providers.

**Files:**
- Read: `hermes_cli/auth.py`
- Read: `hermes_cli/runtime_provider.py`
- Read: `hermes_cli/providers.py`
- Read: `run_agent.py`
- Read: `agent/copilot_acp_client.py`
- Read: `hermes_cli/main.py`
- Read: `tests/run_agent/test_run_agent.py`

**Step 1: Record current hard-coded Copilot assumptions**
Make a checklist of:
- provider ids
- base URL markers
- env var names
- help text / UX text
- any Copilot-specific error messages

**Step 2: Confirm per-request subprocess lifecycle**
Trace `CopilotACPClient` request flow and verify it launches a short-lived ACP subprocess per request instead of reusing sessions.

**Step 3: Confirm gateway and cron use the same runtime provider path**
Verify `cli.py`, `gateway/run.py`, and `cron/scheduler.py` all consume `resolve_runtime_provider(...)` so one provider implementation covers all three.

**Verification:**
- Produce a short implementation note in the PR or commit message describing every hard-coded Copilot assumption that must be generalized.

---

## Task 2: Generalize the ACP shim client

**Objective:** Convert the Copilot-specific ACP client into a provider-agnostic subprocess ACP client without changing behavior.

**Files:**
- Rename/create: `agent/copilot_acp_client.py` → `agent/acp_subprocess_client.py` (or keep old file as thin wrapper)
- Modify: `run_agent.py`
- Modify: `agent/auxiliary_client.py`
- Test: `tests/run_agent/test_run_agent.py`

**Step 1: Create a provider-neutral class name**
Preferred:
```python
class ACPSubprocessClient:
    ...
```

**Step 2: Remove Copilot-only constants from the client core**
Replace:
- `ACP_MARKER_BASE_URL = "acp://copilot"`
- `HERMES_COPILOT_ACP_COMMAND`
- `HERMES_COPILOT_ACP_ARGS`

with constructor-driven values and provider-neutral defaults.

**Step 3: Keep backward compatibility**
If needed, keep:
```python
CopilotACPClient = ACPSubprocessClient
```
so existing imports/tests do not explode mid-refactor.

**Step 4: Generalize run-agent ACP detection**
Replace `copilot-acp` / `acp://copilot` checks with something like:
- provider overlay says `auth_type == external_process`, or
- base URL starts with `acp://`

**Step 5: Update tests**
Keep the existing Copilot test passing, then add one more generic ACP test.

**Verification:**
Run:
```bash
source venv/bin/activate
python -m pytest tests/run_agent/test_run_agent.py -q
```
Expected:
- existing ACP test passes
- no Copilot regression

---

## Task 3: Add a Claude Code ACP provider to the provider registry

**Objective:** Register a new first-class runtime provider for Claude Code CLI.

**Files:**
- Modify: `hermes_cli/auth.py`
- Modify: `hermes_cli/providers.py`
- Modify: `hermes_cli/runtime_provider.py`
- Test: `tests/hermes_cli/test_api_key_providers.py` or a new focused test file

**Step 1: Add provider config in `hermes_cli/auth.py`**
Add a new provider similar to `copilot-acp`:
```python
"claude-code-acp": ProviderConfig(
    id="claude-code-acp",
    name="Claude Code ACP",
    auth_type="external_process",
    inference_base_url="acp://claude-code",
    base_url_env_var="CLAUDE_CODE_ACP_BASE_URL",
)
```

**Step 2: Add Hermes overlay in `hermes_cli/providers.py`**
Add:
```python
"claude-code-acp": HermesOverlay(
    transport="codex_responses",
    auth_type="external_process",
    base_url_override="acp://claude-code",
    base_url_env_var="CLAUDE_CODE_ACP_BASE_URL",
)
```

**Step 3: Add runtime resolution branch in `hermes_cli/runtime_provider.py`**
Mirror the `copilot-acp` branch, but use Claude-specific command/args env vars.

Preferred env vars:
- `HERMES_CLAUDE_ACP_COMMAND`
- `HERMES_CLAUDE_ACP_ARGS`
- optional fallback: `CLAUDE_CODE_CLI_PATH`

Runtime shape should be:
```python
{
  "provider": "claude-code-acp",
  "api_mode": "chat_completions",
  "base_url": "acp://claude-code",
  "api_key": "claude-code-acp",
  "command": "/absolute/path/to/claude",
  "args": ["--acp", "--stdio"],
  "source": "process",
}
```

**Step 4: Add auth/status helper support**
Either:
- generalize `resolve_external_process_provider_credentials()` to dispatch by provider id, or
- create a small provider-to-env-var map so both Copilot and Claude reuse the same helper.

**Verification:**
Run:
```bash
source venv/bin/activate
python -m pytest tests/hermes_cli/test_api_key_providers.py -q
```
Expected:
- Claude ACP status/credential resolution is covered by tests
- Copilot ACP tests still pass

---

## Task 4: Add first-class CLI model flow for Claude Code ACP

**Objective:** Make `hermes model` and provider switching support Claude Code ACP as a first-class option.

**Files:**
- Modify: `hermes_cli/main.py`
- Modify: `hermes_cli/models.py` if needed
- Modify: `hermes_cli/model_switch.py` if needed
- Test: `tests/hermes_cli/test_model_provider_persistence.py`

**Step 1: Add provider choice**
Ensure `claude-code-acp` appears wherever provider choices are listed.

**Step 2: Add a dedicated model flow**
Follow the existing `copilot-acp` flow but tailored to Claude Code:
- explain that Hermes delegates turns to `claude --acp --stdio`
- explain that Hermes currently creates a fresh ACP subprocess per request
- validate the `claude` command exists
- persist:
  - `model.provider = claude-code-acp`
  - `model.base_url = acp://claude-code`
  - `model.api_mode = chat_completions`

**Step 3: Model selection behavior**
Start simple:
- allow manual model entry
- seed suggested models like:
  - `claude-opus-4-6`
  - `claude-sonnet-4-6`
  - whatever Claude Code actually accepts as hints

Do **not** block implementation on a remote model catalog if Claude CLI does not expose one cleanly.

**Verification:**
Run:
```bash
source venv/bin/activate
python -m pytest tests/hermes_cli/test_model_provider_persistence.py -q
```
Expected:
- selected provider/model persists correctly
- no regression for Copilot ACP

---

## Task 5: Add provider-specific env vars, doctor checks, and UX text

**Objective:** Make setup discoverable and diagnosable.

**Files:**
- Modify: `hermes_cli/config.py`
- Modify: `hermes_cli/main.py`
- Modify: `website/docs/integrations/providers.md`
- Modify: `website/docs/reference/environment-variables.md`

**Step 1: Add env var metadata**
Add optional env vars such as:
- `HERMES_CLAUDE_ACP_COMMAND`
- `HERMES_CLAUDE_ACP_ARGS`
- `CLAUDE_CODE_ACP_BASE_URL`
- maybe `CLAUDE_CODE_CLI_PATH`

**Step 2: Add doctor/status messaging**
Expected checks:
- can Hermes resolve the `claude` binary?
- does `claude --version` succeed?
- if not configured, print exact remediation:
  - `claude auth login`
  - set `HERMES_CLAUDE_ACP_COMMAND`

**Step 3: Add docs snippet**
Document the no-API-cost setup flow:
```bash
claude --version
claude auth login
export HERMES_CLAUDE_ACP_COMMAND="$(which claude)"
export HERMES_CLAUDE_ACP_ARGS="--acp --stdio"
hermes model --provider claude-code-acp
```

**Verification:**
- `hermes doctor` output is understandable when Claude CLI is missing
- docs include one complete copy-pasteable setup path

---

## Task 6: Add end-to-end tests for Claude ACP provider resolution

**Objective:** Prevent regressions and prove that Hermes can construct the ACP-backed client route for Claude.

**Files:**
- Modify/create: `tests/run_agent/test_run_agent.py`
- Modify/create: `tests/hermes_cli/test_api_key_providers.py`
- Modify/create: `tests/hermes_cli/test_model_provider_persistence.py`

**Step 1: Runtime provider resolution test**
Mock env:
```python
HERMES_CLAUDE_ACP_COMMAND=/usr/local/bin/claude
HERMES_CLAUDE_ACP_ARGS="--acp --stdio"
```
Assert:
- provider resolves as `claude-code-acp`
- base_url is `acp://claude-code`
- args are parsed correctly

**Step 2: AIAgent client selection test**
Instantiate `AIAgent` with:
```python
provider="claude-code-acp"
base_url="acp://claude-code"
acp_command="/usr/local/bin/claude"
acp_args=["--acp", "--stdio"]
```
Assert:
- Hermes uses the ACP subprocess client
- it does not construct a normal OpenAI client

**Step 3: Backward compatibility test**
Re-run/copied assertion for `copilot-acp`.

**Verification:**
Run:
```bash
source venv/bin/activate
python -m pytest tests/run_agent/test_run_agent.py tests/hermes_cli/test_api_key_providers.py tests/hermes_cli/test_model_provider_persistence.py -q
```
Expected:
- all Claude ACP tests pass
- all existing Copilot ACP tests pass

---

## Task 7: Manual local validation on this machine

**Objective:** Prove the feature works with the real Claude CLI and behaves like a fresh-session backend.

**Files:**
- No code changes required
- Use local config at `~/.hermes/config.yaml` and `~/.hermes/.env`

**Step 1: Confirm Claude CLI is installed and logged in**
Run:
```bash
which claude
claude --version
claude auth login
```

**Step 2: Configure Hermes for Claude ACP**
Either via config or environment:
```bash
export HERMES_CLAUDE_ACP_COMMAND="$(which claude)"
export HERMES_CLAUDE_ACP_ARGS="--acp --stdio"
```
Then set model/provider:
```bash
hermes model --provider claude-code-acp
```

**Step 3: Run simple smoke tests**
```bash
hermes chat -q "Say hello in one sentence."
hermes chat -q "What tools do you have?"
```

**Step 4: Run a tool-using test**
Use a safe local tool call, e.g.:
```bash
hermes chat -q "Read the first 3 lines of README.md and summarize them."
```

**Step 5: Verify no stale-session behavior**
Send several back-to-back queries and confirm there is no alternating crash pattern.

**Verification:**
Success means:
- Hermes answers through Claude CLI
- tools still work
- repeated turns do not fail due to stale ACP session reuse

---

## Task 8: Optional follow-up — generalize to a reusable ACP provider framework

**Objective:** Avoid one provider-specific codepath per ACP CLI.

**Files:**
- `hermes_cli/auth.py`
- `hermes_cli/providers.py`
- `hermes_cli/runtime_provider.py`
- `agent/acp_subprocess_client.py`

**Approach:**
If the Claude implementation lands cleanly, consider a second pass where ACP providers are declared by metadata instead of hardcoded branches. Example shape:

```python
EXTERNAL_PROCESS_PROVIDERS = {
  "copilot-acp": {
    "command_env": "HERMES_COPILOT_ACP_COMMAND",
    "args_env": "HERMES_COPILOT_ACP_ARGS",
    "default_command": "copilot",
    "default_args": ["--acp", "--stdio"],
    "base_url": "acp://copilot",
  },
  "claude-code-acp": {
    "command_env": "HERMES_CLAUDE_ACP_COMMAND",
    "args_env": "HERMES_CLAUDE_ACP_ARGS",
    "default_command": "claude",
    "default_args": ["--acp", "--stdio"],
    "base_url": "acp://claude-code",
  },
}
```

This is optional for the first shipping pass but likely the right architecture.

---

## Risks and mitigations

### Risk 1: Claude CLI ACP protocol differs from Copilot ACP behavior
**Mitigation:** keep the ACP client generic but add a very small Claude-specific normalization layer only if real testing shows a mismatch.

### Risk 2: Claude model hints are ignored or require different names
**Mitigation:** treat the selected Hermes model as a best-effort hint first; do not block the provider on perfect model catalog support.

### Risk 3: Existing `copilot-acp` support regresses
**Mitigation:** preserve backward-compatible tests and aliases during the refactor.

### Risk 4: Gateway / cron path differs from CLI path in subtle ways
**Mitigation:** validate through `resolve_runtime_provider(...)` and one smoke test in CLI first, then gateway.

---

## Acceptance criteria

This feature is complete when all of the following are true:
- Hermes has a selectable provider named `claude-code-acp`.
- Hermes can resolve `claude --acp --stdio` as a subprocess backend.
- Hermes uses a fresh ACP subprocess per request.
- Hermes can complete at least one safe tool-using request through Claude ACP.
- `copilot-acp` still works.
- setup/docs clearly explain how to use Claude Max with Hermes without API spend.

---

## Recommended implementation order

1. Generalize ACP client naming / detection
2. Add `claude-code-acp` provider registry entries
3. Add runtime resolution and env vars
4. Add CLI model flow
5. Add tests
6. Run manual validation with real Claude CLI
7. Document setup

---

## Copy-paste target setup for the finished feature

Once implemented, the user flow should be as simple as:

```bash
claude --version
claude auth login

export HERMES_CLAUDE_ACP_COMMAND="$(which claude)"
export HERMES_CLAUDE_ACP_ARGS="--acp --stdio"

hermes model --provider claude-code-acp
hermes chat -q "Hello from Claude Max through Hermes"
```

And unlike OpenClaw, Hermes should **not** need a `sessionMode: none` workaround because the ACP process lifecycle should remain **one fresh subprocess per request**.
