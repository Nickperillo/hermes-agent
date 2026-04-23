from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agent.claude_cli_client import ClaudeCliClient


def test_client_exposes_openai_chat_interface():
    client = ClaudeCliClient(command="claude", args=["-p"])
    assert hasattr(client, "chat")
    assert hasattr(client.chat, "completions")
    assert hasattr(client.chat.completions, "create")


@patch.object(ClaudeCliClient, "_run_claude_stream")
def test_non_stream_completion_uses_result_text(mock_run):
    mock_run.return_value = {
        "text": "OK",
        "session_id": "sess-1",
        "model": "claude-opus-4-7",
        "finish_reason": "end_turn",
        "usage": {"output_tokens": 2},
    }
    client = ClaudeCliClient(command="claude", args=["-p"])

    response = client.chat.completions.create(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": "Reply exactly OK"}],
    )

    assert response.choices[0].message.content == "OK"
    assert response.choices[0].finish_reason == "end_turn"
    assert response.session_id == "sess-1"


@patch.object(ClaudeCliClient, "_run_claude_stream")
def test_streaming_completion_emits_delta_and_finish(mock_run):
    def fake_run(**kwargs):
        kwargs["on_text_delta"]("Hel", "claude-sonnet-4-6")
        kwargs["on_text_delta"]("lo", "claude-sonnet-4-6")
        return {
            "text": "Hello",
            "session_id": "sess-2",
            "model": "claude-sonnet-4-6",
            "finish_reason": "end_turn",
            "usage": {"output_tokens": 5},
        }

    mock_run.side_effect = fake_run
    client = ClaudeCliClient(command="claude", args=["-p"])

    chunks = list(
        client.chat.completions.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Say hello"}],
            stream=True,
        )
    )

    assert chunks[0].choices[0].delta.content == "Hel"
    assert chunks[1].choices[0].delta.content == "lo"
    assert any(getattr(chunk, "usage", None) for chunk in chunks)
    assert chunks[-1].choices[0].finish_reason == "end_turn"


def test_prompt_formatter_keeps_system_prompt_separate():
    client = ClaudeCliClient(command="claude", args=["-p"])
    with patch.object(client, "_run_claude_stream", return_value={"text": "ok", "session_id": None, "model": "claude-opus-4-7", "finish_reason": "end_turn", "usage": None}) as mock_run:
        client.chat.completions.create(
            model="claude-opus-4-7",
            messages=[
                {"role": "system", "content": "System rule"},
                {"role": "user", "content": "Hi"},
            ],
        )
    assert mock_run.call_args.kwargs["system_prompt"] == "System rule"
    assert "Conversation transcript" in mock_run.call_args.kwargs["prompt"]
