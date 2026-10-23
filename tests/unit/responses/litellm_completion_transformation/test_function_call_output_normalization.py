"""
Tests for normalizing Responses API function_call_output into chat tool messages.

This is important for Gemini/Vertex, which expects tool results to be represented
as tool/function response parts; if the tool output is passed as a list of input_* parts,
we normalize it to text/image blocks or a string.
"""

from copy import deepcopy
from typing import Final

import pytest

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def test_function_call_output_list_input_text_is_converted_to_tool_string_content():
    out = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output={
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [
                {"type": "input_text", "text": "hello"},
                {"type": "input_text", "text": " world"},
            ],
        }
    )

    assert len(out) == 1
    msg = out[0]
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_1"
    assert msg["content"] == "hello world"


def test_function_call_output_string_passthrough():
    out = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output={
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"ok":true}',
        }
    )
    assert len(out) == 1
    assert out[0]["content"] == '{"ok":true}'


@pytest.mark.parametrize("kind", ["function_call_output", "custom_tool_call_output"])
@pytest.mark.parametrize("detail", ["low", "high", "original"])
@pytest.mark.parametrize("nested_url", [False, True])
def test_tool_output_preserves_image_detail_and_adjacent_text(
    kind: str, detail: str, nested_url: bool,
) -> None:
    image: Final = {
        "type": "input_image",
        "image_url": {"url": "data:image/png;base64,AAAA", "detail": detail}
        if nested_url else "data:image/png;base64,AAAA",
        **({} if nested_url else {"detail": detail}),
    }
    request: Final = {"type": kind, "call_id": "call_media", "output": [
        {"type": "input_text", "text": "Screenshot follows"}, image,
    ]}
    original: Final = deepcopy(request)
    messages: Final = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(request)
    assert messages[0]["content"] == [
        {"type": "text", "text": "Screenshot follows"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": detail}},
    ]
    assert request == original


def test_computer_screenshot_output_remains_structured_media() -> None:
    messages: Final = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message({
        "type": "computer_call_output", "call_id": "call_screenshot",
        "output": {"type": "computer_screenshot", "image_url": "data:image/png;base64,AAAA"},
    })
    assert messages[0]["content"] == [{
        "type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "auto"},
    }]


@pytest.mark.parametrize("tool_output", [False, True])
@pytest.mark.parametrize("mime_field", ["format", "mime_type", "content_type"])
def test_responses_audio_and_file_mime_survive_normalization(tool_output: bool, mime_field: str) -> None:
    parts: Final = [
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        {"type": "input_file", "file_data": "data:audio/flac;base64,BBBB", mime_field: "audio/flac"},
    ]
    result: Final = (
        LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message({
            "type": "custom_tool_call_output", "call_id": "call_audio", "output": parts,
        })[0]["content"]
        if tool_output else LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(parts)
    )
    assert result == [
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        {"type": "file", "file": {"file_data": "data:audio/flac;base64,BBBB", "format": "audio/flac"}},
    ]
