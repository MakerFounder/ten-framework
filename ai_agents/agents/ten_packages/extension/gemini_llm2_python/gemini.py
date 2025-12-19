#
# Gemini LLM2 - Native google-genai SDK with TRUE async
# Optimized for low-latency voice conversations
#
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncGenerator, List
from pydantic import BaseModel

from google import genai
from google.genai import types

from ten_ai_base.struct import (
    LLMMessageContent,
    LLMRequest,
    LLMResponse,
    LLMResponseMessageDelta,
    LLMResponseMessageDone,
    TextContent,
)
from ten_ai_base.types import LLMToolMetadata
from ten_runtime.async_ten_env import AsyncTenEnv


@dataclass
class GeminiLLM2Config(BaseModel):
    api_key: str = ""
    model: str = "gemini-3-flash-preview"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    prompt: str = "You are a helpful assistant."
    thinking_level: str = "OFF"
    base_url: str = ""
    black_list_params: List[str] = field(
        default_factory=lambda: ["messages", "tools", "stream", "model"]
    )

    def is_black_list_params(self, key: str) -> bool:
        return key in self.black_list_params


class GeminiChatAPI:
    def __init__(self, ten_env: AsyncTenEnv, config: GeminiLLM2Config):
        self.config = config
        self.ten_env = ten_env
        self.client = genai.Client(api_key=config.api_key)
        ten_env.log_info(
            f"GeminiChatAPI (native async) initialized: model={config.model}"
        )

    def _parse_message_content(self, message: LLMMessageContent) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        text_content = ""
        if isinstance(content, list):
            for part in content:
                if isinstance(part, TextContent):
                    text_content += part.text
                elif isinstance(part, dict) and part.get("type") == "text":
                    text_content += part.get("text", "")
        return text_content

    def _convert_messages_to_contents(self, messages) -> list:
        contents = []
        for message in messages or []:
            if isinstance(message, LLMMessageContent):
                role = message.role.lower()
                content = self._parse_message_content(message)
                if content:
                    gemini_role = "model" if role == "assistant" else "user"
                    contents.append(
                        types.Content(
                            role=gemini_role,
                            parts=[types.Part.from_text(text=content)]
                        )
                    )
        return contents

    async def get_chat_completions(
        self, request_input: LLMRequest
    ) -> AsyncGenerator[LLMResponse, None]:
        try:
            contents = self._convert_messages_to_contents(request_input.messages)
            
            # Dynamically add current date to system instruction
            current_date = datetime.now().strftime("%B %d, %Y")
            system_instruction = f"Today's date is {current_date}. {self.config.prompt}"
            
            generation_config = types.GenerateContentConfig(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_output_tokens=self.config.max_tokens,
                system_instruction=system_instruction,
            )

            self.ten_env.log_info(f"Requesting: model={self.config.model}")

            # Use native async streaming - no executor needed!
            full_content = ""
            async for chunk in await self.client.aio.models.generate_content_stream(
                model=self.config.model,
                contents=contents,
                config=generation_config,
            ):
                if chunk.text:
                    content = chunk.text
                    full_content += content
                    yield LLMResponseMessageDelta(
                        response_id="",
                        role="assistant",
                        content=full_content,
                        delta=content,
                        created=0,
                    )

            yield LLMResponseMessageDone(response_id="", role="assistant", content=full_content)

        except Exception as e:
            self.ten_env.log_error(f"Error in get_chat_completions: {e}")
            raise RuntimeError(f"CreateChatCompletion failed: {e}") from e
