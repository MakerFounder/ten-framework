import asyncio
import aiohttp
import base64
import json
from typing import Callable, Awaitable, Tuple, Optional

from ten_ai_base.message import (
    ModuleError,
    ModuleErrorCode,
)
from ten_runtime import AsyncTenEnv
from .config import InworldTTSConfig


class InworldTTSClient:
    """Inworld AI TTS Client using streaming API."""

    def __init__(
        self,
        config: InworldTTSConfig,
        ten_env: AsyncTenEnv,
        error_callback: Callable[[str, ModuleError], Awaitable[None]] = None,
        response_msgs: asyncio.Queue[Tuple[bytes, bool, str]] = None,
    ) -> None:
        self.config = config
        self.ten_env = ten_env
        self.error_callback = error_callback
        self.response_msgs = response_msgs
        self.text_input_queue: asyncio.Queue = asyncio.Queue()
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._process_task: Optional[asyncio.Task] = None
        self.cur_request_id = ""

    async def start(self) -> None:
        """Start the TTS client."""
        self._running = True
        # Create session with larger buffer limits
        connector = aiohttp.TCPConnector(limit=10)
        self._session = aiohttp.ClientSession(
            connector=connector,
            read_bufsize=1024 * 1024  # 1MB buffer
        )
        self._process_task = asyncio.create_task(self._process_loop())
        self.ten_env.log_info("Inworld TTS client started")

    async def close(self) -> None:
        """Stop the TTS client."""
        self._running = False
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()
        self.ten_env.log_info("Inworld TTS client stopped")

    async def _process_loop(self) -> None:
        """Main processing loop for TTS requests."""
        while self._running:
            try:
                try:
                    text_input = await asyncio.wait_for(
                        self.text_input_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                if text_input is None:
                    continue

                text = text_input.get("text", "")
                request_id = text_input.get("request_id", "")
                is_end = text_input.get("is_end", False)

                if not text and not is_end:
                    continue

                self.cur_request_id = request_id

                if text:
                    await self._synthesize_stream(text, request_id)

                if is_end:
                    if self.response_msgs:
                        await self.response_msgs.put((b"", True, request_id))

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.ten_env.log_error(f"Error in TTS process loop: {e}")
                if self.error_callback:
                    await self.error_callback(
                        self.cur_request_id,
                        ModuleError(
                            message=str(e),
                            code=ModuleErrorCode.RUNTIME_ERROR,
                        ),
                    )

    async def _synthesize_stream(self, text: str, request_id: str) -> None:
        """Synthesize text to speech using Inworld streaming API."""
        if not self._session:
            return

        url = f"{self.config.base_url}/tts/v1/voice:stream"
        
        headers = {
            "Authorization": f"Basic {self.config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "text": text,
            "voiceId": self.config.voice_id,
            "modelId": self.config.model_id,
            "audio_config": {
                "audio_encoding": "LINEAR16",
                "sample_rate_hertz": self.config.sample_rate,
            },
        }

        self.ten_env.log_info(f"Inworld TTS request: voice={self.config.voice_id}, text_len={len(text)}")

        try:
            async with self._session.post(
                url, headers=headers, json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.ten_env.log_error(
                        f"Inworld TTS API error: {response.status} - {error_text}"
                    )
                    if self.error_callback:
                        await self.error_callback(
                            request_id,
                            ModuleError(
                                message=f"API error: {response.status} - {error_text}",
                                code=ModuleErrorCode.VENDOR_ERROR,
                            ),
                        )
                    return

                # Read the full response and process it
                full_response = await response.read()
                self.ten_env.log_info(f"Inworld response: {len(full_response)} bytes")
                response_text = full_response.decode("utf-8")
                
                # Log first 500 chars for debugging
                self.ten_env.log_info(f"Response preview: {response_text[:500] if len(response_text) > 500 else response_text}")
                
                # Process newline-delimited JSON
                lines = response_text.split("\n")
                self.ten_env.log_info(f"Response has {len(lines)} lines")
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                        self.ten_env.log_info(f"Chunk keys: {list(chunk.keys())}")
                        audio_content = chunk.get("result", {}).get("audioContent")
                        
                        if audio_content:
                            audio_bytes = base64.b64decode(audio_content)
                            self.ten_env.log_info(f"Decoded {len(audio_bytes)} bytes, first 4: {audio_bytes[:4] if len(audio_bytes) >= 4 else audio_bytes}")
                            
                            # Check for WAV header (RIFF)
                            if len(audio_bytes) > 44 and audio_bytes[:4] == b"RIFF":
                                audio_bytes = audio_bytes[44:]
                                self.ten_env.log_info(f"Stripped WAV header, now {len(audio_bytes)} bytes")
                            
                            if self.response_msgs and len(audio_bytes) > 0:
                                await self.response_msgs.put(
                                    (audio_bytes, False, request_id)
                                )
                                self.ten_env.log_info(f"Queued {len(audio_bytes)} bytes of audio")
                        else:
                            self.ten_env.log_warn(f"No audioContent in chunk")
                    except json.JSONDecodeError as e:
                        self.ten_env.log_warn(f"Failed to parse JSON: {e}")
                        continue

        except aiohttp.ClientError as e:
            self.ten_env.log_error(f"Inworld TTS request failed: {e}")
            if self.error_callback:
                await self.error_callback(
                    request_id,
                    ModuleError(
                        message=str(e),
                        code=ModuleErrorCode.NETWORK_ERROR,
                    ),
                )

    async def send_text(self, text: str, request_id: str, is_end: bool = False) -> None:
        """Send text for synthesis."""
        await self.text_input_queue.put({
            "text": text,
            "request_id": request_id,
            "is_end": is_end,
        })

    async def flush(self) -> None:
        """Flush the text queue."""
        while not self.text_input_queue.empty():
            try:
                self.text_input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
