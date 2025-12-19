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
    """Inworld AI TTS Client using streaming API with optimized latency."""

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
        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        connector = aiohttp.TCPConnector(
            limit=10,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
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
                self.ten_env.log_error(f"TTS process error: {e}")
                if self.error_callback:
                    await self.error_callback(
                        self.cur_request_id,
                        ModuleError(
                            message=str(e),
                            code=ModuleErrorCode.RUNTIME_ERROR,
                        ),
                    )

    async def _synthesize_stream(self, text: str, request_id: str) -> None:
        """Synthesize text to speech using TRUE streaming - process chunks as they arrive."""
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
            "disable_text_normalization": self.config.disable_text_normalization,
            "audio_config": {
                "audio_encoding": "LINEAR16",
                "sample_rate_hertz": self.config.sample_rate,
            },
        }

        try:
            async with self._session.post(
                url, headers=headers, json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.ten_env.log_error(f"Inworld TTS API error: {response.status}")
                    if self.error_callback:
                        await self.error_callback(
                            request_id,
                            ModuleError(
                                message=f"API error: {response.status} - {error_text}",
                                code=ModuleErrorCode.VENDOR_ERROR,
                            ),
                        )
                    return

                # TRUE STREAMING: Process each line as it arrives
                buffer = b""
                async for chunk in response.content.iter_any():
                    buffer += chunk
                    
                    # Process complete lines (newline-delimited JSON)
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            data = json.loads(line.decode("utf-8"))
                            audio_content = data.get("result", {}).get("audioContent")
                            
                            if audio_content:
                                audio_bytes = base64.b64decode(audio_content)
                                
                                # Strip WAV header if present
                                if len(audio_bytes) > 44 and audio_bytes[:4] == b"RIFF":
                                    audio_bytes = audio_bytes[44:]
                                
                                if self.response_msgs and len(audio_bytes) > 0:
                                    await self.response_msgs.put(
                                        (audio_bytes, False, request_id)
                                    )
                        except json.JSONDecodeError:
                            continue

                # Process any remaining data in buffer
                if buffer.strip():
                    try:
                        data = json.loads(buffer.decode("utf-8"))
                        audio_content = data.get("result", {}).get("audioContent")
                        if audio_content:
                            audio_bytes = base64.b64decode(audio_content)
                            if len(audio_bytes) > 44 and audio_bytes[:4] == b"RIFF":
                                audio_bytes = audio_bytes[44:]
                            if self.response_msgs and len(audio_bytes) > 0:
                                await self.response_msgs.put(
                                    (audio_bytes, False, request_id)
                                )
                    except json.JSONDecodeError:
                        pass

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
