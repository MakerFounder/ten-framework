import asyncio
import traceback
import aiohttp
import base64
import json
from typing import Optional

from ten_ai_base.tts2 import AsyncTTS2BaseExtension, RequestState
from ten_ai_base.struct import TTSTextInput
from ten_ai_base.message import (
    ModuleError,
    ModuleErrorCode,
    ModuleType,
    TTSAudioEndReason,
)
from ten_runtime import AsyncTenEnv
from ten_ai_base.const import LOG_CATEGORY_KEY_POINT

from .config import InworldTTSConfig


class InworldTTSExtension(AsyncTTS2BaseExtension):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.config: InworldTTSConfig = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def on_init(self, ten_env: AsyncTenEnv) -> None:
        try:
            await super().on_init(ten_env)

            if self.config is None:
                config_json, _ = await self.ten_env.get_property_to_json("")
                self.config = InworldTTSConfig.model_validate_json(config_json)
                self.config.update_params()
                
                if not self.config.api_key:
                    raise ValueError("api_key is required")

            # Create optimized HTTP session
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

        except Exception as e:
            ten_env.log_error(f"on_init failed: {traceback.format_exc()}")
            await self.send_tts_error(
                request_id="",
                error=ModuleError(
                    message=str(e),
                    module=ModuleType.TTS,
                    code=ModuleErrorCode.FATAL_ERROR,
                    vendor_info={},
                ),
            )

    async def on_stop(self, ten_env: AsyncTenEnv) -> None:
        if self._session:
            await self._session.close()
        await super().on_stop(ten_env)

    async def on_deinit(self, ten_env: AsyncTenEnv) -> None:
        await super().on_deinit(ten_env)

    def vendor(self) -> str:
        return "inworld"

    def synthesize_audio_sample_rate(self) -> int:
        return self.config.sample_rate if self.config else 16000

    def synthesize_audio_channels(self) -> int:
        return 1

    def synthesize_audio_sample_width(self) -> int:
        return 2

    async def request_tts(self, t: TTSTextInput) -> None:
        """Handle TTS request."""
        if not t.text and not t.text_input_end:
            return
            
        if t.text:
            await self._synthesize(t.text, t.request_id)
            
        if t.text_input_end:
            await self.send_tts_audio_end(
                request_id=t.request_id,
                request_event_interval_ms=0,
                request_total_audio_duration_ms=0,
                reason=TTSAudioEndReason.REQUEST_END,
            )
            await self.finish_request(
                request_id=t.request_id,
                reason=TTSAudioEndReason.REQUEST_END,
            )

    async def _synthesize(self, text: str, request_id: str) -> None:
        """Synthesize text to speech using optimized streaming."""
        if not self._session:
            self.ten_env.log_error(f"[TTS] No session available")
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

        first_audio = True
        total_bytes = 0

        try:
            async with self._session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.ten_env.log_error(f"TTS API error: {response.status} - {error_text}")
                    await self.send_tts_error(
                        request_id=request_id,
                        error=ModuleError(
                            message=f"API error: {response.status}",
                            module=ModuleType.TTS,
                            code=ModuleErrorCode.VENDOR_ERROR,
                            vendor_info={},
                        ),
                    )
                    return

                # TRUE STREAMING: Process chunks as they arrive
                buffer = b""
                async for chunk in response.content.iter_any():
                    buffer += chunk
                    
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
                                
                                if len(audio_bytes) > 0:
                                    total_bytes += len(audio_bytes)
                                    if first_audio:
                                        await self.send_tts_audio_start(request_id=request_id)
                                        first_audio = False
                                    await self.send_tts_audio_data(audio_data=audio_bytes)
                        except json.JSONDecodeError:
                            continue

                # Process remaining buffer
                if buffer.strip():
                    try:
                        data = json.loads(buffer.decode("utf-8"))
                        audio_content = data.get("result", {}).get("audioContent")
                        if audio_content:
                            audio_bytes = base64.b64decode(audio_content)
                            if len(audio_bytes) > 44 and audio_bytes[:4] == b"RIFF":
                                audio_bytes = audio_bytes[44:]
                            if len(audio_bytes) > 0:
                                total_bytes += len(audio_bytes)
                                if first_audio:
                                    await self.send_tts_audio_start(request_id=request_id)
                                    first_audio = False
                                await self.send_tts_audio_data(audio_data=audio_bytes)
                    except json.JSONDecodeError:
                        pass
                
                self.ten_env.log_info(f"[TTS] Done: {total_bytes} bytes")

        except aiohttp.ClientError as e:
            self.ten_env.log_error(f"TTS request failed: {e}")
            await self.send_tts_error(
                request_id=request_id,
                error=ModuleError(
                    message=str(e),
                    module=ModuleType.TTS,
                    code=ModuleErrorCode.NETWORK_ERROR,
                    vendor_info={},
                ),
            )

    async def on_cancel_tts(self, ten_env: AsyncTenEnv, request_id: str) -> None:
        """Handle TTS cancellation."""
        await self.send_tts_audio_end(
            request_id=request_id,
            request_event_interval_ms=0,
            request_total_audio_duration_ms=0,
            reason=TTSAudioEndReason.INTERRUPTED,
        )
        await self.finish_request(request_id=request_id, reason=TTSAudioEndReason.INTERRUPTED)
