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
            ten_env.log_debug("on_init")

            if self.config is None:
                config_json, _ = await self.ten_env.get_property_to_json("")
                self.config = InworldTTSConfig.model_validate_json(config_json)
                self.config.update_params()
                
                if not self.config.api_key:
                    raise ValueError("api_key is required")
                
                self.ten_env.log_info(
                    f"config: {self.config.to_str(sensitive_handling=True)}",
                    category=LOG_CATEGORY_KEY_POINT,
                )

            # Create HTTP session
            connector = aiohttp.TCPConnector(limit=10)
            self._session = aiohttp.ClientSession(connector=connector)
            self.ten_env.log_info("Inworld TTS extension initialized")

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
        ten_env.log_debug("on_stop")

    async def on_deinit(self, ten_env: AsyncTenEnv) -> None:
        await super().on_deinit(ten_env)
        ten_env.log_debug("on_deinit")

    def vendor(self) -> str:
        return "inworld"

    def synthesize_audio_sample_rate(self) -> int:
        return self.config.sample_rate if self.config else 16000

    def synthesize_audio_channels(self) -> int:
        return 1

    def synthesize_audio_sample_width(self) -> int:
        return 2  # 16-bit audio

    async def request_tts(self, t: TTSTextInput) -> None:
        """Handle TTS request - called when text input is received."""
        text_preview = t.text[:50] + "..." if len(t.text) > 50 else t.text
        self.ten_env.log_info(f"request_tts: text='{text_preview}' end={t.text_input_end}")
        
        if not t.text and not t.text_input_end:
            return
            
        if t.text:
            await self._synthesize(t.text, t.request_id)
            
        # Only finalize when text_input_end is True
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
        """Synthesize text to speech using Inworld API."""
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
        
        first_audio = True

        try:
            async with self._session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.ten_env.log_error(f"Inworld TTS API error: {response.status} - {error_text}")
                    await self.send_tts_error(
                        request_id=request_id,
                        error=ModuleError(
                            message=f"API error: {response.status} - {error_text}",
                            module=ModuleType.TTS,
                            code=ModuleErrorCode.VENDOR_ERROR,
                            vendor_info={},
                        ),
                    )
                    return

                # Read the full response
                full_response = await response.read()
                self.ten_env.log_info(f"Inworld response: {len(full_response)} bytes")
                response_text = full_response.decode("utf-8")
                
                # Process newline-delimited JSON
                for line in response_text.split("\n"):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                        audio_content = chunk.get("result", {}).get("audioContent")
                        
                        if audio_content:
                            audio_bytes = base64.b64decode(audio_content)
                            self.ten_env.log_debug(f"Decoded {len(audio_bytes)} bytes")
                            
                            # Check for WAV header (RIFF)
                            if len(audio_bytes) > 44 and audio_bytes[:4] == b"RIFF":
                                audio_bytes = audio_bytes[44:]
                                self.ten_env.log_debug(f"Stripped WAV header, now {len(audio_bytes)} bytes")
                            
                            if len(audio_bytes) > 0:
                                # Send audio start on first chunk
                                if first_audio:
                                    await self.send_tts_audio_start(request_id=request_id)
                                    first_audio = False
                                
                                # Send audio data
                                self.ten_env.log_info(f"Sending {len(audio_bytes)} bytes to TTS")
                                await self.send_tts_audio_data(audio_data=audio_bytes)
                                self.ten_env.log_info(f"Audio data sent successfully")
                                
                    except json.JSONDecodeError as e:
                        self.ten_env.log_warn(f"Failed to parse JSON: {e}")
                        continue

        except aiohttp.ClientError as e:
            self.ten_env.log_error(f"Inworld TTS request failed: {e}")
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
