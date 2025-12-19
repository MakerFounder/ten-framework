from typing import Any, Dict
from pydantic import BaseModel, Field
from dataclasses import dataclass


@dataclass
class InworldTTSConfig(BaseModel):
    api_key: str = ""
    voice_id: str = "Ashley"
    model_id: str = "inworld-tts-1"
    sample_rate: int = 16000
    base_url: str = "https://api.inworld.ai"
    disable_text_normalization: bool = True  # Saves 30-40ms latency
    params: Dict[str, Any] = Field(default_factory=dict)

    def update_params(self) -> None:
        """Update configuration with params dictionary."""
        if self.params:
            if "api_key" in self.params:
                self.api_key = self.params["api_key"]
            if "voice_id" in self.params:
                self.voice_id = self.params["voice_id"]
            if "model_id" in self.params:
                self.model_id = self.params["model_id"]
            if "sample_rate" in self.params:
                self.sample_rate = self.params["sample_rate"]
            if "base_url" in self.params:
                self.base_url = self.params["base_url"]
            if "disable_text_normalization" in self.params:
                self.disable_text_normalization = self.params["disable_text_normalization"]

    def to_str(self, sensitive_handling: bool = False) -> str:
        """Convert config to string with optional sensitive data handling."""
        config_dict = self.model_dump()
        if sensitive_handling and self.api_key:
            config_dict["api_key"] = "***REDACTED***"
        return str(config_dict)
