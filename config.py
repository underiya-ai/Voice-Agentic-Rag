import os
from pydantic_settings import SettingsConfigDict,BaseSettings

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # LiveKit Configuration
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str

    # LLM Configuration
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"
    tavily_api_key:str

    # Pinecone Configuration
    pinecone_api_key: str
    # Voice Configuration
    stt_model: str = "deepgram/nova-3"
    stt_language: str = "multi"
    tts_model: str = "inworld/inworld-tts-2"
    tts_voice: str = "Ashley"


settings = Settings()

os.environ.setdefault("LIVEKIT_URL", settings.livekit_url)
os.environ.setdefault("LIVEKIT_API_KEY", settings.livekit_api_key)
os.environ.setdefault("LIVEKIT_API_SECRET", settings.livekit_api_secret)
os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)
os.environ.setdefault("Tavily_API_KEY", settings.tavily_api_key)
os.environ.setdefault("PINECONE_API_KEY", settings.pinecone_api_key)

