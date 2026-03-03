from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "DocuChat AI"
    API_V1_STR: str = "/api/v1"
    OPENAI_API_KEY: str
    CHROMA_DB_DIR: str = "./chroma_db"

    class Config:
        env_file = ".env"

settings = Settings()
