"""
Configuration Settings for Flask API
"""

import os
from pathlib import Path


class Config:
    """Base configuration"""

    # Flask settings
    SECRET_KEY = (
        os.environ.get("SECRET_KEY")
        or "dev-secret-key-change-in-production"
    )

    DEBUG = False
    TESTING = False

    # Upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    UPLOAD_FOLDER = DATA_DIR / "raw"
    PROCESSED_FOLDER = DATA_DIR / "processed"
    OUTPUT_FOLDER = DATA_DIR / "outputs"

    # Future modules
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    DATABASE_URI = os.environ.get(
        "DATABASE_URI",
        f"sqlite:///{BASE_DIR / 'app.db'}"
    )

    # CORS
    CORS_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:8501",
    ]

    @staticmethod
    def init_app(app):
        """Initialize application"""
        Config.UPLOAD_FOLDER.mkdir(
            parents=True,
            exist_ok=True
        )

        Config.PROCESSED_FOLDER.mkdir(
            parents=True,
            exist_ok=True
        )

        Config.OUTPUT_FOLDER.mkdir(
            parents=True,
            exist_ok=True
        )


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


class TestingConfig(Config):
    TESTING = True


config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}