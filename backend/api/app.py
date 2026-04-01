"""
Flask Application Factory
"""

from flask import Flask
from flask_cors import CORS
from .config import config
import logging

from backend.database.db import init_db
from backend.database import models


def create_app(config_name="development"):
    """
    Create and configure Flask application
    """
    app = Flask(__name__)

    # Load config
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    # Initialize database
    init_db(app)

    # Enable CORS
    CORS(app, origins=app.config["CORS_ORIGINS"])

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Register routes
    from .routes import api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    @app.route("/health")
    def health():
        return {
            "status": "healthy",
            "message": "Fairness Auditing API is running",
        }, 200

    @app.route("/")
    def index():
        return {
            "message": "Automated Dataset Quality Scoring and Fairness Auditing System API",
            "version": "3.0.0",
            "database": "SQLite enabled",
            "endpoints": {
                "health": "/health",
                "upload": "/api/upload",
                "quality": "/api/quality/<dataset_id>",
                "audit": "/api/audit",
                "explain": "/api/explain",
                "results": "/api/results/<dataset_id>",
                "datasets": "/api/datasets",
            },
        }, 200

    return app


if __name__ == "__main__":
    app = create_app("development")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )