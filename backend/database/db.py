"""
Database initialization module
SQLite + SQLAlchemy setup
"""

from flask_sqlalchemy import SQLAlchemy

# Global DB object
db = SQLAlchemy()


def init_db(app):
    """
    Initialize database with Flask app
    """
    app.config["SQLALCHEMY_DATABASE_URI"] = app.config.get(
        "DATABASE_URI",
        "sqlite:///app.db"
    )

    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    with app.app_context():
        db.create_all()