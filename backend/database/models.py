"""
Database models for persistence
"""

from datetime import datetime
from .db import db


class User(db.Model):
    """
    Stores registered users
    """
    __tablename__ = "users"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    username = db.Column(
        db.String(100),
        unique=True,
        nullable=False
    )

    password_hash = db.Column(
        db.String(255),
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    reports = db.relationship(
        "DatasetReport",
        backref="owner",
        lazy=True,
        cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "created_at": self.created_at.isoformat()
        }


class DatasetReport(db.Model):
    """
    Stores uploaded dataset reports
    """
    __tablename__ = "dataset_reports"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    dataset_id = db.Column(
        db.String(100),
        unique=True,
        nullable=False
    )

    filename = db.Column(
        db.String(255),
        nullable=False
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False
    )

    quality_report = db.Column(
        db.JSON,
        nullable=True
    )

    fairness_report = db.Column(
        db.JSON,
        nullable=True
    )

    explanation_report = db.Column(
        db.JSON,
        nullable=True
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    def to_dict(self):
        return {
            "dataset_id": self.dataset_id,
            "filename": self.filename,
            "user_id": self.user_id,
            "quality_report": self.quality_report,
            "fairness_report": self.fairness_report,
            "explanation_report": self.explanation_report,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }