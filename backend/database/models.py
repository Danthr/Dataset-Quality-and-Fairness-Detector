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

    file_path = db.Column(
        db.String(500),
        nullable=False
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False
    )

    stats_report = db.Column(
        db.JSON,
        nullable=True
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

    detected_attributes = db.Column(
        db.JSON,
        nullable=True
    )

    audit_allowed = db.Column(
        db.Boolean,
        default=False
    )

    processed = db.Column(
        db.Boolean,
        default=False
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
            "file_path": self.file_path,
            "user_id": self.user_id,
            "stats_report": self.stats_report,
            "quality_report": self.quality_report,
            "fairness_report": self.fairness_report,
            "explanation_report": self.explanation_report,
            "detected_attributes": self.detected_attributes,
            "audit_allowed": self.audit_allowed,
            "processed": self.processed,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }