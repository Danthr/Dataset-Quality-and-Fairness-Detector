"""
Authentication Routes
Session-based auth with SQLite persistence
"""

from flask import Blueprint, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from backend.database.db import db
from backend.database.models import User
import logging

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/register", methods=["POST"])
def register():
    """
    Register new user
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify(
                {"error": "Username and password required"}
            ), 400

        if len(password) < 6:
            return jsonify(
                {"error": "Password must be at least 6 characters"}
            ), 400

        existing_user = User.query.filter_by(
            username=username
        ).first()

        if existing_user:
            return jsonify(
                {"error": "Username already exists"}
            ), 400

        user = User(
            username=username,
            password_hash=generate_password_hash(password)
        )

        db.session.add(user)
        db.session.commit()

        logger.info(f"New user registered: {username}")

        return jsonify(
            {"message": "User registered successfully"}
        ), 201

    except Exception as e:
        db.session.rollback()
        logger.error(str(e))
        return jsonify({"error": str(e)}), 500


@auth_bp.route("/login", methods=["POST"])
def login():
    """
    Login user
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify(
                {"error": "Username and password required"}
            ), 400

        user = User.query.filter_by(username=username).first()

        if not user:
            return jsonify(
                {"error": "Invalid credentials"}
            ), 401

        if not check_password_hash(user.password_hash, password):
            return jsonify(
                {"error": "Invalid credentials"}
            ), 401

        session["user"] = username
        session["user_id"] = user.id

        logger.info(f"User logged in: {username}")

        return jsonify(
            {
                "message": "Login successful",
                "user": user.to_dict(),
            }
        ), 200

    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 500


@auth_bp.route("/logout", methods=["POST"])
def logout():
    """
    Logout current user
    """
    username = session.get("user", "unknown")
    session.clear()

    logger.info(f"User logged out: {username}")

    return jsonify({"message": "Logout successful"}), 200


@auth_bp.route("/me", methods=["GET"])
def current_user():
    """
    Get currently logged in user
    """
    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    user = User.query.filter_by(
        username=session["user"]
    ).first()

    if not user:
        session.clear()
        return jsonify({"error": "User not found"}), 404

    return jsonify({"user": user.to_dict()}), 200