"""
Authentication routes
Simple session-based auth for backend access
"""

from flask import Blueprint, request, jsonify, session
from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

auth_bp = Blueprint("auth", __name__)

# Temporary in-memory users store
# We can later move this to DB
users_store = {}


@auth_bp.route("/register", methods=["POST"])
def register():
    """
    Register new user
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify(
                {"error": "No data provided"}
            ), 400

        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify(
                {"error": "Username and password required"}
            ), 400

        if username in users_store:
            return jsonify(
                {"error": "User already exists"}
            ), 400

        users_store[username] = generate_password_hash(
            password
        )

        return jsonify(
            {"message": "User registered successfully"}
        ), 201

    except Exception as e:
        return jsonify(
            {"error": str(e)}
        ), 500


@auth_bp.route("/login", methods=["POST"])
def login():
    """
    Login user
    """
    try:
        data = request.get_json()

        username = data.get("username")
        password = data.get("password")

        if username not in users_store:
            return jsonify(
                {"error": "Invalid credentials"}
            ), 401

        if not check_password_hash(
            users_store[username],
            password
        ):
            return jsonify(
                {"error": "Invalid credentials"}
            ), 401

        session["user"] = username

        return jsonify(
            {
                "message": "Login successful",
                "user": username
            }
        ), 200

    except Exception as e:
        return jsonify(
            {"error": str(e)}
        ), 500


@auth_bp.route("/logout", methods=["POST"])
def logout():
    """
    Logout current user
    """
    session.pop("user", None)

    return jsonify(
        {"message": "Logout successful"}
    ), 200


@auth_bp.route("/me", methods=["GET"])
def current_user():
    """
    Get logged-in user
    """
    if "user" not in session:
        return jsonify(
            {"error": "Not logged in"}
        ), 401

    return jsonify(
        {"user": session["user"]}
    ), 200