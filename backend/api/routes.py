"""
API Routes
Endpoints for dataset upload, quality scoring,
fairness auditing, explanation, and DB persistence
"""

from flask import Blueprint, request, jsonify, session
from werkzeug.utils import secure_filename
import numpy as np
import uuid
from pathlib import Path
import logging

from backend.data_processing.ingestion import DataIngestion
from backend.quality.data_quality_scorer import DataQualityScorer
from backend.fairness.auditor import FairnessAuditor
from backend.explainer.ai_explainer import AIExplainer
from backend.database.db import db
from backend.database.models import DatasetReport

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)

# In-memory store keyed by dataset_id.
# Stores the preprocessed DataFrame so every endpoint reuses it directly
results_store = {}


def get_current_user_id():
    """
    Get currently logged-in user id from session.
    """
    return session.get("user_id")


def validate_dataset_ownership(dataset_id):
    """
    Ensure requested dataset belongs to logged-in user.
    """
    report = DatasetReport.query.filter_by(dataset_id=dataset_id).first()

    if not report:
        return None, {"error": "Dataset not found"}, 404

    current_user_id = get_current_user_id()

    if not current_user_id:
        return None, {"error": "Login required"}, 401

    if report.user_id != current_user_id:
        return None, {"error": "Unauthorized dataset access"}, 403

    return report, None, None


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types.
    Required because jsonify() cannot serialise numpy int64/float64.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in {"csv", "xlsx", "xls"}
    )


@api_bp.route("/upload", methods=["POST"])
def upload_file():
    """
    Upload a CSV or Excel dataset.
    """
    try:
        current_user_id = get_current_user_id()

        if not current_user_id:
            return jsonify({"error": "Login required"}), 401

        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Only CSV and Excel files allowed"}), 400

        dataset_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)

        file_path = Path("data/raw") / f"{dataset_id}_{filename}"
        file.save(str(file_path))

        ingestion = DataIngestion()
        df_raw, message = ingestion.load_dataset(str(file_path))

        if df_raw is None:
            return jsonify({"error": message}), 400

        df = ingestion.preprocess_dataset(df_raw)
        stats = convert_numpy_types(ingestion.get_basic_stats(df))

        results_store[dataset_id] = {
            "filename": filename,
            "user_id": current_user_id,
            "df": df,
            "stats": stats,
            "quality": None,
            "fairness": None,
            "explanation": None,
            "processed": False,
        }

        report = DatasetReport(
            dataset_id=dataset_id,
            filename=filename,
            user_id=current_user_id
        )

        db.session.add(report)
        db.session.commit()

        return jsonify({
            "dataset_id": dataset_id,
            "filename": filename,
            "message": "File uploaded successfully",
            "stats": stats,
        }), 200

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/quality/<dataset_id>", methods=["GET"])
def get_quality(dataset_id):
    """
    Run data quality scoring on the stored preprocessed DataFrame.
    """
    try:
        report_obj, error_response, status = validate_dataset_ownership(dataset_id)
        if error_response:
            return jsonify(error_response), status

        if dataset_id not in results_store:
            return jsonify({"error": "Dataset not found. Upload the file first."}), 404

        df = results_store[dataset_id]["df"]

        scorer = DataQualityScorer()
        quality_result = convert_numpy_types(scorer.score_all(df))

        results_store[dataset_id]["quality"] = quality_result

        report_obj.quality_report = quality_result
        db.session.commit()

        return jsonify({
            "dataset_id": dataset_id,
            "data_quality": quality_result,
            "message": "Quality scoring complete",
        }), 200

    except Exception as e:
        logger.error(f"Quality error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/audit", methods=["POST"])
def audit_dataset():
    """
    Run fairness audit on the stored preprocessed DataFrame.
    """
    try:
        data = request.get_json()

        if not data or "dataset_id" not in data:
            return jsonify({"error": "dataset_id required"}), 400

        dataset_id = data["dataset_id"]

        report_obj, error_response, status = validate_dataset_ownership(dataset_id)
        if error_response:
            return jsonify(error_response), status

        if dataset_id not in results_store:
            return jsonify({"error": "Dataset not found. Upload the file first."}), 404

        df = results_store[dataset_id]["df"]

        protected_attributes = data.get("protected_attributes", None)
        outcome_attr = data.get("outcome_attribute", None)

        if outcome_attr is not None and outcome_attr not in df.columns:
            return jsonify({
                "error": f"Outcome column '{outcome_attr}' not found. "
                         f"Available columns: {df.columns.tolist()}"
            }), 400

        auditor = FairnessAuditor()
        fairness_result = convert_numpy_types(
            auditor.audit_all(df, protected_attributes, outcome_attr)
        )

        results_store[dataset_id]["fairness"] = fairness_result
        results_store[dataset_id]["processed"] = True

        report_obj.fairness_report = fairness_result
        db.session.commit()

        return jsonify({
            "dataset_id": dataset_id,
            "fairness_audit": fairness_result,
            "message": "Fairness audit complete",
        }), 200

    except Exception as e:
        logger.error(f"Audit error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/explain", methods=["POST"])
def explain_results():
    """
    Generate AI explanation using Claude API.
    """
    try:
        data = request.get_json()

        if not data or "dataset_id" not in data:
            return jsonify({"error": "dataset_id required"}), 400

        dataset_id = data["dataset_id"]

        report_obj, error_response, status = validate_dataset_ownership(dataset_id)
        if error_response:
            return jsonify(error_response), status

        if dataset_id in results_store:
            stored_data = results_store[dataset_id]
        else:
            stored_data = {
                "quality": report_obj.quality_report,
                "fairness": report_obj.fairness_report,
                "explanation": report_obj.explanation_report,
            }

        if not stored_data.get("quality"):
            return jsonify({"error": "Run /quality first before requesting explanation"}), 400

        if not stored_data.get("fairness"):
            return jsonify({"error": "Run /audit first before requesting explanation"}), 400

        explainer = AIExplainer()
        explanation_result = explainer.generate_full_report(
            stored_data["quality"],
            stored_data["fairness"],
        )

        if dataset_id in results_store:
            results_store[dataset_id]["explanation"] = explanation_result

        report_obj.explanation_report = explanation_result
        db.session.commit()

        return jsonify({
            "dataset_id": dataset_id,
            "explanation": explanation_result,
            "message": "AI explanation generated successfully",
        }), 200

    except Exception as e:
        logger.error(f"Explain error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/results/<dataset_id>", methods=["GET"])
def get_results(dataset_id):
    """
    Get all stored results for a dataset.
    """
    try:
        report_obj, error_response, status = validate_dataset_ownership(dataset_id)
        if error_response:
            return jsonify(error_response), status

        if dataset_id not in results_store:
            return jsonify({"error": "Dataset not found"}), 404

        stored = results_store[dataset_id]

        return jsonify(convert_numpy_types({
            "filename": stored["filename"],
            "stats": stored["stats"],
            "quality": stored["quality"],
            "fairness": stored["fairness"],
            "processed": stored["processed"],
        })), 200

    except Exception as e:
        logger.error(f"Results error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/datasets", methods=["GET"])
def list_datasets():
    """
    List only datasets uploaded by current user.
    """
    try:
        current_user_id = get_current_user_id()

        if not current_user_id:
            return jsonify({"error": "Login required"}), 401

        reports = DatasetReport.query.filter_by(
            user_id=current_user_id
        ).order_by(
            DatasetReport.created_at.desc()
        ).all()

        datasets = [
            {
                "dataset_id": report.dataset_id,
                "filename": report.filename,
                "created_at": report.created_at.isoformat(),
                "quality_available": report.quality_report is not None,
                "fairness_available": report.fairness_report is not None,
                "explanation_available": report.explanation_report is not None
            }
            for report in reports
        ]

        return jsonify({
            "total_datasets": len(datasets),
            "datasets": datasets,
        }), 200

    except Exception as e:
        logger.error(f"List error: {str(e)}")
        return jsonify({"error": str(e)}), 500