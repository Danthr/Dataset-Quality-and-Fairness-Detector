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

results_store = {}


def get_current_user_id():
    return session.get("user_id")


def validate_dataset_ownership(dataset_id):
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


def recover_from_db(report_obj):
    """
    Recover persisted results after server restart.
    """
    return {
        "filename": report_obj.filename,
        "stats": report_obj.stats_report,
        "quality": report_obj.quality_report,
        "fairness": report_obj.fairness_report,
        "explanation": report_obj.explanation_report,
        "audit_allowed": report_obj.audit_allowed,
        "detected_attributes": report_obj.detected_attributes or [],
        "processed": report_obj.processed,
    }


def get_dataframe(dataset_id, report_obj):
    """
    Get dataframe from memory cache.
    If missing after restart, rebuild from stored file path.
    """
    if dataset_id in results_store and "df" in results_store[dataset_id]:
        return results_store[dataset_id]["df"]

    ingestion = DataIngestion()

    df_raw, message = ingestion.load_dataset(report_obj.file_path)

    if df_raw is None:
        raise ValueError(
            f"Could not reload dataset from disk: {message}"
        )

    df = ingestion.preprocess_dataset(df_raw)

    cached_data = recover_from_db(report_obj)
    cached_data["df"] = df
    cached_data["filename"] = report_obj.filename
    cached_data["user_id"] = report_obj.user_id

    results_store[dataset_id] = cached_data

    return df


@api_bp.route("/upload", methods=["POST"])
def upload_file():
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
            "audit_allowed": False,
            "detected_attributes": [],
        }

        report = DatasetReport(
            dataset_id=dataset_id,
            filename=filename,
            file_path=str(file_path),
            user_id=current_user_id,
            stats_report=stats,
            audit_allowed=False,
            processed=False,
            detected_attributes=[]
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
    try:
        report_obj, error_response, status = validate_dataset_ownership(dataset_id)
        if error_response:
            return jsonify(error_response), status

        df = get_dataframe(dataset_id, report_obj)

        scorer = DataQualityScorer()
        quality_result = convert_numpy_types(scorer.score_all(df))

        auditor = FairnessAuditor()
        eligibility = auditor.evaluate_audit_eligibility(df)

        results_store[dataset_id]["quality"] = quality_result
        results_store[dataset_id]["audit_allowed"] = eligibility["audit_allowed"]
        results_store[dataset_id]["detected_attributes"] = eligibility["detected_attributes"]

        report_obj.quality_report = quality_result
        report_obj.audit_allowed = eligibility["audit_allowed"]
        report_obj.detected_attributes = eligibility["detected_attributes"]

        db.session.commit()

        return jsonify({
            "dataset_id": dataset_id,
            "data_quality": quality_result,
            "audit_allowed": eligibility["audit_allowed"],
            "detected_attributes": eligibility["detected_attributes"],
            "detection_source": eligibility["source"],
            "next_step": "audit" if eligibility["audit_allowed"] else "explain",
            "message": eligibility["message"],
        }), 200

    except Exception as e:
        logger.error(f"Quality error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/audit", methods=["POST"])
def audit_dataset():
    try:
        data = request.get_json()

        if not data or "dataset_id" not in data:
            return jsonify({"error": "dataset_id required"}), 400

        dataset_id = data["dataset_id"]

        report_obj, error_response, status = validate_dataset_ownership(dataset_id)
        if error_response:
            return jsonify(error_response), status

        df = get_dataframe(dataset_id, report_obj)

        protected_attributes = data.get("protected_attributes")
        outcome_attr = data.get("outcome_attribute")

        auditor = FairnessAuditor()

        fairness_result = convert_numpy_types(
            auditor.audit_all(df, protected_attributes, outcome_attr)
        )

        if not fairness_result.get("audit_allowed", False):
            return jsonify(fairness_result), 400

        results_store[dataset_id]["fairness"] = fairness_result
        results_store[dataset_id]["processed"] = True

        report_obj.fairness_report = fairness_result
        report_obj.processed = True

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
    try:
        data = request.get_json()

        if not data or "dataset_id" not in data:
            return jsonify({"error": "dataset_id required"}), 400

        dataset_id = data["dataset_id"]

        report_obj, error_response, status = validate_dataset_ownership(dataset_id)
        if error_response:
            return jsonify(error_response), status

        stored_data = (
            results_store[dataset_id]
            if dataset_id in results_store
            else recover_from_db(report_obj)
        )

        if not stored_data.get("quality"):
            return jsonify({"error": "Run /quality first"}), 400

        explainer = AIExplainer()

        explanation_result = explainer.generate_full_report(
            stored_data["quality"],
            stored_data.get("fairness"),
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
    try:
        report_obj, error_response, status = validate_dataset_ownership(dataset_id)
        if error_response:
            return jsonify(error_response), status

        stored = (
            results_store[dataset_id]
            if dataset_id in results_store
            else recover_from_db(report_obj)
        )

        return jsonify(convert_numpy_types(stored)), 200

    except Exception as e:
        logger.error(f"Results error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/datasets", methods=["GET"])
def list_datasets():
    try:
        current_user_id = get_current_user_id()

        if not current_user_id:
            return jsonify({"error": "Login required"}), 401

        reports = DatasetReport.query.filter_by(
            user_id=current_user_id
        ).order_by(
            DatasetReport.created_at.desc()
        ).all()

        datasets = [report.to_dict() for report in reports]

        return jsonify({
            "total_datasets": len(datasets),
            "datasets": datasets,
        }), 200

    except Exception as e:
        logger.error(f"List error: {str(e)}")
        return jsonify({"error": str(e)}), 500