"""
API Routes
DB-first logic + conditional pipeline (FIXED: outcome column consistency)
"""

from flask import Blueprint, request, jsonify, session
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import uuid
from pathlib import Path
import logging

from backend.data_processing.ingestion import DataIngestion
from backend.quality.data_quality_scorer import DataQualityScorer
from backend.fairness.auditor import FairnessAuditor
from backend.explainer.ai_explainer import AIExplainer
from backend.database.db import db
from backend.database.models import DatasetReport
from backend.auth.auth import decode_token

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__)



def get_current_user_id():
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    
    token = auth_header.split(" ")[1]
    payload = decode_token(token)
    
    if not payload:
        return None
    
    return payload.get("user_id")


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
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"csv", "xlsx", "xls"}


def get_dataframe(report_obj):
    ingestion = DataIngestion()
    df_raw, message = ingestion.load_dataset(report_obj.file_path)

    if df_raw is None:
        raise ValueError(f"Could not reload dataset: {message}")

    return ingestion.preprocess_dataset(df_raw)


# ------------------ UPLOAD (UPDATED WITH PHASE 2) ------------------

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
            return jsonify({"error": "Invalid file type"}), 400

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

        # ✅ PHASE 2: Run semantic column analysis
        explainer = AIExplainer()
        semantic_analysis = convert_numpy_types(
            explainer.analyze_column_semantics(df)
        )

        report = DatasetReport(
            dataset_id=dataset_id,
            filename=filename,
            file_path=str(file_path),
            user_id=current_user_id,
            stats_report=stats,
            semantic_analysis=semantic_analysis  # NEW FIELD
        )

        db.session.add(report)
        db.session.commit()

        return jsonify({
            "dataset_id": dataset_id,
            "filename": filename,
            "stats": stats,
            "semantic_analysis": semantic_analysis  # RETURNED TO FRONTEND
        }), 200

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ------------------ QUALITY ------------------

@api_bp.route("/quality/<dataset_id>", methods=["GET"])
def get_quality(dataset_id):
    try:
        report_obj, error, status = validate_dataset_ownership(dataset_id)
        if error:
            return jsonify(error), status

        if report_obj.quality_report:
            return jsonify({
                "dataset_id": dataset_id,
                "data_quality": report_obj.quality_report,
                "audit_allowed": report_obj.audit_allowed,
                "detected_attributes": report_obj.detected_attributes,
                "message": "Loaded from DB"
            }), 200

        df = get_dataframe(report_obj)

        scorer = DataQualityScorer()
        quality_result = convert_numpy_types(scorer.score_all(df))

        auditor = FairnessAuditor()
        eligibility = auditor.evaluate_audit_eligibility(df)

        report_obj.quality_report = quality_result
        report_obj.audit_allowed = eligibility["audit_allowed"]
        report_obj.detected_attributes = eligibility["detected_attributes"]

        db.session.commit()

        return jsonify({
            "dataset_id": dataset_id,
            "data_quality": quality_result,
            "audit_allowed": eligibility["audit_allowed"],
            "detected_attributes": eligibility["detected_attributes"],
            "message": "Computed"
        }), 200

    except Exception as e:
        logger.error(f"Quality error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ------------------ AUDIT (FIXED: Outcome column handling) ------------------

@api_bp.route("/audit", methods=["POST"])
def audit_dataset():
    try:
        data = request.get_json()
        dataset_id = data.get("dataset_id")

        report_obj, error, status = validate_dataset_ownership(dataset_id)
        if error:
            return jsonify(error), status

        if report_obj.fairness_report:
            return jsonify({
                "dataset_id": dataset_id,
                "fairness_audit": report_obj.fairness_report,
                "message": "Loaded from DB"
            }), 200

        if not report_obj.quality_report:
            return jsonify({
                "error": "Quality analysis required before audit",
                "next_step": "/api/quality/<dataset_id>"
            }), 400

        df = get_dataframe(report_obj)

        # 🔧 FIX: Get outcome column from request or semantic analysis
        outcome_attr = data.get("outcome_attribute") or data.get("outcome_column")
        
        # If still not provided, try semantic analysis
        if not outcome_attr and report_obj.semantic_analysis:
            outcome_variables = report_obj.semantic_analysis.get("outcome_variables", [])
            if outcome_variables:
                outcome_attr = outcome_variables[0]
                logger.info(f"Using outcome column from semantic analysis: {outcome_attr}")

        # 🔧 CRITICAL: Validate outcome column exists in dataset
        if outcome_attr and outcome_attr not in df.columns:
            return jsonify({
                "error": f"Outcome column '{outcome_attr}' not found in dataset",
                "available_columns": list(df.columns),
                "hint": "Check semantic_analysis.outcome_variables from upload response"
            }), 400

        auditor = FairnessAuditor()
        fairness_result = convert_numpy_types(
            auditor.audit_all(df, data.get("protected_attributes"), outcome_attr)
        )

        report_obj.fairness_report = fairness_result
        report_obj.processed = True

        db.session.commit()

        return jsonify({
            "dataset_id": dataset_id,
            "fairness_audit": fairness_result,
            "message": "Computed"
        }), 200

    except Exception as e:
        logger.error(f"Audit error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ------------------ EXPLAIN (FIXED: Pass stored outcome column) ------------------

@api_bp.route("/explain", methods=["POST"])
def explain_results():
    try:
        data = request.get_json()
        dataset_id = data.get("dataset_id")

        report_obj, error, status = validate_dataset_ownership(dataset_id)
        if error:
            return jsonify(error), status

        if not report_obj.quality_report:
            return jsonify({
                "error": "Quality analysis required before explanation",
                "next_step": "/api/quality/<dataset_id>"
            }), 400

        if report_obj.audit_allowed and not report_obj.fairness_report:
            return jsonify({
                "error": "Fairness audit required before explanation",
                "next_step": "/api/audit"
            }), 400

        # Reload dataframe to get dataset info
        df = get_dataframe(report_obj)
        
        # 🔧 FIX: Get outcome column from fairness_report (source of truth)
        outcome_column = None
        if report_obj.fairness_report:
            outcome_column = report_obj.fairness_report.get("outcome_attribute")
        
        dataset_info = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "outcome_column": outcome_column  # 🔧 NEW: Pass to remediation plan
        }

        explainer = AIExplainer()

        # Generate standard explanation
        explanation = convert_numpy_types(
            explainer.generate_full_report(
                report_obj.quality_report,
                report_obj.fairness_report
            )
        )

        # ✅ PHASE 3: Generate remediation plan with correct outcome column
        remediation_plan = convert_numpy_types(
            explainer.generate_remediation_plan(
                report_obj.quality_report,
                report_obj.fairness_report,
                dataset_info
            )
        )

        # Store both
        report_obj.explanation_report = explanation
        report_obj.remediation_plan = remediation_plan
        db.session.commit()

        return jsonify({
            "dataset_id": dataset_id,
            "explanation": explanation,
            "remediation_plan": remediation_plan,
            "message": "Computed"
        }), 200

    except Exception as e:
        logger.error(f"Explain error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ------------------ RESULTS ------------------

@api_bp.route("/results/<dataset_id>", methods=["GET"])
def get_results(dataset_id):
    try:
        report_obj, error, status = validate_dataset_ownership(dataset_id)
        if error:
            return jsonify(error), status

        return jsonify(report_obj.to_dict()), 200

    except Exception as e:
        logger.error(f"Results error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ------------------ DATASETS ------------------

@api_bp.route("/datasets", methods=["GET"])
def list_datasets():
    try:
        current_user_id = get_current_user_id()

        reports = DatasetReport.query.filter_by(
            user_id=current_user_id
        ).all()

        return jsonify({
            "datasets": [r.to_dict() for r in reports]
        }), 200

    except Exception as e:
        logger.error(f"List error: {str(e)}")
        return jsonify({"error": str(e)}), 500