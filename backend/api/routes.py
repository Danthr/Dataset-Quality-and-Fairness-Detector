"""
API Routes
Endpoints for dataset upload, quality scoring,
fairness auditing, explanation, and DB persistence
"""

from flask import Blueprint, request, jsonify
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
# Stores the preprocessed DataFrame so every endpoint reuses it directly —
# no re-loading, no double-preprocessing.
results_store = {}


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

    Saves the raw file, runs ingestion + preprocessing ONCE,
    and stores the preprocessed DataFrame in results_store.
    All subsequent endpoints (quality, audit) read from results_store
    directly — they never re-load or re-preprocess the file.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Only CSV and Excel files allowed"}), 400

        dataset_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)

        # Save the raw uploaded file to disk
        file_path = Path("data/raw") / f"{dataset_id}_{filename}"
        file.save(str(file_path))

        # Load and validate
        ingestion = DataIngestion()
        df_raw, message = ingestion.load_dataset(str(file_path))

        if df_raw is None:
            return jsonify({"error": message}), 400

        # Preprocess ONCE here — normalise column names, deduplicate, strip whitespace
        df = ingestion.preprocess_dataset(df_raw)

        # Get stats from the preprocessed df so column names are already normalised
        stats = convert_numpy_types(ingestion.get_basic_stats(df))

        # Store preprocessed DataFrame — quality and audit endpoints use this directly
        results_store[dataset_id] = {
            "filename": filename,
            "df": df,          # preprocessed and ready to use
            "stats": stats,
            "quality": None,
            "fairness": None,
            "explanation": None,
            "processed": False,
        }

        # Persist metadata to DB
        report = DatasetReport(dataset_id=dataset_id, filename=filename)
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
    No file re-loading — reads directly from results_store.
    """
    try:
        if dataset_id not in results_store:
            return jsonify({"error": "Dataset not found. Upload the file first."}), 404

        # Reuse the already-preprocessed DataFrame
        df = results_store[dataset_id]["df"]

        scorer = DataQualityScorer()
        quality_result = convert_numpy_types(scorer.score_all(df))

        results_store[dataset_id]["quality"] = quality_result

        # Persist to DB
        report = DatasetReport.query.filter_by(dataset_id=dataset_id).first()
        if report:
            report.quality_report = quality_result
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

    Request body (JSON):
        dataset_id:           required
        protected_attributes: optional list — auto-detected if omitted
        outcome_attribute:    optional string — auto-selected if omitted
    """
    try:
        data = request.get_json()

        if not data or "dataset_id" not in data:
            return jsonify({"error": "dataset_id required"}), 400

        dataset_id = data["dataset_id"]

        if dataset_id not in results_store:
            return jsonify({"error": "Dataset not found. Upload the file first."}), 404

        # Reuse the already-preprocessed DataFrame
        df = results_store[dataset_id]["df"]

        protected_attributes = data.get("protected_attributes", None)
        outcome_attr = data.get("outcome_attribute", None)

        # Confirm outcome column exists if user specified one
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

        # Persist to DB
        report = DatasetReport.query.filter_by(dataset_id=dataset_id).first()
        if report:
            report.fairness_report = fairness_result
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
    Requires /quality and /audit to have been run first.
    """
    try:
        data = request.get_json()

        if not data or "dataset_id" not in data:
            return jsonify({"error": "dataset_id required"}), 400

        dataset_id = data["dataset_id"]

        if dataset_id in results_store:
            stored_data = results_store[dataset_id]
        else:
            # Fallback to DB after server restart
            report = DatasetReport.query.filter_by(dataset_id=dataset_id).first()
            if not report:
                return jsonify({"error": "Dataset not found"}), 404
            stored_data = {
                "quality": report.quality_report,
                "fairness": report.fairness_report,
                "explanation": report.explanation_report,
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

        report = DatasetReport.query.filter_by(dataset_id=dataset_id).first()
        if report:
            report.explanation_report = explanation_result
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
    Returns everything except the raw DataFrame object.
    """
    try:
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
    List all uploaded datasets currently in memory.
    """
    try:
        datasets = []
        for dataset_id, info in results_store.items():
            datasets.append({
                "dataset_id": dataset_id,
                "filename": info["filename"],
                "processed": info.get("processed", False),
                "total_rows": info["stats"]["total_rows"],
                "total_columns": info["stats"]["total_columns"],
            })

        return jsonify({
            "total_datasets": len(datasets),
            "datasets": datasets,
        }), 200

    except Exception as e:
        logger.error(f"List error: {str(e)}")
        return jsonify({"error": str(e)}), 500