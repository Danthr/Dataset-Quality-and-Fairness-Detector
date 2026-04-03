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

# Temporary in-memory storage
results_store = {}


def convert_numpy_types(obj):
    """Convert numpy types to native Python types"""
    if isinstance(obj, (np.integer,)):
        return int(obj)

    elif isinstance(obj, (np.floating,)):
        return float(obj)

    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)

    elif isinstance(obj, dict):
        return {
            key: convert_numpy_types(value)
            for key, value in obj.items()
        }

    elif isinstance(obj, list):
        return [
            convert_numpy_types(item)
            for item in obj
        ]

    return obj


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in {"csv", "xlsx", "xls"}
    )


@api_bp.route("/upload", methods=["POST"])
def upload_file():
    """
    Upload dataset
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify(
                {"error": "Only CSV and Excel files allowed"}
            ), 400

        dataset_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)

        file_path = Path("data/raw") / f"{dataset_id}_{filename}"
        file.save(str(file_path))

        ingestion = DataIngestion()
        df, message = ingestion.load_dataset(str(file_path))

        if df is None:
            return jsonify({"error": message}), 400

        stats = convert_numpy_types(
            ingestion.get_basic_stats(df)
        )

        results_store[dataset_id] = {
            "filename": filename,
            "file_path": str(file_path),
            "stats": stats,
            "processed": False,
        }

        report = DatasetReport(
            dataset_id=dataset_id,
            filename=filename
        )

        db.session.add(report)
        db.session.commit()

        return jsonify(
            {
                "dataset_id": dataset_id,
                "filename": filename,
                "message": "File uploaded successfully",
                "stats": stats,
            }
        ), 200

    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 500


@api_bp.route("/quality/<dataset_id>", methods=["GET"])
def get_quality(dataset_id):
    """
    Run data quality scoring
    """
    try:
        if dataset_id not in results_store:
            return jsonify({"error": "Dataset not found"}), 404

        file_path = results_store[dataset_id]["file_path"]

        ingestion = DataIngestion()
        df, _ = ingestion.load_dataset(file_path)

        df_processed = ingestion.preprocess_dataset(df)

        scorer = DataQualityScorer()
        quality_result = scorer.score_all(df_processed)

        quality_result = convert_numpy_types(
            quality_result
        )

        results_store[dataset_id]["quality"] = quality_result

        report = DatasetReport.query.filter_by(
            dataset_id=dataset_id
        ).first()

        if report:
            report.quality_report = quality_result
            db.session.commit()

        return jsonify(
            {
                "dataset_id": dataset_id,
                "data_quality": quality_result,
                "message": "Quality scoring complete",
            }
        ), 200

    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 500


@api_bp.route("/audit", methods=["POST"])
def audit_dataset():
    """
    Run fairness audit
    """
    try:
        data = request.get_json()

        if not data or "dataset_id" not in data:
            return jsonify(
                {"error": "dataset_id required"}
            ), 400

        dataset_id = data["dataset_id"]

        if dataset_id not in results_store:
            return jsonify({"error": "Dataset not found"}), 404

        protected_attributes = data.get(
            "protected_attributes",
            None,
        )

        outcome_attr = data.get(
            "outcome_attribute",
            None,
        )

        file_path = results_store[dataset_id]["file_path"]

        ingestion = DataIngestion()
        df, _ = ingestion.load_dataset(file_path)

        df_processed = ingestion.preprocess_dataset(df)

        auditor = FairnessAuditor()

        fairness_result = auditor.audit_all(
            df_processed,
            protected_attributes,
            outcome_attr,
        )

        fairness_result = convert_numpy_types(
            fairness_result
        )

        results_store[dataset_id]["fairness"] = fairness_result
        results_store[dataset_id]["processed"] = True

        report = DatasetReport.query.filter_by(
            dataset_id=dataset_id
        ).first()

        if report:
            report.fairness_report = fairness_result
            db.session.commit()

        return jsonify(
            {
                "dataset_id": dataset_id,
                "fairness_audit": fairness_result,
                "message": "Fairness audit complete",
            }
        ), 200

    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 500


@api_bp.route("/explain", methods=["POST"])
def explain_results():
    """
    Generate AI explanation report
    """
    try:
        data = request.get_json()

        if not data or "dataset_id" not in data:
            return jsonify(
                {"error": "dataset_id required"}
            ), 400

        dataset_id = data["dataset_id"]

        # First try memory store
        if dataset_id in results_store:
            stored_data = results_store[dataset_id]

        else:
            # Fallback to DB after restart
            report = DatasetReport.query.filter_by(
                dataset_id=dataset_id
            ).first()

            if not report:
                return jsonify(
                    {"error": "Dataset not found"}
                ), 404

            stored_data = {
                "quality": report.quality_report,
                "fairness": report.fairness_report,
                "explanation": report.explanation_report,
            }

        if "quality" not in stored_data or not stored_data["quality"]:
            return jsonify(
                {"error": "Run /quality first"}
            ), 400

        if "fairness" not in stored_data or not stored_data["fairness"]:
            return jsonify(
                {"error": "Run /audit first"}
            ), 400

        explainer = AIExplainer()

        explanation_result = (
            explainer.generate_full_report(
                stored_data["quality"],
                stored_data["fairness"],
            )
        )

        # Update memory store if present
        if dataset_id in results_store:
            results_store[dataset_id]["explanation"] = explanation_result

        # Always update DB
        report = DatasetReport.query.filter_by(
            dataset_id=dataset_id
        ).first()

        if report:
            report.explanation_report = explanation_result
            db.session.commit()

        return jsonify(
            {
                "dataset_id": dataset_id,
                "explanation": explanation_result,
                "message": "AI explanation generated successfully",
            }
        ), 200

    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 500


@api_bp.route("/results/<dataset_id>", methods=["GET"])
def get_results(dataset_id):
    """
    Get full results
    """
    try:
        if dataset_id not in results_store:
            return jsonify({"error": "Dataset not found"}), 404

        return jsonify(
            convert_numpy_types(
                results_store[dataset_id]
            )
        ), 200

    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 500


@api_bp.route("/datasets", methods=["GET"])
def list_datasets():
    """
    List all uploaded datasets
    """
    try:
        datasets = []

        for dataset_id, info in results_store.items():
            datasets.append(
                {
                    "dataset_id": dataset_id,
                    "filename": info["filename"],
                    "processed": info.get(
                        "processed",
                        False,
                    ),
                    "total_rows": info["stats"][
                        "total_rows"
                    ],
                    "total_columns": info["stats"][
                        "total_columns"
                    ],
                }
            )

        return jsonify(
            {
                "total_datasets": len(datasets),
                "datasets": datasets,
            }
        ), 200

    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 500