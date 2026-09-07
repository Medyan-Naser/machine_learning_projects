import asyncio
from datetime import timedelta
from temporalio import workflow, activity
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    import time
    import random


# ─── Activities ────────────────────────────────────────────────────────────────

@activity.defn
async def collect_model_metrics() -> dict:
    """Collect current model performance metrics from monitoring system."""
    time.sleep(0.3)
    # Simulate metrics collection
    return {
        "model_name": "FraudDetectionModel_v1",
        "requests_per_hour": random.randint(800, 1200),
        "avg_latency_ms": round(random.uniform(45, 120), 2),
        "accuracy": round(random.uniform(0.88, 0.97), 4),
        "precision": round(random.uniform(0.85, 0.95), 4),
        "recall": round(random.uniform(0.87, 0.96), 4),
        "drift_score": round(random.uniform(0.01, 0.15), 4),
    }


@activity.defn
async def check_model_drift(metrics: dict) -> dict:
    """Check if model has drifted beyond acceptable threshold."""
    time.sleep(0.2)
    drift_threshold = 0.10
    drift_detected = metrics["drift_score"] > drift_threshold
    accuracy_degraded = metrics["accuracy"] < 0.90

    return {
        "drift_detected": drift_detected,
        "accuracy_degraded": accuracy_degraded,
        "needs_retraining": drift_detected or accuracy_degraded,
        "drift_score": metrics["drift_score"],
        "threshold": drift_threshold,
    }


@activity.defn
async def send_monitoring_report(metrics: dict, drift_info: dict, run_number: int) -> str:
    """Send monitoring report to the team (email/Slack/PagerDuty)."""
    time.sleep(0.1)
    status = "⚠️ ACTION REQUIRED" if drift_info["needs_retraining"] else "✅ Healthy"
    report = (
        f"[Run #{run_number}] Model Monitoring Report\n"
        f"  Status: {status}\n"
        f"  Accuracy: {metrics['accuracy']:.4f}\n"
        f"  Drift Score: {drift_info['drift_score']:.4f} (threshold: {drift_info['threshold']})\n"
        f"  Requests/hr: {metrics['requests_per_hour']}\n"
        f"  Avg Latency: {metrics['avg_latency_ms']}ms\n"
        f"  Needs Retraining: {drift_info['needs_retraining']}"
    )
    activity.logger.info(f"Report sent:\n{report}")
    return report


@activity.defn
async def trigger_retraining_pipeline(model_name: str) -> str:
    """Trigger retraining if drift detected."""
    time.sleep(0.2)
    workflow_id = f"retrain-{model_name}-{int(time.time())}"
    activity.logger.info(f"Triggered retraining workflow: {workflow_id}")
    return workflow_id

