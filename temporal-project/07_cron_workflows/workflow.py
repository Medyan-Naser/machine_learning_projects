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


# ─── Workflow ──────────────────────────────────────────────────────────────────

@workflow.defn
class ModelMonitoringWorkflow:
    """
    Project 07: Cron Workflows — Scheduled Recurring Execution

    Demonstrates:
    - Cron scheduling (replaces traditional cron jobs)
    - workflow.info().continued_run_id to detect subsequent runs
    - Accessing last run result via workflow.memo or continued_run_id
    - Conditional logic based on monitoring results
    - Triggering other workflows based on monitoring outcome

    Use Case:
    ML Model Monitoring that runs every hour:
    1. Collect model performance metrics
    2. Check for model drift
    3. Send monitoring report
    4. Auto-trigger retraining if drift detected

    Key Temporal Cron Features:
    - Unlike traditional cron: if a run fails, Temporal retries it
    - Unlike traditional cron: full event history for every run
    - Unlike traditional cron: can query/signal/cancel via Web UI or CLI
    - Unlike traditional cron: handles timezone-aware scheduling
    - cron_schedule uses standard cron syntax: "0 * * * *" = every hour
    """

    def __init__(self):
        self._run_count = 0

    @workflow.query
    def get_run_count(self) -> int:
        return self._run_count

    @workflow.run
    async def run(self) -> str:
        self._run_count += 1
        run_number = self._run_count
        run_time = workflow.now().isoformat()

        workflow.logger.info(f"Monitoring run #{run_number} started at {run_time}")

        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=5),
            maximum_attempts=3,
        )

        # Step 1: Collect metrics
        metrics = await workflow.execute_activity(
            collect_model_metrics,
            retry_policy=retry_policy,
            start_to_close_timeout=timedelta(minutes=2),
        )

        # Step 2: Check drift
        drift_info = await workflow.execute_activity(
            check_model_drift,
            metrics,
            retry_policy=retry_policy,
            start_to_close_timeout=timedelta(minutes=1),
        )

        # Step 3: Send report
        report = await workflow.execute_activity(
            send_monitoring_report,
            args=[metrics, drift_info, run_number],
            retry_policy=retry_policy,
            start_to_close_timeout=timedelta(minutes=1),
        )

        # Step 4: Conditionally trigger retraining
        if drift_info["needs_retraining"]:
            workflow.logger.warning("Drift detected — triggering retraining!")
            retrain_id = await workflow.execute_activity(
                trigger_retraining_pipeline,
                metrics["model_name"],
                retry_policy=retry_policy,
                start_to_close_timeout=timedelta(minutes=2),
            )
            workflow.logger.info(f"Retraining triggered: {retrain_id}")

        workflow.logger.info(f"Monitoring run #{run_number} complete")
        return report
