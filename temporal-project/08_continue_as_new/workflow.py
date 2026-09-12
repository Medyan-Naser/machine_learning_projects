import asyncio
from datetime import timedelta
from temporalio import workflow, activity

with workflow.unsafe.imports_passed_through():
    import time
    import random


# ─── Activities ────────────────────────────────────────────────────────────────

@activity.defn
async def check_ml_model_health(model_name: str) -> dict:
    """Check model health metrics — simulates calling a monitoring API."""
    time.sleep(0.2)
    return {
        "model": model_name,
        "accuracy": round(random.uniform(0.88, 0.97), 4),
        "latency_ms": round(random.uniform(40, 120), 1),
        "requests_per_min": random.randint(50, 500),
        "drift_score": round(random.uniform(0.01, 0.20), 4),
        "healthy": random.random() > 0.1,  # 10% chance of unhealthy
    }


@activity.defn
async def send_alert(model_name: str, iteration: int, metrics: dict) -> str:
    """Send an alert if the model is unhealthy."""
    msg = (
        f"[ALERT] Model '{model_name}' is unhealthy on iteration #{iteration}!\n"
        f"  Accuracy: {metrics['accuracy']}\n"
        f"  Drift: {metrics['drift_score']}"
    )
    activity.logger.warning(msg)
    return msg


# ─── Workflow ──────────────────────────────────────────────────────────────────

HISTORY_RESET_THRESHOLD = 10  # Reset every 10 iterations for demo (use 1000 in prod)


@workflow.defn
class ContinuousModelMonitorWorkflow:
    """
    Project 08: Continue-As-New — Infinite Long-Running Workflow

    Demonstrates:
    - workflow.continue_as_new() to reset event history before it hits limits
    - Passing forward state (iteration count, cumulative stats) across resets
    - Why Continue-As-New is necessary for production workflows
    - The pattern for infinite polling/monitoring workflows

    The Problem Without Continue-As-New:
    Temporal stores every event (activity scheduled, started, completed, etc.).
    A workflow running 24/7 would hit the 51,200 event limit in days/weeks.
    Once the limit is hit, the workflow FAILS permanently.

    The Solution:
    Every HISTORY_RESET_THRESHOLD iterations, call continue_as_new().
    Temporal starts a FRESH execution with a clean history, passing forward
    only the minimal state we care about (iteration count, cumulative stats).

    K8s/Charm relevance:
    The temporal-worker-k8s charm runs workers that may run infinite workflows.
    Understanding Continue-As-New is essential for operating Temporal correctly.
    """

    @workflow.run
    async def run(
        self,
        model_name: str,
        iteration: int = 0,
        total_alerts: int = 0,
        run_count: int = 0,
    ) -> str:
        run_count += 1
        workflow.logger.info(
            f"Monitor run #{run_count} | iteration={iteration} | "
            f"total_alerts={total_alerts} | "
            f"history_reset_every={HISTORY_RESET_THRESHOLD}"
        )

        for i in range(HISTORY_RESET_THRESHOLD):
            current_iteration = iteration + i
            workflow.logger.info(f"Checking health — iteration #{current_iteration}")

            metrics = await workflow.execute_activity(
                check_ml_model_health,
                model_name,
                start_to_close_timeout=timedelta(seconds=30),
            )

            if not metrics["healthy"]:
                total_alerts += 1
                await workflow.execute_activity(
                    send_alert,
                    args=[model_name, current_iteration, metrics],
                    start_to_close_timeout=timedelta(seconds=10),
                )

            # In production you'd sleep between checks; using 0.1s for demo
            await workflow.sleep(timedelta(milliseconds=100))

        new_iteration = iteration + HISTORY_RESET_THRESHOLD
        workflow.logger.info(
            f"Reached {HISTORY_RESET_THRESHOLD} iterations — calling continue_as_new. "
            f"History will reset. Passing forward: iteration={new_iteration}, "
            f"total_alerts={total_alerts}, run_count={run_count}"
        )

        # This raises ContinueAsNew — a fresh execution starts with these args.
        # The current run's history is archived; new run starts with event count = 1.
        workflow.continue_as_new(
            args=[model_name, new_iteration, total_alerts, run_count],
        )

        # This line never executes — continue_as_new raises immediately
        return "unreachable"
