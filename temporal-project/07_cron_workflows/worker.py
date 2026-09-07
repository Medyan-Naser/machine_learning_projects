import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from workflow import (
    ModelMonitoringWorkflow,
    collect_model_metrics,
    check_model_drift,
    send_monitoring_report,
    trigger_retraining_pipeline,
)

TASK_QUEUE = "cron-queue"


async def main():
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ModelMonitoringWorkflow],
        activities=[
            collect_model_metrics,
            check_model_drift,
            send_monitoring_report,
            trigger_retraining_pipeline,
        ],
    )
    print(f"Cron Worker started — polling: '{TASK_QUEUE}'")
    print("Check the Temporal UI at: http://localhost:8233")
    print("  → Cron workflow runs every minute (for demo purposes)")
    print("  → Each run creates a new execution visible in the UI")
    print("Press Ctrl+C to stop.")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
