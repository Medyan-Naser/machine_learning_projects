import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from workflow import ContinuousModelMonitorWorkflow, check_ml_model_health, send_alert

TASK_QUEUE = "continue-as-new-queue"


async def main():
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ContinuousModelMonitorWorkflow],
        activities=[check_ml_model_health, send_alert],
    )
    print(f"Continue-As-New Worker started — polling: '{TASK_QUEUE}'")
    print("Check the Temporal UI at: http://localhost:8233")
    print("  → Watch the workflow reset its history every 10 iterations!")
    print("Press Ctrl+C to stop.")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
