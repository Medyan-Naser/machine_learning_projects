import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from workflow import ModelApprovalWorkflow

TASK_QUEUE = "approval-queue"


async def main():
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ModelApprovalWorkflow],
    )
    print(f"Approval Worker started — polling: '{TASK_QUEUE}'")
    print("Check the Temporal UI at: http://localhost:8233")
    print("Press Ctrl+C to stop.")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
