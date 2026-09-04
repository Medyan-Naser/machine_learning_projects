import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from activities import flaky_data_fetch, validate_data, process_with_heartbeat, compensate_cleanup
from workflow import RetryAndErrorWorkflow

TASK_QUEUE = "retry-errors-queue"


async def main():
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[RetryAndErrorWorkflow],
        activities=[flaky_data_fetch, validate_data, process_with_heartbeat, compensate_cleanup],
    )
    print(f"Retry & Error Worker started — polling: '{TASK_QUEUE}'")
    print("Check the Temporal UI at: http://localhost:8233")
    print("  → Watch 'flaky_data_fetch' fail and auto-retry!")
    print("Press Ctrl+C to stop.")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
