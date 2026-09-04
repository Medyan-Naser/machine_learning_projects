import asyncio
from temporalio.client import Client

from workflow import RetryAndErrorWorkflow

TASK_QUEUE = "retry-errors-queue"


async def main():
    client = await Client.connect("localhost:7233")

    print("Starting Retry & Error Handling Workflow...")
    print("Watch in the Temporal UI: flaky_data_fetch will fail 2x before succeeding!\n")

    result = await client.execute_workflow(
        RetryAndErrorWorkflow.run,
        "https://api.example.com/data",
        id="retry-errors-001",
        task_queue=TASK_QUEUE,
    )

    print(f"\n=== Workflow Result ===\n{result}")
    print("\nCheck the Temporal UI at: http://localhost:8233")
    print("  → Workflow ID: retry-errors-001")
    print("  → Click on 'flaky_data_fetch' activity → see multiple retry attempts")
    print("  → Each attempt shows the error message and timestamp")


if __name__ == "__main__":
    asyncio.run(main())
