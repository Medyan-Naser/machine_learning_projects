import asyncio
from temporalio.client import Client

from workflow import HelloWorldWorkflow

TASK_QUEUE = "hello-world-queue"


async def main():
    client = await Client.connect("localhost:7233")

    print("Starting HelloWorldWorkflow...")
    result = await client.execute_workflow(
        HelloWorldWorkflow.run,
        "Temporal Learner",
        id="hello-world-001",
        task_queue=TASK_QUEUE,
    )

    print(f"\n=== Workflow Result ===\n{result}")
    print("\nCheck the Temporal UI at: http://localhost:8233")
    print("  → Find workflow ID: hello-world-001")
    print("  → Inspect activity inputs/outputs and event history")


if __name__ == "__main__":
    asyncio.run(main())
