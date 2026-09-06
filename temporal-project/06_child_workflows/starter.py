import asyncio
from temporalio.client import Client

from workflow import DataAnalysisPipelineWorkflow

TASK_QUEUE = "child-workflows-queue"


async def main():
    client = await Client.connect("localhost:7233")

    print("Starting Data Analysis Pipeline with Child Workflows...")
    print("This will spawn 5 child workflows in parallel!\n")

    result = await client.execute_workflow(
        DataAnalysisPipelineWorkflow.run,
        args=["sales_data_2024", 1_000_000, 5],
        id="data-analysis-001",
        task_queue=TASK_QUEUE,
    )

    print(f"\n=== Workflow Result ===\n{result}")
    print("\nCheck the Temporal UI at: http://localhost:8233")
    print("  → Workflow ID: data-analysis-001")
    print("  → See the parent workflow timeline")
    print("  → Search for child workflows: data-analysis-001-sales_data_2024-chunk-*")
    print("  → Each child has its own independent event history!")


if __name__ == "__main__":
    asyncio.run(main())
