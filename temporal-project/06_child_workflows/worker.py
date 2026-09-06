import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from workflow import DataAnalysisPipelineWorkflow, DataChunkWorkflow, analyze_dataset_chunk, aggregate_results

TASK_QUEUE = "child-workflows-queue"


async def main():
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DataAnalysisPipelineWorkflow, DataChunkWorkflow],
        activities=[analyze_dataset_chunk, aggregate_results],
    )
    print(f"Child Workflows Worker started — polling: '{TASK_QUEUE}'")
    print("Check the Temporal UI at: http://localhost:8233")
    print("  → See parent workflow spawn N child workflows in parallel!")
    print("Press Ctrl+C to stop.")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
