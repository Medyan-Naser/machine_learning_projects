import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from activities import (
    load_and_validate_data,
    preprocess_data,
    train_model,
    evaluate_model,
    register_model,
)
from workflow import MLTrainingPipelineWorkflow

TASK_QUEUE = "ml-pipeline-queue"


async def main():
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MLTrainingPipelineWorkflow],
        activities=[
            load_and_validate_data,
            preprocess_data,
            train_model,
            evaluate_model,
            register_model,
        ],
    )
    print(f"ML Pipeline Worker started — polling: '{TASK_QUEUE}'")
    print("Check the Temporal UI at: http://localhost:8233")
    print("Press Ctrl+C to stop.")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
