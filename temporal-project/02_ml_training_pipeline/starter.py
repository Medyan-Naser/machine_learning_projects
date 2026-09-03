import asyncio
from temporalio.client import Client

from workflow import MLTrainingPipelineWorkflow

TASK_QUEUE = "ml-pipeline-queue"


async def main():
    client = await Client.connect("localhost:7233")

    print("Starting ML Training Pipeline Workflow...")
    print("This runs 5 stages: Load → Preprocess → Train → Evaluate → Register\n")

    result = await client.execute_workflow(
        MLTrainingPipelineWorkflow.run,
        args=["fraud_detection_dataset", "FraudDetectionModel_v1"],
        id="ml-pipeline-001",
        task_queue=TASK_QUEUE,
    )

    print(f"\n=== Pipeline Result ===\n{result}")
    print("\nCheck the Temporal UI at: http://localhost:8233")
    print("  → Find workflow ID: ml-pipeline-001")
    print("  → See all 5 activity executions with inputs/outputs")
    print("  → Inspect heartbeats and retry policies")


if __name__ == "__main__":
    asyncio.run(main())
