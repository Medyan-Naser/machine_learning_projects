import asyncio
import time as _time
from temporalio.client import Client

from workflow import ModelApprovalWorkflow

TASK_QUEUE = "approval-queue"
WORKFLOW_ID = f"model-approval-{int(_time.time())}"


async def main():
    client = await Client.connect("localhost:7233")

    model_metrics = {
        "accuracy": 0.934,
        "f1_score": 0.921,
        "dataset": "fraud_detection",
        "training_samples": 8000,
    }

    print(f"Starting Model Approval Workflow for: FraudDetectionModel_v1")
    print("The workflow will WAIT for a human signal (approve/reject).\n")

    # Start workflow WITHOUT waiting for result (it's waiting for signals)
    handle = await client.start_workflow(
        ModelApprovalWorkflow.run,
        args=["FraudDetectionModel_v1", model_metrics],
        id=WORKFLOW_ID,
        task_queue=TASK_QUEUE,
    )

    print(f"Workflow started! ID: {WORKFLOW_ID}")
    print("Check the Temporal UI at: http://localhost:8233")
    print(f"  → Find workflow: {WORKFLOW_ID}")
    print(f"  → Status will show 'Running' — waiting for signal\n")

    # Demonstrate QUERY — read state without affecting workflow
    await asyncio.sleep(1)
    status = await handle.query(ModelApprovalWorkflow.get_status)
    print(f"[QUERY] Current status: {status}")

    # Add a comment via signal
    await asyncio.sleep(1)
    print("\n[SIGNAL] Sending comment signal...")
    await handle.signal(ModelApprovalWorkflow.add_comment, "Model performance looks good. Reviewing edge cases...")

    # Query again to see the comment
    await asyncio.sleep(0.5)
    comments = await handle.query(ModelApprovalWorkflow.get_comments)
    print(f"[QUERY] Comments: {comments}")

    # Simulate reviewer approving after review
    await asyncio.sleep(2)
    print("\n[SIGNAL] Sending APPROVE signal...")
    await handle.signal(ModelApprovalWorkflow.approve, args=["alice@company.com", "Excellent F1 score. Approved for production."])

    # Now wait for the workflow to complete
    result = await handle.result()
    print(f"\n=== Workflow Result ===\n{result}")

    # Final query (workflow is done, queries still work on completed workflows)
    report = await handle.query(ModelApprovalWorkflow.get_full_report)
    print(f"\n=== Full Report ===\n{report}")


if __name__ == "__main__":
    asyncio.run(main())
