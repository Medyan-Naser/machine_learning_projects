import asyncio
from temporalio.client import Client

from workflow import ContinuousModelMonitorWorkflow, HISTORY_RESET_THRESHOLD

TASK_QUEUE = "continue-as-new-queue"
WORKFLOW_ID = "model-monitor-infinite"


async def main():
    client = await Client.connect("localhost:7233")

    print("Continue-As-New Demo — Infinite Model Monitoring Workflow")
    print(f"History resets every {HISTORY_RESET_THRESHOLD} iterations.\n")

    # Cancel any existing instance
    try:
        h = client.get_workflow_handle(WORKFLOW_ID)
        await h.terminate(reason="Restarting demo")
        await asyncio.sleep(1)
    except Exception:
        pass

    handle = await client.start_workflow(
        ContinuousModelMonitorWorkflow.run,
        args=["ProductionModel_v2", 0, 0, 0],
        id=WORKFLOW_ID,
        task_queue=TASK_QUEUE,
    )
    print(f"Workflow started: {WORKFLOW_ID}")
    print("Letting it run through 3 continue-as-new resets (~30 iterations)...\n")

    # Watch it reset 3 times (30 iterations at 100ms each = ~3 seconds)
    completed_runs = set()
    for _ in range(40):
        await asyncio.sleep(0.5)
        closed = [
            wf async for wf in client.list_workflows(
                f'WorkflowId = "{WORKFLOW_ID}" AND ExecutionStatus = "ContinuedAsNew"'
            )
        ]
        if len(closed) > len(completed_runs):
            new_runs = len(closed)
            print(f"✅ Continue-As-New triggered! Total resets so far: {new_runs}")
            completed_runs = set(range(new_runs))
            if new_runs >= 3:
                break

    print(f"\n{'='*60}")
    print(f"Observed {len(completed_runs)} Continue-As-New resets!")
    print("\nIn the Temporal UI (http://localhost:8233):")
    print(f"  → Search: WorkflowId = \"{WORKFLOW_ID}\"")
    print(f"  → You'll see multiple executions, each with status 'ContinuedAsNew'")
    print(f"  → Each execution has a SMALL history (only {HISTORY_RESET_THRESHOLD} iterations)")
    print(f"  → Without this: one run would accumulate thousands of events and FAIL!")
    print(f"\nWorkflow is still running. To stop it:")
    print(f"  temporal workflow terminate --workflow-id {WORKFLOW_ID}")


if __name__ == "__main__":
    asyncio.run(main())
