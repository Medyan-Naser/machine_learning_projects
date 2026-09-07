import asyncio
from temporalio.client import Client
from temporalio.service import RPCError

from workflow import ModelMonitoringWorkflow

TASK_QUEUE = "cron-queue"
WORKFLOW_ID = "model-monitoring-cron"

# WHY this starter does NOT call handle.result():
#
# Cron workflows never "terminate" — each completed run immediately continues
# to the next scheduled execution. Calling handle.result() would block forever.
#
# Instead we: start → poll describe() to detect first run → exit.
# The cron keeps running in the background until explicitly cancelled.


async def main():
    client = await Client.connect("localhost:7233")

    print("Starting Model Monitoring Cron Workflow...")
    print("Schedule: every minute (use '0 * * * *' for hourly in production)\n")

    # Try to cancel any existing cron with the same ID first
    try:
        existing = client.get_workflow_handle(WORKFLOW_ID)
        await existing.cancel()
        await asyncio.sleep(1)
        print(f"Cancelled previous cron workflow: {WORKFLOW_ID}")
    except Exception:
        pass  # No existing workflow, that's fine

    handle = await client.start_workflow(
        ModelMonitoringWorkflow.run,
        id=WORKFLOW_ID,
        task_queue=TASK_QUEUE,
        cron_schedule="* * * * *",  # Every minute (use "0 * * * *" for hourly in prod)
    )

    print(f"Cron workflow started! ID: {WORKFLOW_ID}")
    print()
    print("NOTE: Temporal cron first run fires at the NEXT minute boundary.")
    print("      Each run gets its own run_id. Watching for completed runs...\n")
    print("      (The worker terminal will show activity output when runs execute.)\n")

    # Poll via list_workflows to detect completed executions for this workflow_id.
    # describe() always shows the NEXT scheduled run (history_length=1), so we
    # must query the history of CLOSED (completed) executions instead.
    for attempt in range(90):
        await asyncio.sleep(1)
        completed = [
            wf async for wf in client.list_workflows(
                f'WorkflowId = "{WORKFLOW_ID}" AND ExecutionStatus = "Completed"'
            )
        ]
        print(f"  [{attempt+1:02d}s] completed runs so far: {len(completed)}", end="\r")
        if len(completed) >= 1:
            print(f"\n\n✅ First cron run completed! Total completed runs: {len(completed)}")
            break
    else:
        print(f"\n\n⏳ Timed out waiting — the cron is scheduled. Check the UI for activity.")

    print(f"\nCheck the Temporal UI at: http://localhost:8233")
    print(f"  → Search workflow ID: {WORKFLOW_ID}")
    print(f"  → Status = 'Running' (always — it's a cron)")
    print(f"  → Each minute: a new execution appears in the history list")
    print(f"\nTo stop the cron:")
    print(f"  export PATH=\"$PATH:/home/medy/.temporalio/bin\"")
    print(f"  temporal workflow cancel --workflow-id {WORKFLOW_ID}")


if __name__ == "__main__":
    asyncio.run(main())
