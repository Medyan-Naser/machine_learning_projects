import asyncio
from datetime import timedelta
from dataclasses import dataclass
from temporalio import workflow, activity
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    import time
    import random


# ─── Activities ────────────────────────────────────────────────────────────────

@activity.defn
async def analyze_dataset_chunk(chunk_id: str, data_range: dict) -> dict:
    """Analyze a chunk of a large dataset. In a real scenario, this would query a DB or S3."""
    time.sleep(0.5)  # Simulate work
    return {
        "chunk_id": chunk_id,
        "start": data_range["start"],
        "end": data_range["end"],
        "records_processed": data_range["end"] - data_range["start"],
        "anomalies_found": random.randint(0, 5),
        "mean_value": round(random.uniform(50, 150), 2),
    }


@activity.defn
async def aggregate_results(results: list[dict]) -> dict:
    """Aggregate results from all chunks into a final report."""
    total_records = sum(r["records_processed"] for r in results)
    total_anomalies = sum(r["anomalies_found"] for r in results)
    avg_mean = sum(r["mean_value"] for r in results) / len(results) if results else 0

    return {
        "total_chunks": len(results),
        "total_records_processed": total_records,
        "total_anomalies": total_anomalies,
        "overall_mean": round(avg_mean, 2),
        "anomaly_rate": round(total_anomalies / total_records * 100, 3) if total_records else 0,
    }


# ─── Child Workflow ────────────────────────────────────────────────────────────

@workflow.defn
class DataChunkWorkflow:
    """
    Child Workflow: processes a single chunk of data independently.
    Has its own retry policy, timeout, and event history.
    Can be cancelled, queried, or signalled independently of the parent.
    """

    @workflow.run
    async def run(self, chunk_id: str, data_range: dict) -> dict:
        workflow.logger.info(f"Child workflow processing chunk: {chunk_id}")

        result = await workflow.execute_activity(
            analyze_dataset_chunk,
            args=[chunk_id, data_range],
            retry_policy=RetryPolicy(maximum_attempts=3),
            start_to_close_timeout=timedelta(minutes=5),
        )

        workflow.logger.info(f"Chunk {chunk_id} complete: {result['records_processed']} records, {result['anomalies_found']} anomalies")
        return result


# ─── Parent Workflow ───────────────────────────────────────────────────────────

@workflow.defn
class DataAnalysisPipelineWorkflow:
    """
    Project 06: Child Workflows — Fan-Out / Fan-In Parallel Processing

    Demonstrates:
    - Starting multiple child workflows in parallel
    - Fan-out: distribute work across N child workflows
    - Fan-in: gather all child results and aggregate
    - Each child has independent retry/timeout
    - asyncio.gather for true parallelism

    Use Case:
    Large dataset analysis. Instead of one workflow processing 1M records
    sequentially, we split into 10 chunks of 100K records each and process
    them in parallel with child workflows. This is the Temporal equivalent
    of MapReduce or Spark partitioning.

    Key Difference from activities:
    - Activities: single unit of work, runs on one worker
    - Child Workflows: full workflows, can be long-running, independently durable
    """

    @workflow.run
    async def run(self, dataset_name: str, total_records: int, num_chunks: int) -> str:
        workflow.logger.info(
            f"Parent workflow started: dataset={dataset_name}, "
            f"records={total_records}, chunks={num_chunks}"
        )

        # ── Fan-Out: Create N chunk workflows ──────────────────────────────────
        chunk_size = total_records // num_chunks
        child_handles = []

        for i in range(num_chunks):
            chunk_id = f"{dataset_name}-chunk-{i:03d}"
            data_range = {
                "start": i * chunk_size,
                "end": (i + 1) * chunk_size if i < num_chunks - 1 else total_records,
            }

            # Start child workflow WITHOUT awaiting (fire-and-forget start)
            child_handle = await workflow.start_child_workflow(
                DataChunkWorkflow.run,
                args=[chunk_id, data_range],
                id=f"{workflow.info().workflow_id}-{chunk_id}",
                task_queue=workflow.info().task_queue,
                execution_timeout=timedelta(minutes=10),
            )
            child_handles.append(child_handle)
            workflow.logger.info(f"Started child workflow: {chunk_id}")

        workflow.logger.info(f"All {num_chunks} child workflows started — awaiting results...")

        # ── Fan-In: Wait for ALL children to complete ──────────────────────────
        chunk_results = await asyncio.gather(*child_handles)
        workflow.logger.info(f"All {num_chunks} chunks complete — aggregating...")

        # ── Aggregate ──────────────────────────────────────────────────────────
        summary = await workflow.execute_activity(
            aggregate_results,
            list(chunk_results),
            start_to_close_timeout=timedelta(minutes=2),
        )

        return (
            f"Dataset Analysis Complete: {dataset_name}\n"
            f"  Chunks processed: {summary['total_chunks']} (in PARALLEL)\n"
            f"  Total records: {summary['total_records_processed']:,}\n"
            f"  Total anomalies: {summary['total_anomalies']}\n"
            f"  Anomaly rate: {summary['anomaly_rate']}%\n"
            f"  Overall mean value: {summary['overall_mean']}"
        )
