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

