from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities import (
        DatasetInfo,
        TrainingConfig,
        ModelMetrics,
        load_and_validate_data,
        preprocess_data,
        train_model,
        evaluate_model,
        register_model,
    )


@workflow.defn
class MLTrainingPipelineWorkflow:
    """
    Project 02: ML Training Pipeline

    Demonstrates:
    - Multi-step orchestration (5 activities in sequence)
    - Heartbeats for long-running activities
    - RetryPolicy configuration per activity
    - Quality gate pattern (activity raises non-retryable error)
    - Dataclasses as typed activity inputs/outputs
    - start_to_close_timeout for each step

    Pipeline stages:
      1. Load & Validate Data
      2. Preprocess Data
      3. Train Model
      4. Evaluate Model
      5. Register Model (quality gate: accuracy > 0.85)
    """

    @workflow.run
    async def run(self, dataset_name: str, model_name: str) -> str:
        workflow.logger.info(f"ML Pipeline started: dataset={dataset_name}, model={model_name}")

        config = TrainingConfig(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            test_size=0.2,
        )

        standard_retry = RetryPolicy(
            initial_interval=timedelta(seconds=2),
            backoff_coefficient=2.0,
            maximum_interval=timedelta(minutes=1),
            maximum_attempts=3,
        )

        # Stage 1: Load & Validate
        workflow.logger.info("Stage 1/5: Loading dataset...")
        dataset_info: DatasetInfo = await workflow.execute_activity(
            load_and_validate_data,
            dataset_name,
            retry_policy=standard_retry,
            start_to_close_timeout=timedelta(minutes=5),
            heartbeat_timeout=timedelta(seconds=30),
        )
        workflow.logger.info(f"Dataset: {dataset_info.num_samples} samples, {dataset_info.num_features} features")

        # Stage 2: Preprocess
        workflow.logger.info("Stage 2/5: Preprocessing data...")
        await workflow.execute_activity(
            preprocess_data,
            args=[dataset_name, config],
            retry_policy=standard_retry,
            start_to_close_timeout=timedelta(minutes=5),
            heartbeat_timeout=timedelta(seconds=30),
        )

        # Stage 3: Train
        workflow.logger.info("Stage 3/5: Training model...")
        model_path: str = await workflow.execute_activity(
            train_model,
            config,
            retry_policy=standard_retry,
            start_to_close_timeout=timedelta(minutes=30),
            heartbeat_timeout=timedelta(seconds=60),
        )

        # Stage 4: Evaluate
        workflow.logger.info("Stage 4/5: Evaluating model...")
        metrics: ModelMetrics = await workflow.execute_activity(
            evaluate_model,
            model_path,
            retry_policy=standard_retry,
            start_to_close_timeout=timedelta(minutes=10),
            heartbeat_timeout=timedelta(seconds=30),
        )
        workflow.logger.info(
            f"Metrics → Accuracy: {metrics.accuracy}, F1: {metrics.f1_score}"
        )

        # Stage 5: Register (quality gate — non-retryable on failure)
        workflow.logger.info("Stage 5/5: Registering model...")
        registration_result: str = await workflow.execute_activity(
            register_model,
            args=[metrics, model_name],
            retry_policy=RetryPolicy(maximum_attempts=1),  # No retry for quality gate
            start_to_close_timeout=timedelta(minutes=2),
        )

        summary = (
            f"ML Pipeline completed successfully!\n"
            f"  Dataset: {dataset_name} ({dataset_info.num_samples} samples)\n"
            f"  Model: {model_name}\n"
            f"  Accuracy: {metrics.accuracy}\n"
            f"  Precision: {metrics.precision}\n"
            f"  Recall: {metrics.recall}\n"
            f"  F1 Score: {metrics.f1_score}\n"
            f"  Registration: {registration_result}"
        )
        workflow.logger.info(summary)
        return summary
