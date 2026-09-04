import asyncio
from datetime import timedelta
from typing import Optional
from temporalio import workflow
from temporalio.exceptions import ApplicationError


@workflow.defn
class ModelApprovalWorkflow:
    """
    Project 03: Signals & Queries — Human-in-the-Loop Approval

    Demonstrates:
    - @workflow.signal: Receive async messages from outside
    - @workflow.query: Read workflow state without changing it
    - workflow.wait_condition: Block until a condition is true
    - Timeout on wait_condition (approval deadline)
    - Multiple signal types (approve / reject / add_comment)

    Use Case:
    An ML model is trained and waits for a human reviewer to approve
    or reject it before deployment. The reviewer can query the status
    and send a signal with their decision and comments.
    """

    def __init__(self):
        self._status: str = "waiting_for_review"
        self._decision: Optional[str] = None
        self._comments: list[str] = []
        self._reviewer: Optional[str] = None

    # ─── Signals (async messages INTO the workflow) ───────────────────────────

    @workflow.signal
    async def approve(self, reviewer: str, comment: str = "") -> None:
        """Signal: reviewer approves the model for deployment."""
        self._decision = "approved"
        self._reviewer = reviewer
        self._status = "approved"
        if comment:
            self._comments.append(f"[{reviewer}] APPROVE: {comment}")
        workflow.logger.info(f"Model APPROVED by {reviewer}")

    @workflow.signal
    async def reject(self, reviewer: str, reason: str) -> None:
        """Signal: reviewer rejects the model."""
        self._decision = "rejected"
        self._reviewer = reviewer
        self._status = "rejected"
        self._comments.append(f"[{reviewer}] REJECT: {reason}")
        workflow.logger.info(f"Model REJECTED by {reviewer}: {reason}")

    @workflow.signal
    async def add_comment(self, comment: str) -> None:
        """Signal: reviewer adds a comment without deciding yet."""
        self._comments.append(comment)
        workflow.logger.info(f"Comment added: {comment}")

    # ─── Queries (sync reads of workflow state) ───────────────────────────────

    @workflow.query
    def get_status(self) -> str:
        """Query: get the current approval status."""
        return self._status

    @workflow.query
    def get_comments(self) -> list[str]:
        """Query: get all reviewer comments."""
        return self._comments

    @workflow.query
    def get_full_report(self) -> dict:
        """Query: get the full review report."""
        return {
            "status": self._status,
            "decision": self._decision,
            "reviewer": self._reviewer,
            "comments": self._comments,
        }

    # ─── Workflow Run ──────────────────────────────────────────────────────────

    @workflow.run
    async def run(self, model_name: str, model_metrics: dict) -> str:
        workflow.logger.info(f"Approval workflow started for model: {model_name}")
        workflow.logger.info(f"Model metrics: {model_metrics}")

        self._status = "waiting_for_review"

        # Wait for a decision signal, with a 7-day deadline
        # In this demo we set it to 2 minutes for testability
        try:
            await workflow.wait_condition(
                lambda: self._decision is not None,
                timeout=timedelta(minutes=2),
            )
        except asyncio.TimeoutError:
            self._status = "timed_out"
            return f"Model '{model_name}' approval timed out — auto-rejected after 2 minutes."

        if self._decision == "approved":
            # In a real workflow, this would trigger deployment
            return (
                f"Model '{model_name}' APPROVED by {self._reviewer}.\n"
                f"Metrics: {model_metrics}\n"
                f"Comments: {self._comments}\n"
                f"Status: Deploying to production..."
            )
        else:
            return (
                f"Model '{model_name}' REJECTED by {self._reviewer}.\n"
                f"Reason: {self._comments}\n"
                f"Status: Sent back for retraining."
            )
