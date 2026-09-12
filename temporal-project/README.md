# Temporal Learning Projects

A complete hands-on roadmap to mastering Temporal — from "Hello World" to production-grade patterns.
All projects use **Python**, the **Temporal Python SDK**, and include a **working local Temporal server with Web UI**.

---

## Quick Start

### Prerequisites

1. **Install Temporal CLI** (already done — binary at `~/.temporalio/bin/temporal`):
   ```bash
   export PATH="$PATH:/home/medy/.temporalio/bin"
   temporal --version   # should show: temporal version 1.8.x
   ```

2. **Activate the virtual environment**:
   ```bash
   cd temporal-project/
   source venv/bin/activate
   ```

### Start the Temporal Server (Required for ALL projects)

Open a **dedicated terminal** and run:
```bash
export PATH="$PATH:/home/medy/.temporalio/bin"
temporal server start-dev --db-filename temporal.db
```

This starts:
- **Temporal gRPC Server** → `localhost:7233`
- **Temporal Web UI** → [http://localhost:8233](http://localhost:8233)

> **Keep this terminal open while running any project.**
> The `--db-filename temporal.db` flag persists workflow history between server restarts.

---

## Project Overview

| # | Project | Concepts Learned | Difficulty |
|---|---------|-----------------|------------|
| [01](#01-hello-world) | Hello World | Workflow, Activity, Worker, Client | ⭐ Beginner |
| [02](#02-ml-training-pipeline) | ML Training Pipeline | Multi-step, Heartbeats, RetryPolicy, Quality Gate | ⭐⭐ Intermediate |
| [03](#03-signals--queries) | Signals & Queries | Signal, Query, wait_condition, Human-in-the-Loop | ⭐⭐ Intermediate |
| [04](#04-retry--error-handling) | Retry & Error Handling | RetryPolicy, Non-retryable errors, Heartbeats, Compensation | ⭐⭐ Intermediate |
| [05](#05-saga-pattern) | Saga Pattern | Distributed transactions, Compensations, Rollback | ⭐⭐⭐ Advanced |
| [06](#06-child-workflows) | Child Workflows | Fan-out/Fan-in, Parallel execution, asyncio.gather | ⭐⭐⭐ Advanced |
| [07](#07-cron-workflows) | Cron Workflows | Scheduled workflows, cron_schedule, Model monitoring | ⭐⭐ Intermediate |

---

## How to Run Each Project

Each project needs **2 terminals** (+ the Temporal server terminal = 3 total):
- **Terminal 1**: Temporal Server (`temporal server start-dev`)
- **Terminal 2**: Worker (`python worker.py`)
- **Terminal 3**: Starter (`python starter.py`)

---

### 01: Hello World

**What it does**: Two activities in sequence — `say_hello` then `say_goodbye`.

**Concepts**: Workflow definition, Activity execution, Worker registration, Client.

```bash
cd 01_hello_world/

# Terminal 2 — start the worker
python worker.py

# Terminal 3 — run the workflow
python starter.py
```

**What to look for in the UI** (`http://localhost:8233`):
- Find workflow ID: `hello-world-001`
- See 2 activity executions with inputs and outputs
- Inspect the full event history

---

### 02: ML Training Pipeline

**What it does**: A 5-stage ML pipeline: Load Data → Preprocess → Train → Evaluate → Register (with quality gate).

**Concepts**: Multi-step workflows, activity heartbeats, per-activity retry policies, quality gates with non-retryable errors, typed dataclass inputs/outputs.

```bash
cd 02_ml_training_pipeline/

# Terminal 2
python worker.py

# Terminal 3
python starter.py
```

**What to look for in the UI**:
- Workflow ID: `ml-pipeline-001`
- See all 5 stages with timing
- Click each activity to see inputs/outputs
- Observe heartbeat events during `train_model`

---

### 03: Signals & Queries

**What it does**: ML model review workflow that waits for a human to approve or reject via signals. Queries let you check status without affecting the workflow.

**Concepts**: `@workflow.signal`, `@workflow.query`, `workflow.wait_condition`, signal-based decision gates.

```bash
cd 03_signals_and_queries/

# Terminal 2
python worker.py

# Terminal 3
python starter.py
```

**What to look for in the UI**:
- Workflow ID: `model-approval-001`
- See the workflow in "Running" state while waiting
- After signals: see `SignalReceived` events in history
- Try sending a signal from the UI: "Send Signal" → `approve`

**Send signals manually from CLI**:
```bash
export PATH="$PATH:/home/medy/.temporalio/bin"
temporal workflow signal \
  --workflow-id model-approval-001 \
  --name approve \
  --input '"reviewer@company.com"' \
  --input '"Looks good!"'
```

---

### 04: Retry & Error Handling

**What it does**: Demonstrates automatic retries on flaky activities, non-retryable errors, heartbeats, and compensation on failure.

**Concepts**: `RetryPolicy`, `ApplicationError(non_retryable=True)`, `activity.heartbeat()`, compensation pattern.

```bash
cd 04_retry_and_errors/

# Terminal 2
python worker.py

# Terminal 3
python starter.py
```

**What to look for in the UI**:
- Workflow ID: `retry-errors-001`
- Click `flaky_data_fetch` activity → see 3 attempts (2 failures + 1 success)
- Each attempt shows the error message, timestamp, and retry delay

---

### 05: Saga Pattern

**What it does**: Order processing with 3 steps (Inventory → Payment → Shipping). If any step fails, all previous steps are compensated (rolled back) in reverse order.

**Concepts**: Saga pattern, compensation activities, reverse-order rollback, distributed transactions.

```bash
cd 05_saga_pattern/

# Terminal 2
python worker.py

# Terminal 3 — runs 3 scenarios automatically
python starter.py
```

**What to look for in the UI**:
- Compare 3 workflow executions (success, inventory-fail, payment-fail)
- In failed scenarios, see compensation activities run in reverse order

---

### 06: Child Workflows

**What it does**: Parent workflow splits a 1M-record dataset into 5 chunks and processes them in parallel using child workflows, then aggregates results.

**Concepts**: `start_child_workflow`, fan-out/fan-in, `asyncio.gather`, parallel execution, independent child histories.

```bash
cd 06_child_workflows/

# Terminal 2
python worker.py

# Terminal 3
python starter.py
```

**What to look for in the UI**:
- Parent workflow ID: `data-analysis-001`
- Search for child workflows: `data-analysis-001-*`
- Each child has its own separate event history
- The parent timeline shows all children running in parallel

---

### 07: Cron Workflows

**What it does**: ML model monitoring workflow that runs every minute, checks model drift, sends reports, and auto-triggers retraining if drift is detected.

**Concepts**: `cron_schedule`, repeated executions, workflow.now() for schedule-aware logic.

```bash
cd 07_cron_workflows/

# Terminal 2
python worker.py

# Terminal 3
python starter.py
```

**What to look for in the UI**:
- Workflow ID: `model-monitoring-cron`
- See new executions appear every minute
- Check which runs triggered retraining (drift detected)
- Cancel via UI or CLI: `temporal workflow cancel --workflow-id model-monitoring-cron`

---

## Documentation

See [`docs/temporal-guide.md`](docs/temporal-guide.md) for a comprehensive guide covering:
- Core concepts (Workflow, Activity, Worker, Signal, Query, Timer, etc.)
- Architecture deep-dive (event sourcing, replay, determinism)
- When to use / not use Temporal
- Temporal vs alternatives (Airflow, Celery, Step Functions)
- Temporal + Kubernetes deployment
- Key patterns with code examples
- Python SDK reference
- Production considerations

---

## Project Structure

```
temporal-project/
├── README.md                    ← This file
├── PROGRESS.md                  ← Work tracker
├── requirements.txt             ← Python dependencies
├── temporal.db                  ← Local Temporal DB (created at runtime)
├── venv/                        ← Virtual environment
├── docs/
│   └── temporal-guide.md        ← Complete Temporal knowledge base
├── 01_hello_world/
│   ├── activities.py
│   ├── workflow.py
│   ├── worker.py
│   └── starter.py
├── 02_ml_training_pipeline/
│   ├── activities.py
│   ├── workflow.py
│   ├── worker.py
│   └── starter.py
├── 03_signals_and_queries/
│   ├── workflow.py
│   ├── worker.py
│   └── starter.py
├── 04_retry_and_errors/
│   ├── activities.py
│   ├── workflow.py
│   ├── worker.py
│   └── starter.py
├── 05_saga_pattern/
│   ├── activities.py
│   ├── workflow.py
│   ├── worker.py
│   └── starter.py
├── 06_child_workflows/
│   ├── workflow.py
│   ├── worker.py
│   └── starter.py
└── 07_cron_workflows/
    ├── workflow.py
    ├── worker.py
    └── starter.py
```

---

## Common Commands

```bash
# Temporal CLI (add to PATH first)
export PATH="$PATH:/home/medy/.temporalio/bin"

# Start dev server
temporal server start-dev --db-filename temporal.db

# List all workflows
temporal workflow list

# Describe a workflow
temporal workflow describe --workflow-id <id>

# View full event history
temporal workflow show --workflow-id <id>

# Send a signal
temporal workflow signal --workflow-id <id> --name <signal-name>

# Cancel a workflow
temporal workflow cancel --workflow-id <id>

# Terminate a workflow (force kill)
temporal workflow terminate --workflow-id <id>
```

---

## Troubleshooting

**"Connection refused" / "Failed to connect"**
→ Temporal server is not running. Start it: `temporal server start-dev`

**"Task queue has no pollers"**
→ Worker is not running. Run `python worker.py` in the project directory.

**"Workflow already exists"**
→ The workflow ID is already in use. Either:
  - Change the ID in `starter.py`, or
  - Terminate the existing one: `temporal workflow terminate --workflow-id <id>`

**Activity retries not showing**
→ Make sure you're looking at the correct workflow in the UI and expanding the activity events.

**Python import errors**
→ Make sure the venv is activated: `source venv/bin/activate`
