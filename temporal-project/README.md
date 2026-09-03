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
