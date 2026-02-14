# CrewAI

## Overview
CrewAI is a framework for building agentic systems as a **team of role-based agents** (“a crew”) that collaborate to complete a goal.

A good mental model is a project team:

- Each agent has a **role** (planner, researcher, writer, reviewer, etc.)
- Agents can have **tools** to take actions (search, code execution, database access)
- Work is organized as **tasks** toward a shared **goal**

This is useful when one agent is not enough, or when you want specialization, separation of concerns, and more predictable workflows.

## Core concepts

### Roles
Agents are typically defined with a role and a goal-aligned instruction set. Roles help:

- Constrain behavior
- Improve reliability (specialization)
- Make multi-step workflows easier to debug

### Tools
Tools let agents act beyond text generation (search, retrieval, APIs, code execution).

A key design decision is how to assign tools.

#### Agent-centric tools (generalist)
You give the agent a “toolbox” containing all tools it might need. The agent decides at runtime.

- **Pros**: flexible
- **Cons**: can be less predictable; agent may misuse tools

#### Task-centric tools (specialist)
You attach tools to the specific tasks that require them. The agent only gets the tools needed for the current task.

- **Pros**: more focused and predictable
- **Cons**: more upfront workflow design

### Delegation (`allow_delegation`)
Some CrewAI setups allow agents to delegate sub-questions to other agents.

Example pattern:

- A “Writer” agent has `allow_delegation=True`
- While drafting, it delegates targeted research questions to a “Research Analyst” agent
- The analyst returns findings
- The writer integrates those findings into the final output

This can improve quality while keeping each agent focused on its specialty.

### `@CrewBase`
In CrewAI, `@CrewBase` is a Python class decorator used to automate collection and wiring of agents and tasks. This is especially helpful when organizing crew logic in code, YAML config, or a hybrid approach.

## Integrations: Serper
Serper is commonly used as a real-time Google Search API integration so agents can access up-to-date web information.

In practice, this is often used as a “search tool” in a crew so research-oriented agents can ground outputs in current sources.
