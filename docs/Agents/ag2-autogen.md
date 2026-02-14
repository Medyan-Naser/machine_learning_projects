# AG2 (AutoGen)

## Overview
AG2 (AutoGen) is a framework for building multi-agent systems that can solve complex tasks by breaking them down into smaller subtasks and coordinating specialized agents.

A common pattern is:

- A coordinator agent decomposes a goal into steps
- Sub-agents execute parts of the work
- The system iterates via conversation until a completion condition is reached

## Key building blocks

### ConversableAgent
`ConversableAgent` is a foundational building block designed for message-driven collaboration.

Key characteristics:

- **Communication**: can send and receive messages
- **Processing**: interprets context and generates responses
- **Personality/behavior**: controlled by system instructions
- **Extensibility**: base for specialized agent types

### GroupChat
`GroupChat` defines a set of agents and how they interact in a shared conversation.

It typically includes:

- **Agents**: list of AI (and sometimes human) participants
- **Speaker selection method**: policy for choosing who speaks next

Common speaker selection options:

- `"auto"`: choose the most contextually appropriate agent
- `"round_robin"`: fixed turn-taking order
- `"manual"`: a human chooses the next speaker
- `"random"`: random next speaker

## Where this fits in your docs
- For general agent design patterns (Reflection/Routing/Parallelization), see `workflow-patterns.md`.
- For LangChain/LangGraph agent types and execution loops, see `langchain.md`.
