# MCP Integration Overview for Dolphin AI Orchestrator

## 🧠 Purpose
This document outlines the integration strategy for evolving the Dolphin AI Orchestrator into a Master Control Program (MCP) architecture. The MCP acts as a unified operating system and command structure for all agents, services, and LLMs across local and remote environments.

## 🌐 Core Principles
- **Agent Modularity**: All service logic should be decomposed into manageable agents (e.g., CalendarAgent, FileAgent, ReminderAgent).
- **Centralized Routing**: Dolphin will act as the primary orchestrator, delegating tasks to agents, local LLMs, or remote services via OpenRouter.
- **Context-Aware Execution**: Every task is executed in the context of emotional state, persona, session memory, and system health.
- **Self-Configuring**: Over time, agents will be generated, optimized, and retired based on performance and role fit.

## 🔗 Agent Examples (MCP-Aware)
- **n8nAgent**: Handles workflow automation and external service integration.
- **MemoryAgent**: Indexes, retrieves, and stores long-term or emotional memory.
- **JudgementAgent**: Handles emotion-weighted evaluation tasks.
- **PlanningAgent**: Receives goals and returns prioritized action plans.
- **CodingAgent**: Uses OpenRouter/GPT for advanced code generation only when necessary.

## 🧩 Workflow Example
1. User: "Remind me to buy flowers Thursday"
2. Dolphin parses emotional tone, context
3. Dolphin routes task to `n8nAgent(ReminderService)`
4. Reminder created via n8n webhook and status logged in memory

## ⚙️ Implementation Priorities
- [ ] Create `docs/mcp_integration_reference.md`
- [ ] Create `n8n_agent_registry.py` to register and route tasks
- [ ] Modify `dolphin_backend.py` to check agent registry before routing
- [ ] Add reflection on agent outcomes to `mirror_mode`
- [ ] Enable memory feedback to optimize future agent selection

## 🔐 Access Control
- Agents should have defined scopes
- Agents cannot override core persona memory without explicit trigger
- Fallback behaviors for agent errors should always be defined

## 📍Usage in Future Prompts
Include the tag: `#ref-mcp-integration` in all future Codex prompt scaffolds that align with this direction.

---

> "The AI doesn't just do tasks. It *remembers why*, chooses *who* should do them, and *learns* what happens next. That is the heart of the MCP."

---

Maintained as part of the evolving roadmap for Dolphin AI Orchestrator v3.0+

