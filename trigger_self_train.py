#!/usr/bin/env python3
"""Self-Training Trigger

Invoked by :mod:`idle_watchdog` when the system has been idle long enough.
This module gathers recent memories, runs the reflection engine, and optionally
notifies an n8n workflow.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, List

import aiohttp

from memory_system import MemorySystem
from reflection_engine import initialize_reflection_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENABLE_N8N = os.getenv("ENABLE_N8N", "False").lower() == "true"
N8N_WEBHOOK = os.getenv("N8N_WEBHOOK", "http://localhost:5678/webhook/self_train_trigger")


async def trigger_self_training() -> None:
    """Launch the reflection engine using recent memories."""
    memory_system = MemorySystem()
    engine = initialize_reflection_engine(memory_system, None)

    try:
        memory_batch: List[Any] = memory_system.get_recent_memories()
    except AttributeError:
        memory_batch = memory_system.short_term_memory.get("recent_interactions", [])[-100:]

    if hasattr(engine, "run"):
        await engine.run(memory_batch)
    else:
        # Fallback for older engines
        if hasattr(engine, "_generate_reflections"):
            engine.reflection_queue.extend(memory_batch)
            await engine._generate_reflections()

    logger.info("Self-training routine completed")

    if ENABLE_N8N:
        await _dispatch_n8n_webhook(len(memory_batch))


async def _dispatch_n8n_webhook(count: int) -> None:
    """Send a webhook notification to n8n if integration is enabled."""
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(N8N_WEBHOOK, json={"processed_memories": count})
        logger.info("n8n webhook dispatched")
    except Exception as exc:
        logger.error("Failed to dispatch n8n webhook: %s", exc)


if __name__ == "__main__":
    asyncio.run(trigger_self_training())
