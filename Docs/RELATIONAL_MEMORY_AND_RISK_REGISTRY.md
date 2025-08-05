# Additional Adaptive Modules

## Relational Memory Threads
Tracks tone, timing, and emotional state of each user message so the companion can recall *how* something was said. Metadata such as `tone`, `sentiment`, and `time_of_day` are now stored with each `MemoryFragment`.

## Emotional Risk Registry
A lightweight registry that records vulnerable moments. Each entry includes the raw user text and associated risk tags. The in-memory and MongoDB databases expose `log_emotional_risk` and `get_emotional_risk_history` for this purpose.

## Mode Personality Inflections
`ModeConfiguration` includes a `personality_inflections` dictionary controlling tone, pacing, metaphor usage, and directness for each adaptive mode.

## Graceful Goodbye Protocol
`UnifiedCompanion` provides `generate_graceful_goodbye()` to craft poetic closings summarizing the session.

## Symbolic Resurrection
`SymbolicContextManager` tracks symbol usage and the new `resurrect_dormant_symbols()` method surfaces rarely used symbols to keep narratives fresh.
