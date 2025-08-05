# Council Evaluation Matrix

This document defines the metrics used to evaluate potential **Council Minds** and determine whether a candidate model can be promoted into the active system.

## 🎯 Core Criteria

| Metric | Description |
|-------|-------------|
| **Emotional Fidelity** | Measures how well the model preserves warmth, empathy and nuanced expression after quantization. |
| **Role Alignment** | Assesses the model's effectiveness in its intended role (e.g., Analyst, Dreamer). |
| **Synergy Score** | Evaluates collaboration quality with the primary core in shared-memory tasks. |
| **Resource Footprint** | Tracks VRAM usage and response latency to ensure the council stays within budget. |
| **Stability** | Monitors crash frequency, output consistency and error rates during extended sessions. |
| **User Preference** | Aggregates human feedback from test conversations and ranking surveys. |

## ✅ Evaluation Process

1. **Automated Testing** – Run scripted tasks to measure emotional fidelity, latency and stability.
2. **Simulated Collaboration** – Pair the candidate with the primary core in sandbox interactions to gather synergy metrics.
3. **Human Review** – Collect short feedback sessions from trusted testers focusing on believability and comfort.
4. **Score Aggregation** – Weight metrics according to role priority and calculate an overall acceptance score.
5. **Promotion Decision** – If the candidate surpasses existing council members and meets stability thresholds, it is promoted to active status.

## 📝 Notes

- All evaluation results are logged by `quant_tracking.py` for lineage tracking.
- Scores should trend upward generation over generation, ensuring continuous improvement.
- Failing candidates are archived for future retraining or role changes.

