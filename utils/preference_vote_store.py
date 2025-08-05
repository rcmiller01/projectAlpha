import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)

# Label for internal reference
EMOTION_EVAL_LABEL = "Emotion Eval Mode"


@dataclass
class PreferenceVote:
    prompt: str
    response_a: str
    response_b: str
    winner: str  # 'a' or 'b'
    timestamp: str


class PreferenceVoteStore:
    """Store preference votes using SQLite and/or JSONL."""

    def __init__(self, storage_dir: str = "data", use_sqlite: bool = True, use_jsonl: bool = True):
        self.storage_path = Path(storage_dir)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.use_sqlite = use_sqlite
        self.use_jsonl = use_jsonl
        self.sqlite_path = self.storage_path / "preference_votes.db"
        self.jsonl_path = self.storage_path / "preference_votes.jsonl"

        logger.info("%s initialized", EMOTION_EVAL_LABEL)

    async def initialize(self) -> None:
        if self.use_sqlite:
            await self._init_sqlite()

    async def _init_sqlite(self) -> None:
        async with aiosqlite.connect(self.sqlite_path.as_posix()) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS preference_votes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response_a TEXT NOT NULL,
                    response_b TEXT NOT NULL,
                    winner TEXT NOT NULL
                )
                """
            )
            await db.commit()
        logger.info("PreferenceVoteStore SQLite initialized at %s", self.sqlite_path)

    async def record_vote(self, prompt: str, response_a: str, response_b: str, winner: str) -> bool:
        vote = PreferenceVote(
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            winner=winner,
            timestamp=datetime.utcnow().isoformat(),
        )
        try:
            if self.use_sqlite:
                await self._save_sqlite(vote)
            if self.use_jsonl:
                self._save_jsonl(vote)
            return True
        except Exception as exc:
            logger.error("Failed to record preference vote: %s", exc)
            return False

    async def _save_sqlite(self, vote: PreferenceVote) -> None:
        async with aiosqlite.connect(self.sqlite_path.as_posix()) as db:
            await db.execute(
                """
                INSERT INTO preference_votes (timestamp, prompt, response_a, response_b, winner)
                VALUES (?, ?, ?, ?, ?)
                """,
                (vote.timestamp, vote.prompt, vote.response_a, vote.response_b, vote.winner),
            )
            await db.commit()

    def _save_jsonl(self, vote: PreferenceVote) -> None:
        entry = {
            "timestamp": vote.timestamp,
            "prompt": vote.prompt,
            "response_a": vote.response_a,
            "response_b": vote.response_b,
            "winner": vote.winner,
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


