import unittest
import asyncio
from datetime import datetime, timedelta

from backend.modules.memory.symbolic_memory import SymbolicMemorySystem
from backend.modules.memory.symbol_resurrection import SymbolResurrectionEngine
from modules.core.guidance_coordinator import GuidanceCoordinator, GuidancePackage

class TestSymbolResurrection(unittest.TestCase):
    def setUp(self):
        self.memory = SymbolicMemorySystem(db_path=":memory:")
        asyncio.get_event_loop().run_until_complete(self.memory.initialize())
        self.engine = SymbolResurrectionEngine(self.memory, threshold_hours=1)

    def test_symbol_tagging_and_decay(self):
        asyncio.get_event_loop().run_until_complete(
            self.memory.store_memory("A warm flame", ["flame"], {"affection": 1.0})
        )
        # simulate inactivity
        self.memory.symbols_cache["flame"]["last_used"] = datetime.now() - timedelta(hours=2)
        symbol = self.engine.get_dormant_symbol()
        self.assertEqual(symbol, "flame")

    def test_resurrection_line_generation(self):
        asyncio.get_event_loop().run_until_complete(
            self.memory.store_memory("Hidden garden", ["garden"], {"peace": 0.5})
        )
        self.memory.symbols_cache["garden"]["last_used"] = datetime.now() - timedelta(hours=3)
        line = self.engine.propose_resurrection_line()
        self.assertIn("garden", line)

    def test_guidance_integration(self):
        coordinator = GuidanceCoordinator("tester")
        coordinator.symbol_res_engine = self.engine
        line = self.engine.propose_resurrection_line()
        context = {"current_emotional_state": {"calm": 0.5}}
        guidance = asyncio.get_event_loop().run_until_complete(
            coordinator.analyze_and_guide("hello", context)
        )
        if line:
            self.assertEqual(guidance.symbolic_resurrection_line, line)

if __name__ == '__main__':
    unittest.main()
