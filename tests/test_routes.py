import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import types; sys.modules.setdefault("aiosqlite", types.ModuleType("aiosqlite"))

import json
from fastapi.testclient import TestClient
from dolphin_backend import app

client = TestClient(app)

class DummyResponse:
    def __init__(self, stdout='', returncode=0):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = ''

def test_quantization_status(monkeypatch):
    def fake_run(*args, **kwargs):
        data = {"autopilot": {"running": False}, "bootloader": {"running": True}}
        return DummyResponse(stdout=json.dumps(data))
    monkeypatch.setattr('subprocess.run', fake_run)
    resp = client.get('/api/quantization/status')
    assert resp.status_code == 200
    assert 'bootloader_running' in resp.json()

def test_persona_create_validation():
    resp = client.post('/api/personas/create', json={"id": "../bad"})
    assert resp.status_code == 400

def test_chat_route(monkeypatch):
    async def fake_process(self, req, persona_token=None):
        return {"response": "ok", "handler": "DOLPHIN", "reasoning": "test", "metadata": {}, "timestamp": "t", "session_id": "1", "persona_used": "p"}
    monkeypatch.setattr('dolphin_backend.orchestrator.DolphinOrchestrator.process_chat_request', fake_process)
    resp = client.post('/api/chat', json={"message": "hi"})
    assert resp.status_code == 200
    assert resp.json()['response'] == 'ok'
