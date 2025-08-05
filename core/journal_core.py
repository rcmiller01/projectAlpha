from modules.memory.journal.journal_core import JournalCore

_journals = {}


def _get_journal(persona: str) -> JournalCore:
    if persona not in _journals:
        _journals[persona] = JournalCore(persona)
    return _journals[persona]


def create_private_bloom(persona: str, text: str):
    """Create a private bloom entry for the given persona and return the entry."""
    journal = _get_journal(persona)
    journal.auto_bloom("bloom", text)
    last_entry = journal.entries[-1] if journal.entries else {}
    return {"timestamp": last_entry.get("timestamp"), "entry": text} 