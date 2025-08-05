from fastapi import APIRouter, Body
from journal.live_entry_handler import save_journal_entry

router = APIRouter()

@router.post("/journal/entry")
def add_journal_entry(
    persona_name: str = Body(...),
    entry_text: str = Body(...),
    mood_tags: list = Body(...),
    symbols: list = Body(...)
):
    result = save_journal_entry(persona_name, entry_text, mood_tags, symbols)
    return {"status": "success", "entry": result}
