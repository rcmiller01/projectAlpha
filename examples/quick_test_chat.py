#!/usr/bin/env python3
"""
Quick Chat Test for Dolphin v2.1 Advanced Features
"""

import asyncio
import json

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

async def test_chat_with_advanced_features():
    print("Testing chat with advanced features...")
    
    if not AIOHTTP_AVAILABLE:
        print("SKIP: aiohttp not available")
        return True
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test chat with persona
            chat_data = {
                "message": "I need help organizing my daily tasks. What's your advice?",
                "session_id": "test_qa_session",
                "persona": "coach"
            }
            
            async with session.post("http://localhost:8000/api/chat", json=chat_data) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    response_text = data.get('response', '')
                    handler = data.get('handler', 'unknown')
                    persona_used = data.get('persona_used', 'unknown')
                    
                    print("SUCCESS: Chat with persona")
                    print(f"  Handler: {handler}")
                    print(f"  Persona: {persona_used}")
                    print(f"  Response length: {len(response_text)} chars")
                    print(f"  Preview: {response_text[:150]}...")
                    
                    return True
                else:
                    print(f"FAILED: Chat status {response.status}")
                    return False
                    
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_chat_with_advanced_features())
    exit(0 if result else 1)
