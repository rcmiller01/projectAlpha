#!/usr/bin/env python3
"""Check existing database schema"""

import sqlite3
from pathlib import Path

def check_database(db_path):
    """Check database schema"""
    if not Path(db_path).exists():
        print(f"Database {db_path} does not exist")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print(f"Tables in {db_path}:")
    for table in tables:
        table_name = table[0]
        print(f"\n  {table_name}:")
        
        # Get table info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        for col in columns:
            print(f"    {col[1]} {col[2]} {'NOT NULL' if col[3] else ''}")
    
    conn.close()

if __name__ == "__main__":
    check_database("emotion_training.db")
    
    # Also check the autopilot database if it exists
    autopilot_db = "emotion_quant_autopilot/emotion_training.db"
    if Path(autopilot_db).exists():
        print("\n" + "="*50)
        check_database(autopilot_db)
