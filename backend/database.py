import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease TEXT,
            confidence TEXT,
            severity TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_log(disease, confidence, severity):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO logs (disease, confidence, severity, timestamp) VALUES (?, ?, ?, ?)",
        (disease, confidence, severity, datetime.now())
    )
    conn.commit()
    conn.close()
