from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import sqlite3
import os
import subprocess
import uvicorn
import time
import shutil

app = FastAPI()

# --- 1. Enable CORS (The Bridge) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Database Setup ---
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- 3. Registration Route ---
@app.post("/register")
async def register_user(data: dict):
    username = data.get("username")
    email = data.get("email")
    if not username or not email:
        raise HTTPException(status_code=400, detail="Missing fields")
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, email) VALUES (?, ?)", (username, email))
        conn.commit()
        conn.close()
        return {"message": "Success"}
    except sqlite3.IntegrityError:
        return {"message": "Success"} 

# --- 4. Generation Route ---
@app.post("/generate")
async def generate_music():
    try:
        print("🪄 AI is composing...")
        subprocess.run(["python", "generator/generator.py"], capture_output=True, text=True)
        
        original_midi = os.path.join("output", "generated_lofi.mid")
        temp_midi = os.path.join("output", "temp_vibe.mid")
        
        time.sleep(2) 
        
        if os.path.exists(original_midi):
            shutil.copy2(original_midi, temp_midi)
            print(f"✅ Vibe stabilized. Sending to browser...")
            return FileResponse(
                path=temp_midi, 
                media_type='audio/midi', 
                filename="beatweaver_vibe.mid"
            )
        else:
            return {"error": "File not found. Please try again."}
            
    except Exception as e:
        print(f"❌ Server Error: {str(e)}")
        return {"error": str(e)}

# --- 5. Secret Admin Route (What's New) ---
# Visit this URL in your browser to see your users!
@app.get("/view-users-list")
async def view_users():
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()
        conn.close()
        # Returns the data in a clean JSON format
        return {"total_users": len(users), "users": users}
    except Exception as e:
        return {"error": str(e)}