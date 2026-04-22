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
# This allows your frontend to talk to the Railway backend
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
        # If user already exists, we still count it as a success for the UI flow
        return {"message": "Success"} 

# --- 4. Generation Route ---
@app.post("/generate")
async def generate_music():
    try:
        print("🪄 AI is composing...")
        # 1. Run the AI generator
        subprocess.run(["python", "generator/generator.py"], capture_output=True, text=True)
        
        original_midi = os.path.join("output", "generated_lofi.mid")
        temp_midi = os.path.join("output", "temp_vibe.mid")
        
        # 2. Safety delay for file writing
        time.sleep(2) 
        
        if os.path.exists(original_midi):
            # 3. Create a static copy for the download
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

# Note: We removed the uvicorn.run block here because Railway 
# uses the 'Procfile' to start the server properly in the cloud.