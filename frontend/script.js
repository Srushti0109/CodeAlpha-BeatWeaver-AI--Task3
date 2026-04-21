// --- 1. Login & Register Logic ---
document.querySelector('.btn-glow').addEventListener('click', async () => {
    const username = document.getElementById('username').value;
    const email = document.getElementById('useremail').value;
    
    if (username && email) {
        try {
            const response = await fetch('http://127.0.0.1:8000/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, email })
            });
            const result = await response.json();
            
            if (result.message === "Success") {
                document.getElementById('auth-section').style.display = 'none';
                document.querySelector('.divider').style.display = 'none';
                document.getElementById('studio-section').style.display = 'block';
                document.getElementById('status').innerText = `Welcome, ${username}!`;
            }
        } catch (e) { alert("Server error! Is main.py running?"); }
    } else { alert("Fill in both fields!"); }
});

// --- 2. Generation & Gallery Logic ---
document.getElementById('generateBtn').addEventListener('click', async () => {
    const status = document.getElementById('status');
    const visualizer = document.querySelector('.visualizer-container');
    const songList = document.getElementById('song-list');
    
    status.innerHTML = "🪄 AI is composing...";
    visualizer.classList.add('animating');

    try {
        const response = await fetch('http://127.0.0.1:8000/generate', { method: 'POST' });
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            
            // Add to Spotify-style list
            const trackName = `Vibe_Track_${Math.floor(Math.random() * 9000 + 1000)}.mid`;
            const item = document.createElement('div');
            item.style = "display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); font-size: 0.85rem;";
            item.innerHTML = `<span>${trackName}</span> <a href="${url}" download="${trackName}" style="color: #fc00ff; text-decoration: none; font-weight: bold;">Download</a>`;
            songList.prepend(item);

            status.innerHTML = "✅ Vibe added to your library!";
        }
    } catch (e) { status.innerText = "❌ Generation Error."; }
    finally { visualizer.classList.remove('animating'); }
});