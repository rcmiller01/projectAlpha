from dolphin_backend.main import app

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv('DOLPHIN_PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
