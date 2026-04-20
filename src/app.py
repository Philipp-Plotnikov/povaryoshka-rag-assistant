import uvicorn
from api.server import POVARYOSHKA_SERVER


if __name__ == "__main__":
    uvicorn.run(
        POVARYOSHKA_SERVER, 
        host="0.0.0.0", 
        port=8080,
        ws_max_size=16 * 1024 * 1024  # 16 MB для WebSocket
    )
