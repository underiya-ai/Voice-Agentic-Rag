import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from livekit import api

load_dotenv()

app = FastAPI(title="LiveKit Token Server")

# React app (Vite) ka default URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
                   "http://127.0.0.1:5173",],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/token")
def create_token():
    """Har browser session ke liye naya room aur token banata hai."""
    room_name = f"room-{uuid.uuid4().hex[:8]}"
    identity = f"user-{uuid.uuid4().hex[:6]}"

    token = (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(identity)
        .with_name(identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )

    return {"token": token, "url": os.environ["LIVEKIT_URL"], "room": room_name}