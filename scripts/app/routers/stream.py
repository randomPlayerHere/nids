import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import settings
from ..services.demo_stream import next_demo_alert

router = APIRouter()

STREAM_INTERVAL = 1.0 / settings.STREAM_RATE_HZ if settings.STREAM_RATE_HZ > 0 else 1.0


@router.websocket("/ws/alerts")
async def alerts_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            alert = next_demo_alert()
            data = alert.model_dump(mode="json")
            await ws.send_json(data)
            await asyncio.sleep(STREAM_INTERVAL)
    except WebSocketDisconnect:
        return
