from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..services.demo_stream import next_demo_alert

router = APIRouter()

STREAM_INTERVAL_SEC = 1.0


@router.websocket("/ws/alerts")
async def alerts_ws(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            alert = next_demo_alert()
            await ws.send_json(alert.model_dump(mode="json"))
            await asyncio.sleep(STREAM_INTERVAL_SEC)
    except WebSocketDisconnect:
        return
