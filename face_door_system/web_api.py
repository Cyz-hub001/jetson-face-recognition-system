from contextlib import asynccontextmanager
from pathlib import Path
import sys
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from main import FaceDoorSystem, load_config
from services import DoorSystemService


CONFIG_PATH = MODULE_DIR / "config.json"
service: Optional[DoorSystemService] = None


class StatusResponse(BaseModel):
    state: str
    current_person: Optional[str] = None
    score_count: int
    last_similarity: float
    recent_recognition: Optional[dict] = None
    in_cooldown: bool
    cooldown_remaining: float
    owner_name: str
    window_name: str
    service_time: str
    serial_enabled: bool
    serial_connected: bool
    camera_running: bool
    recognition_running: bool


class HealthResponse(BaseModel):
    status: str
    serial_available: bool
    serial_enabled: bool
    camera_running: bool
    recognition_running: bool
    state: str
    service_time: str


class LogResponse(BaseModel):
    items: List[Dict] = Field(default_factory=list)


class ManualOpenResponse(BaseModel):
    success: bool
    message: str
    state: str


class RecognitionControlResponse(BaseModel):
    success: bool
    message: str
    camera_running: bool


class ConfigResponse(BaseModel):
    system: dict
    recognition: dict
    serial: dict
    camera: dict
    web: dict


def load_runtime_config() -> dict:
    config = load_config(str(CONFIG_PATH))
    log_dir = Path(config["logging"]["log_dir"])
    if not log_dir.is_absolute():
        config["logging"]["log_dir"] = str(CONFIG_PATH.parent / log_dir)

    for path_key in ("known_faces_dir", "trt_cache_dir", "insightface_model_root"):
        raw_path = Path(config["paths"][path_key])
        if not raw_path.is_absolute() and not str(raw_path).startswith("~"):
            config["paths"][path_key] = str(CONFIG_PATH.parent / raw_path)
    return config


def get_service() -> DoorSystemService:
    if service is None:
        raise HTTPException(status_code=503, detail="System is not initialized")
    return service


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service

    config = load_runtime_config()
    system = FaceDoorSystem(config)
    system.initialize()
    service = DoorSystemService(system=system, config_path=CONFIG_PATH)
    if config.get("web", {}).get("auto_start_recognition", False):
        service.start_recognition_loop()

    try:
        yield
    finally:
        service.shutdown()
        service = None


app = FastAPI(title="Face Door System API", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(MODULE_DIR / "static" / "index.html")


@app.get("/api/health", response_model=HealthResponse)
def get_health():
    return get_service().get_health()


@app.get("/api/status", response_model=StatusResponse)
def get_status():
    return get_service().get_status()


@app.get("/api/logs/access", response_model=LogResponse)
def get_access_logs(limit: int = Query(default=50, ge=1, le=500)):
    return get_service().get_recent_access_logs(limit=limit)


@app.get("/api/logs/system", response_model=LogResponse)
def get_system_logs(limit: int = Query(default=100, ge=1, le=500)):
    return get_service().get_recent_system_logs(limit=limit)


@app.get("/api/config", response_model=ConfigResponse)
def get_config():
    return get_service().get_config()


@app.post("/api/door/open", response_model=ManualOpenResponse)
def manual_open(request: Request):
    result = get_service().manual_open(
        client_host=request.client.host if request.client else None
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/api/recognition/start", response_model=RecognitionControlResponse)
def start_recognition():
    return get_service().start_recognition_loop()


@app.post("/api/recognition/stop", response_model=RecognitionControlResponse)
def stop_recognition():
    return get_service().stop_recognition_loop()


@app.get("/api/recording/status")
def get_recording_status():
    svc = get_service()
    return {
        "recording": svc.system.video_recorder.is_recording,
        "enabled": svc.system.video_recorder.enabled,
    }


@app.get("/api/recording/list")
def list_recordings():
    svc = get_service()
    save_dir = svc.system.video_recorder.save_dir
    recordings_dir = Path(save_dir)
    if not recordings_dir.is_absolute() and svc.config_path:
        recordings_dir = svc.config_path.parent / recordings_dir
    if not recordings_dir.exists():
        return {"items": []}
    files = sorted(recordings_dir.glob("*.mp4"), reverse=True)
    return {
        "items": [
            {
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "created": datetime.fromtimestamp(f.stat().st_ctime).isoformat(),
            }
            for f in files[:100]
        ]
    }
