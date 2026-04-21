import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Optional

from main import FaceDoorSystem


class DoorSystemService:
    def __init__(
        self,
        system: FaceDoorSystem,
        config_path: Optional[Path] = None
    ):
        self.system = system
        self.config_path = Path(config_path).resolve() if config_path else None
        self.lock = RLock()
        self._recognition_stop_event = Event()
        self._recognition_thread: Optional[Thread] = None
        self._recognition_starting = False

    def get_status(self) -> dict:
        with self.lock:
            status = self.system.get_status_snapshot()
            status.update(
                {
                    "owner_name": self.system.owner_name,
                    "window_name": self.system.window_name,
                    "service_time": self._service_time(),
                    "serial_enabled": self.system.serial_manager.enabled,
                    "serial_connected": self._is_serial_connected(),
                    "camera_running": self.is_recognition_running(),
                }
            )
            return status

    def get_health(self) -> dict:
        with self.lock:
            return {
                "status": "ok",
                "serial_available": self._is_serial_connected(),
                "serial_enabled": self.system.serial_manager.enabled,
                "camera_running": self.is_recognition_running(),
                "state": self.system.state_machine.state,
                "service_time": self._service_time(),
            }

    def get_recent_access_logs(self, limit: int = 50) -> dict:
        limit = max(1, min(limit, 500))
        log_file = self._access_log_file()

        if not log_file.exists():
            return {"items": []}

        items = []
        with self.lock:
            lines = log_file.read_text(encoding="utf-8").splitlines()

        for line in lines[-limit:]:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "timestamp" not in item and "time" in item:
                item["timestamp"] = item["time"]
            items.append(item)

        return {"items": items}

    def get_recent_system_logs(self, limit: int = 100) -> dict:
        limit = max(1, min(limit, 500))
        log_file = self._system_log_file()

        if not log_file.exists():
            return {"items": []}

        try:
            with self.lock:
                lines = log_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return {"items": []}

        return {
            "items": [
                self._parse_system_log_line(line)
                for line in lines[-limit:]
            ]
        }

    def manual_open(self, client_host: Optional[str] = None) -> dict:
        if not self._is_manual_open_allowed(client_host):
            return {
                "success": False,
                "message": "Manual open is not allowed from this client",
                "state": self.system.state_machine.state
            }

        with self.lock:
            return self.system.manual_open(source="api_manual")

    def start_recognition_loop(self) -> dict:
        with self.lock:
            if self.is_recognition_running() or self._recognition_starting:
                return {
                    "success": True,
                    "message": "Recognition loop already running",
                    "camera_running": self.is_recognition_running(),
                }
            self._recognition_starting = True

        engine_started = self.system.start_recognition_engine()
        if not engine_started:
            with self.lock:
                self._recognition_starting = False
                self._recognition_thread = None
            self.system.access_logger.write_event(
                event_type="recognition_loop",
                result="failed",
                extra={"error": "Recognition engine failed to start"}
            )
            return {
                "success": False,
                "message": "Recognition engine failed to start",
                "camera_running": False,
            }

        with self.lock:
            self._recognition_stop_event.clear()
            self._recognition_thread = Thread(
                target=self._run_recognition_loop,
                name="FaceRecognitionLoop",
                daemon=True
            )
            self._recognition_thread.start()
            self._recognition_starting = False
            self.system.logger.info("Recognition loop start requested")
            return {
                "success": True,
                "message": "Recognition loop started",
                "camera_running": True,
            }

    def stop_recognition_loop(self) -> dict:
        with self.lock:
            thread = self._recognition_thread
            if not self.is_recognition_running():
                self._recognition_thread = None
                self.system.stop_recognition_engine()
                return {
                    "success": True,
                    "message": "Recognition loop already stopped",
                    "camera_running": False,
                }

            self._recognition_stop_event.set()

        thread.join(timeout=5)

        with self.lock:
            running = self.is_recognition_running()
            if not running:
                self._recognition_thread = None
                self.system.stop_recognition_engine()
            self.system.logger.info("Recognition loop stop requested")
            return {
                "success": not running,
                "message": (
                    "Recognition loop stopped"
                    if not running
                    else "Recognition loop did not stop before timeout"
                ),
                "camera_running": running,
            }

    def is_recognition_running(self) -> bool:
        return bool(self._recognition_thread and self._recognition_thread.is_alive())

    def get_config(self) -> dict:
        config = deepcopy(self.system.config)
        return {
            "system": config.get("system", {}),
            "recognition": config.get("recognition", {}),
            "serial": self._safe_serial_config(config.get("serial", {})),
            "camera": config.get("camera", {}),
            "web": config.get("web", {}),
        }

    def shutdown(self) -> None:
        self.stop_recognition_loop()
        with self.lock:
            self.system.shutdown()

    def _access_log_file(self) -> Path:
        logging_config = self.system.config["logging"]
        log_dir = Path(logging_config["log_dir"])
        if not log_dir.is_absolute() and self.config_path:
            log_dir = self.config_path.parent / log_dir
        return log_dir / logging_config["access_log_file"]

    def _system_log_file(self) -> Path:
        logging_config = self.system.config["logging"]
        log_dir = Path(logging_config["log_dir"])
        if not log_dir.is_absolute() and self.config_path:
            log_dir = self.config_path.parent / log_dir
        return log_dir / logging_config["system_log_file"]

    def _is_serial_connected(self) -> bool:
        serial_obj = self.system.serial_manager.ser
        return bool(serial_obj and serial_obj.is_open)

    def _safe_serial_config(self, serial_config: dict) -> dict:
        return {
            "enabled": serial_config.get("enabled"),
            "port": serial_config.get("port"),
            "baudrate": serial_config.get("baudrate"),
            "timeout": serial_config.get("timeout"),
        }

    def _run_recognition_loop(self) -> None:
        self.system.logger.info("Recognition loop started")
        self.system.access_logger.write_event(
            event_type="recognition_loop",
            result="started"
        )

        try:
            while not self._recognition_stop_event.is_set():
                result = self.system.get_recognition_result()

                if result:
                    best_name, best_score = result
                    with self.lock:
                        self.system.process_recognition_result(best_name, best_score)

                self._recognition_stop_event.wait(
                    self.system.recognition_loop_interval
                )
        except Exception as e:
            self.system.logger.exception(f"Recognition loop crashed: {e}")
            self.system.access_logger.write_event(
                event_type="recognition_loop",
                result="failed",
                extra={"error": str(e)}
            )
        finally:
            self.system.stop_recognition_engine()
            self.system.logger.info("Recognition loop stopped")
            self.system.access_logger.write_event(
                event_type="recognition_loop",
                result="stopped"
            )

    def _is_manual_open_allowed(self, client_host: Optional[str]) -> bool:
        web_config = self.system.config.get("web", {})
        if not web_config.get("allow_manual_open", True):
            return False

        if not web_config.get("manual_open_local_only", True):
            return True

        return client_host in {"127.0.0.1", "::1", "localhost", None}

    def _service_time(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _parse_system_log_line(self, line: str) -> dict:
        parts = line.split(" | ", 3)
        if len(parts) != 4:
            return {"raw": line}

        timestamp, level, logger_name, message = parts
        return {
            "timestamp": timestamp,
            "level": level,
            "logger": logger_name,
            "message": message,
            "raw": line,
        }
