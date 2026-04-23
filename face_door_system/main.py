import json
import os
import time
from typing import Optional, Tuple

from access_log import AccessLogger
from logger_setup import setup_logger
from recognition_engine import RecognitionEngine
from serial_comm import SerialErrorCode, SerialManager
from state_machine import StateMachine, SystemState


def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


class FaceDoorSystem:
    def __init__(self, config: dict):
        self.config = config

        recog = config["recognition"]
        serial_cfg = config["serial"]
        camera_cfg = config["camera"]
        log_cfg = config["logging"]
        paths_cfg = config["paths"]
        system_cfg = config["system"]

        self.logger = setup_logger(
            log_dir=log_cfg["log_dir"],
            log_file=log_cfg["system_log_file"],
            level=log_cfg["level"],
            max_bytes=log_cfg.get("max_bytes", 2 * 1024 * 1024),
            backup_count=log_cfg.get("backup_count", 5)
        )

        self.access_logger = AccessLogger(
            log_dir=log_cfg["log_dir"],
            log_file=log_cfg["access_log_file"],
            max_bytes=log_cfg.get("max_bytes", 2 * 1024 * 1024),
            backup_count=log_cfg.get("backup_count", 5)
        )

        self.owner_name = system_cfg["owner_name"]
        self.window_name = system_cfg["window_name"]

        self.known_faces_dir = os.path.expanduser(paths_cfg["known_faces_dir"])
        self.trt_cache_dir = os.path.expanduser(paths_cfg["trt_cache_dir"])
        self.insightface_model_root = os.path.expanduser(
            paths_cfg.get("insightface_model_root", "../models/insightface")
        )

        self.camera_sensor_id = camera_cfg["sensor_id"]
        self.camera_width = camera_cfg["width"]
        self.camera_height = camera_cfg["height"]
        self.camera_fps = camera_cfg["fps"]
        self.camera_flip_method = camera_cfg["flip_method"]

        self.serial_manager = SerialManager(
            enabled=serial_cfg["enabled"],
            port=serial_cfg["port"],
            baudrate=serial_cfg["baudrate"],
            timeout=serial_cfg["timeout"],
            logger=self.logger
        )
        self.open_command = serial_cfg["open_command"]

        self.state_machine = StateMachine(logger=self.logger)

        self.threshold = recog["threshold"]
        self.cooldown_seconds = recog["cooldown_seconds"]
        self.open_score = recog["open_score"]
        self.frame_skip = recog["frame_skip"]
        self.det_size = recog["det_size"]
        self.print_unknown = recog["print_unknown"]
        self.show_fps_every = recog["show_fps_every"]
        self.recognition_loop_interval = recog.get("loop_interval", 0.2)
        self.recognition_engine = RecognitionEngine(
            known_faces_dir=self.known_faces_dir,
            camera_sensor_id=self.camera_sensor_id,
            camera_width=self.camera_width,
            camera_height=self.camera_height,
            camera_fps=self.camera_fps,
            camera_backend=camera_cfg.get("backend", "auto"),
            det_size=tuple(self.det_size),
            model_root=self.insightface_model_root,
            logger=self.logger
        )

    @property
    def score_count(self) -> int:
        return self.state_machine.score_count

    @property
    def cooldown_until(self) -> float:
        return self.state_machine.cooldown_until

    @property
    def current_person(self):
        return self.state_machine.current_person

    @property
    def last_similarity(self) -> float:
        return self.state_machine.last_similarity

    def refresh_state(self) -> None:
        self.state_machine.refresh_cooldown(time.time())

    def is_in_cooldown(self) -> bool:
        now = time.time()
        if self.state_machine.is_in_cooldown(now):
            return True

        self.state_machine.refresh_cooldown(now)
        return False

    def get_status_snapshot(self) -> dict:
        now = time.time()
        self.state_machine.refresh_cooldown(now)
        return self.state_machine.get_status_snapshot(now)

    def enter_cooldown(self, reason: str = "open_success") -> None:
        now = time.time()
        self.state_machine.enter_cooldown(
            now + self.cooldown_seconds,
            reason=reason,
            now=now
        )
        self.logger.info(f"Entering cooldown for {self.cooldown_seconds} seconds")

    def initialize(self) -> None:
        self.logger.info("System initializing...")
        self.logger.info(
            f"Camera config loaded: sensor_id={self.camera_sensor_id}, "
            f"size={self.camera_width}x{self.camera_height}, fps={self.camera_fps}"
        )
        self.logger.info(f"Known faces dir: {self.known_faces_dir}")
        self.serial_manager.connect()
        self.state_machine.set_state(SystemState.WAITING, reason="initialize")
        self.logger.info("System initialized successfully")

    def process_recognition_result(self, best_name: str, best_score: float) -> None:
        self.refresh_state()
        self.state_machine.record_recognition(best_name, best_score)

        self.logger.info(
            f"Recognition result: name={best_name}, similarity={best_score:.3f}"
        )

        self.access_logger.write_event(
            event_type="recognize",
            person_name=best_name,
            similarity=round(best_score, 3),
            result="candidate",
            state_to=self.state_machine.state,
            source="face_recognition"
        )

        if self.is_in_cooldown():
            if self.state_machine.state != SystemState.COOLDOWN:
                self.state_machine.set_state(
                    SystemState.COOLDOWN,
                    reason="cooldown_active",
                    similarity=best_score
                )

            self.logger.info("Request blocked by cooldown")
            self.access_logger.write_event(
                event_type="cooldown",
                person_name=best_name,
                similarity=round(best_score, 3),
                result="blocked",
                state_to=self.state_machine.state,
                failure_reason={
                    "code": "COOLDOWN_ACTIVE",
                    "message": "Recognition blocked while door is cooling down"
                },
                source="face_recognition"
            )
            return

        if best_score >= self.threshold:
            if self.state_machine.state == SystemState.WAITING:
                self.state_machine.start_matching(
                    best_name,
                    similarity=best_score,
                    reason="first_valid_match"
                )
                self.logger.info(
                    f"Enter MATCHING: person={best_name}, "
                    f"score_count={self.score_count}"
                )

                if self.score_count >= self.open_score:
                    self.state_machine.confirm(
                        reason="open_score_reached",
                        similarity=best_score
                    )
                    self.handle_confirmed(best_name, best_score)
                return

            if self.state_machine.state == SystemState.MATCHING:
                previous_person = self.current_person
                same_person = self.state_machine.accumulate_matching(best_name)
                if same_person:
                    self.logger.info(
                        f"Matching: person={best_name}, "
                        f"score_count={self.score_count}"
                    )
                else:
                    self.logger.info(
                        f"Person changed: {previous_person} -> {best_name}, "
                        f"reset matching"
                    )

                if self.score_count >= self.open_score:
                    self.state_machine.confirm(
                        reason="open_score_reached",
                        similarity=best_score
                    )
                    self.handle_confirmed(best_name, best_score)
                return

        self.logger.info("Similarity below threshold, reset to WAITING")
        self.access_logger.write_event(
            event_type="deny",
            person_name=best_name,
            similarity=round(best_score, 3),
            result="denied",
            state_from=self.state_machine.state,
            state_to=SystemState.WAITING,
            failure_reason={
                "code": "SIMILARITY_BELOW_THRESHOLD",
                "message": "Similarity is below configured threshold"
            },
            source="face_recognition"
        )

        self.state_machine.reset_matching(
            reason="similarity_below_threshold",
            similarity=best_score
        )

    def start_recognition_engine(self) -> bool:
        return self.recognition_engine.start()

    def stop_recognition_engine(self) -> None:
        self.recognition_engine.stop()

    def get_recognition_result(self) -> Optional[Tuple[str, float]]:
        return self.recognition_engine.recognize_once()

    def confirm_and_open(self, best_name: str, best_score: float) -> None:
        self.state_machine.confirm(
            reason="manual_confirm",
            similarity=best_score
        )
        self.handle_confirmed(best_name, best_score)

    def handle_confirmed(self, best_name: str, best_score: float) -> None:
        self.logger.info(
            f"Access confirmed: person={best_name}, similarity={best_score:.3f}"
        )

        self.access_logger.write_event(
            event_type="confirm",
            person_name=best_name,
            similarity=round(best_score, 3),
            result="confirmed",
            state_to=self.state_machine.state,
            source="face_recognition"
        )

        self.handle_open(best_name, best_score)

    def handle_open(self, best_name: str, best_score: float) -> None:
        self.logger.info("Sending OPEN command")
        send_result = self.serial_manager.send_command(self.open_command)
        send_ok = send_result.get("ok", False)
        failure_reason = None if send_ok else self.get_serial_failure_reason(
            send_result
        )

        self.access_logger.write_event(
            event_type="serial_send",
            person_name=best_name,
            similarity=round(best_score, 3),
            result="sent" if send_ok else "failed",
            door_result="command_sent" if send_ok else "command_failed",
            failure_reason=failure_reason,
            source="face_recognition",
            extra={
                "command": self.open_command.strip(),
                "failure_reason": (
                    failure_reason.get("code") if failure_reason else None
                ),
                "message": (
                    failure_reason.get("message") if failure_reason else None
                )
            }
        )

        if send_ok:
            self.state_machine.open(
                reason="serial_send_success",
                similarity=best_score
            )
            self.logger.info(f"Door opened for {best_name}")
            self.access_logger.write_event(
                event_type="open",
                person_name=best_name,
                similarity=round(best_score, 3),
                result="success",
                state_to=self.state_machine.state,
                door_result="opened",
                source="face_recognition"
            )

            self.state_machine.clear_match_context()
            self.enter_cooldown()
        else:
            self.logger.error("Door open failed because serial send failed")
            self.access_logger.write_event(
                event_type="open",
                person_name=best_name,
                similarity=round(best_score, 3),
                result="failed",
                state_from=self.state_machine.state,
                state_to=SystemState.WAITING,
                door_result="open_failed",
                failure_reason=failure_reason,
                source="face_recognition"
            )
            self.state_machine.reset_matching(
                reason="serial_send_failed",
                similarity=best_score
            )

    def manual_open(self, source: str = "api_manual") -> dict:
        if self.is_in_cooldown():
            self.logger.info("Manual open blocked by cooldown")
            self.access_logger.write_event(
                event_type="manual_open",
                person_name="admin",
                similarity=0.0,
                result="blocked",
                state_to=self.state_machine.state,
                door_result="blocked",
                failure_reason={
                    "code": "COOLDOWN_ACTIVE",
                    "message": "System in cooldown"
                },
                source=source,
                extra={
                    "source": source,
                    "reason": "System in cooldown"
                }
            )
            return {
                "success": False,
                "message": "System in cooldown",
                "state": self.state_machine.state
            }

        self.logger.info("Manual open requested from API")
        send_result = self.serial_manager.send_command(self.open_command)
        send_ok = send_result.get("ok", False)
        failure_reason = None if send_ok else self.get_serial_failure_reason(
            send_result,
            "Door open command could not be sent"
        )

        self.access_logger.write_event(
            event_type="manual_open",
            person_name="admin",
            similarity=0.0,
            result="success" if send_ok else "failed",
            state_to=self.state_machine.state,
            door_result="command_sent" if send_ok else "command_failed",
            failure_reason=failure_reason,
            source=source,
            extra={
                "source": source,
                "command": self.open_command.strip(),
                "failure_reason": (
                    failure_reason.get("code") if failure_reason else None
                ),
                "message": (
                    failure_reason.get("message") if failure_reason else None
                )
            }
        )

        if send_ok:
            self.state_machine.open(reason="manual_open_success", similarity=0.0)
            self.state_machine.clear_match_context()
            self.enter_cooldown(reason="manual_open_success")
            return {
                "success": True,
                "message": "Door open command sent",
                "state": self.state_machine.state
            }

        self.logger.error("Manual open failed because serial send failed")
        self.state_machine.reset_matching(
            reason="manual_serial_send_failed",
            similarity=0.0
        )
        return {
            "success": False,
            "message": failure_reason["code"],
            "failure_reason": failure_reason,
            "state": self.state_machine.state
        }

    def get_serial_failure_reason(
        self,
        serial_result: Optional[dict] = None,
        fallback_message: str = "Serial command failed"
    ) -> dict:
        if serial_result and serial_result.get("code"):
            return {
                "code": serial_result["code"],
                "message": serial_result.get("message", fallback_message),
            }

        last_error = getattr(self.serial_manager, "last_error", None)
        if last_error:
            return last_error

        return {
            "code": SerialErrorCode.WRITE_FAILED,
            "message": fallback_message,
        }

    def shutdown(self) -> None:
        self.logger.info("System shutting down...")
        self.serial_manager.close()
        self.logger.info("System stopped")


def main():
    config = load_config("config.json")

    recog = config["recognition"]
    serial_cfg = config["serial"]
    camera_cfg = config["camera"]
    log_cfg = config["logging"]
    paths_cfg = config["paths"]
    system_cfg = config["system"]

    known_faces_dir = os.path.expanduser(paths_cfg["known_faces_dir"])
    threshold = recog["threshold"]
    cooldown = recog["cooldown_seconds"]
    open_score = recog["open_score"]
    port = serial_cfg["port"]
    baudrate = serial_cfg["baudrate"]
    width = camera_cfg["width"]
    height = camera_cfg["height"]

    logger = setup_logger(
        log_dir=log_cfg["log_dir"],
        log_file=log_cfg["system_log_file"],
        level=log_cfg["level"],
        max_bytes=log_cfg.get("max_bytes", 2 * 1024 * 1024),
        backup_count=log_cfg.get("backup_count", 5)
    )

    access_logger = AccessLogger(
        log_dir=log_cfg["log_dir"],
        log_file=log_cfg["access_log_file"],
        max_bytes=log_cfg.get("max_bytes", 2 * 1024 * 1024),
        backup_count=log_cfg.get("backup_count", 5)
    )

    logger.info(
        f"Config loaded for {system_cfg['owner_name']}: "
        f"threshold={threshold}, cooldown={cooldown}, open_score={open_score}"
    )
    logger.info(
        f"Serial config loaded: port={port}, baudrate={baudrate}"
    )
    logger.info(
        f"Camera config loaded: width={width}, height={height}"
    )
    logger.info(f"Known faces dir resolved: {known_faces_dir}")

    system = FaceDoorSystem(config)

    try:
        system.initialize()

        fake_results = [
            ("Alice", 0.62),
            ("Alice", 0.68),
            ("Alice", 0.70),
            ("Alice", 0.72),
            ("Alice", 0.74),
            ("Unknown", 0.40),
        ]

        for name, score in fake_results:
            system.process_recognition_result(name, score)
            time.sleep(1)

    except KeyboardInterrupt:
        system.logger.info("KeyboardInterrupt received, exiting...")
    except Exception:
        logger.exception("Error occurred")
    finally:
        system.shutdown()


if __name__ == "__main__":
    main()
