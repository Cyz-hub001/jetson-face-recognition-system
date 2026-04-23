import json
import os
import time

import main as app_main
from access_log import AccessLogger
from serial_comm import SerialErrorCode
from services import DoorSystemService
from state_machine import SystemState


class RecordingLogger:
    def __init__(self):
        self.messages = []

    def _record(self, level: str, message: str, *args) -> None:
        if args:
            message = message % args
        self.messages.append((level, message))

    def info(self, message: str, *args) -> None:
        self._record("INFO", message, *args)

    def warning(self, message: str, *args) -> None:
        self._record("WARNING", message, *args)

    def error(self, message: str, *args) -> None:
        self._record("ERROR", message, *args)

    def exception(self, message: str, *args) -> None:
        self._record("EXCEPTION", message, *args)

    def contains(self, text: str) -> bool:
        return any(text in message for _, message in self.messages)


class RecordingAccessLogger:
    def __init__(self):
        self.events = []

    def write_event(self, **kwargs) -> None:
        self.events.append(kwargs)

    def contains_event(self, event_type: str, result: str) -> bool:
        return any(
            event.get("event_type") == event_type and event.get("result") == result
            for event in self.events
        )

    def find_event(self, event_type: str, result: str) -> dict:
        for event in self.events:
            if event.get("event_type") == event_type and event.get("result") == result:
                return event
        return {}


def load_test_config(log_dir: str) -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["logging"]["log_dir"] = log_dir
    config["serial"]["enabled"] = False
    config["recognition"]["threshold"] = 0.5
    config["recognition"]["cooldown_seconds"] = 30.0
    return config


def make_system(log_dir: str, open_score: int, send_ok: bool) -> app_main.FaceDoorSystem:
    config = load_test_config(log_dir)
    config["recognition"]["open_score"] = open_score

    logger = RecordingLogger()
    original_setup_logger = app_main.setup_logger
    app_main.setup_logger = lambda **kwargs: logger
    try:
        system = app_main.FaceDoorSystem(config)
    finally:
        app_main.setup_logger = original_setup_logger

    system.logger = logger
    system.state_machine.logger = logger
    system.serial_manager.logger = logger
    system.access_logger = RecordingAccessLogger()
    if send_ok:
        send_result = {"ok": True}
    else:
        send_result = {
            "ok": False,
            "code": SerialErrorCode.NOT_CONNECTED,
            "message": "Serial port is not open"
        }
    system.serial_manager.send_command = lambda command: send_result
    system.start_recognition_engine = lambda: True
    system.stop_recognition_engine = lambda: None
    system.get_recognition_result = lambda: None
    return system


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_open_success_enters_cooldown(log_dir: str) -> None:
    system = make_system(log_dir, open_score=3, send_ok=True)

    system.process_recognition_result("Alice", 0.70)
    system.process_recognition_result("Alice", 0.71)
    system.process_recognition_result("Alice", 0.72)

    assert_true(system.state_machine.state == SystemState.COOLDOWN, "should enter COOLDOWN")
    assert_true(system.score_count == 0, "score_count should reset after opening")
    assert_true(system.current_person is None, "current_person should reset after opening")
    assert_true(system.logger.contains("STATE MATCHING -> CONFIRMED"), "missing CONFIRMED transition")
    assert_true(system.logger.contains("STATE CONFIRMED -> OPEN"), "missing OPEN transition")
    assert_true(system.logger.contains("STATE OPEN -> COOLDOWN"), "missing COOLDOWN transition")


def test_match_interrupted_resets_waiting(log_dir: str) -> None:
    system = make_system(log_dir, open_score=3, send_ok=True)

    system.process_recognition_result("Alice", 0.70)
    system.process_recognition_result("Unknown", 0.20)

    assert_true(system.state_machine.state == SystemState.WAITING, "interrupted match should go WAITING")
    assert_true(system.score_count == 0, "score_count should reset after interruption")
    assert_true(system.current_person is None, "current_person should reset after interruption")
    assert_true(
        system.logger.contains("reason=similarity_below_threshold"),
        "missing interruption reason"
    )


def test_person_switch_resets_count(log_dir: str) -> None:
    system = make_system(log_dir, open_score=3, send_ok=True)

    system.process_recognition_result("Alice", 0.70)
    system.process_recognition_result("Bob", 0.73)

    assert_true(system.state_machine.state == SystemState.MATCHING, "person switch should stay MATCHING")
    assert_true(system.current_person == "Bob", "current_person should switch to Bob")
    assert_true(system.score_count == 1, "score_count should restart after person switch")


def test_serial_failure_returns_waiting(log_dir: str) -> None:
    system = make_system(log_dir, open_score=1, send_ok=False)

    system.process_recognition_result("Alice", 0.70)

    assert_true(system.state_machine.state == SystemState.WAITING, "serial failure should return WAITING")
    assert_true(system.score_count == 0, "score_count should reset after serial failure")
    assert_true(system.current_person is None, "current_person should reset after serial failure")
    assert_true(system.logger.contains("reason=serial_send_failed"), "missing serial failure reason")
    event = system.access_logger.find_event("serial_send", "failed")
    assert_true(
        event["failure_reason"]["code"] == SerialErrorCode.NOT_CONNECTED,
        "serial failure should keep specific error code"
    )
    assert_true(
        event["extra"]["failure_reason"] == SerialErrorCode.NOT_CONNECTED,
        "serial failure extra should include code for log statistics"
    )


def test_cooldown_blocks_recognition(log_dir: str) -> None:
    system = make_system(log_dir, open_score=1, send_ok=True)

    system.process_recognition_result("Alice", 0.70)
    system.process_recognition_result("Alice", 0.75)

    assert_true(system.state_machine.state == SystemState.COOLDOWN, "cooldown should remain active")
    assert_true(system.logger.contains("Request blocked by cooldown"), "missing cooldown block log")
    assert_true(
        system.access_logger.contains_event("cooldown", "blocked"),
        "missing cooldown block event"
    )


def test_cooldown_expiry_returns_waiting(log_dir: str) -> None:
    system = make_system(log_dir, open_score=3, send_ok=True)
    system.state_machine.enter_cooldown(
        cooldown_until=time.time() - 1,
        reason="test_expired_cooldown",
        now=time.time() - 2
    )

    system.process_recognition_result("Alice", 0.70)

    assert_true(system.state_machine.state == SystemState.MATCHING, "expired cooldown should allow matching")
    assert_true(system.logger.contains("reason=cooldown_expired"), "missing cooldown expiry reason")


def test_status_refresh_expires_cooldown(log_dir: str) -> None:
    system = make_system(log_dir, open_score=3, send_ok=True)
    system.state_machine.enter_cooldown(
        cooldown_until=time.time() - 1,
        reason="test_expired_cooldown",
        now=time.time() - 2
    )

    snapshot = system.get_status_snapshot()

    assert_true(snapshot["state"] == SystemState.WAITING, "status should refresh expired cooldown")
    assert_true(snapshot["in_cooldown"] is False, "expired cooldown should report inactive")
    assert_true(system.logger.contains("reason=cooldown_expired"), "missing status refresh transition")


def test_status_snapshot(log_dir: str) -> None:
    system = make_system(log_dir, open_score=3, send_ok=True)

    system.process_recognition_result("Alice", 0.70)
    snapshot = system.get_status_snapshot()

    assert_true(snapshot["state"] == SystemState.MATCHING, "snapshot state mismatch")
    assert_true(snapshot["current_person"] == "Alice", "snapshot current_person mismatch")
    assert_true(snapshot["score_count"] == 1, "snapshot score_count mismatch")
    assert_true(snapshot["last_similarity"] == 0.70, "snapshot similarity mismatch")
    assert_true(
        snapshot["recent_recognition"]["person_name"] == "Alice",
        "snapshot recent recognition name mismatch"
    )
    assert_true(snapshot["in_cooldown"] is False, "snapshot cooldown mismatch")


def test_manual_open_success_enters_cooldown(log_dir: str) -> None:
    system = make_system(log_dir, open_score=3, send_ok=True)

    result = system.manual_open()

    assert_true(result["success"] is True, "manual open should succeed")
    assert_true(result["state"] == SystemState.COOLDOWN, "manual open should enter cooldown")
    event = system.access_logger.find_event("manual_open", "success")
    assert_true(bool(event), "missing manual open event")
    assert_true(event["extra"]["source"] == "api_manual", "manual open event should include source")
    assert_true(system.logger.contains("reason=manual_open_success"), "missing manual open cooldown transition")


def test_manual_open_blocked_by_cooldown(log_dir: str) -> None:
    system = make_system(log_dir, open_score=1, send_ok=True)

    system.process_recognition_result("Alice", 0.70)
    result = system.manual_open()

    assert_true(result["success"] is False, "manual open should be blocked")
    assert_true(result["state"] == SystemState.COOLDOWN, "blocked manual open should stay cooldown")
    event = system.access_logger.find_event("manual_open", "blocked")
    assert_true(bool(event), "missing blocked manual event")
    assert_true(event["extra"]["source"] == "api_manual", "blocked manual event should include source")


def test_manual_open_serial_failure_returns_code(log_dir: str) -> None:
    system = make_system(log_dir, open_score=3, send_ok=False)

    result = system.manual_open()

    assert_true(result["success"] is False, "manual open should fail")
    assert_true(
        result["message"] == SerialErrorCode.NOT_CONNECTED,
        "manual open should return serial error code"
    )
    assert_true(
        result["failure_reason"]["code"] == SerialErrorCode.NOT_CONNECTED,
        "manual open should include structured failure reason"
    )
    event = system.access_logger.find_event("manual_open", "failed")
    assert_true(
        event["extra"]["failure_reason"] == SerialErrorCode.NOT_CONNECTED,
        "manual open event should include serial code in extra"
    )


def test_service_status_and_config(log_dir: str) -> None:
    system = make_system(log_dir, open_score=3, send_ok=True)
    service = DoorSystemService(system)

    status = service.get_status()
    health = service.get_health()
    config = service.get_config()

    assert_true("serial_enabled" in status, "service status should include serial_enabled")
    assert_true("camera_running" in status, "service status should include camera_running")
    assert_true("recognition_running" in status, "service status should include recognition_running")
    assert_true(status["owner_name"] == "pyy02", "service status should include owner_name")
    assert_true(status["window_name"] == "Face Door System", "service status should include window_name")
    assert_true("service_time" in status, "service status should include service_time")
    assert_true(health["status"] == "ok", "health status should be ok")
    assert_true("serial_available" in health, "health should include serial_available")
    assert_true("system" in config, "safe config should include system")
    assert_true("recognition" in config, "safe config should include recognition")
    assert_true("serial" in config, "safe config should include serial")
    assert_true("open_command" not in config["serial"], "safe config should hide open_command")
    assert_true("web" in config, "safe config should include web")


def test_service_recognition_loop_start_stop(log_dir: str) -> None:
    system = make_system(log_dir, open_score=3, send_ok=True)
    service = DoorSystemService(system)

    start_result = service.start_recognition_loop()
    time.sleep(0.1)
    health = service.get_health()
    stop_result = service.stop_recognition_loop()

    assert_true(start_result["success"] is True, "recognition loop should start")
    assert_true(health["camera_running"] is True, "health should show recognition running")
    assert_true(stop_result["success"] is True, "recognition loop should stop")
    assert_true(stop_result["camera_running"] is False, "stop should report not running")
    assert_true(
        system.access_logger.contains_event("recognition_loop", "started"),
        "missing recognition loop start event"
    )
    assert_true(
        system.access_logger.contains_event("recognition_loop", "stopped"),
        "missing recognition loop stop event"
    )


def test_service_recognition_loop_start_failure(log_dir: str) -> None:
    system = make_system(log_dir, open_score=3, send_ok=True)
    system.start_recognition_engine = lambda: False
    service = DoorSystemService(system)

    result = service.start_recognition_loop()

    assert_true(result["success"] is False, "recognition loop start should fail")
    assert_true(result["camera_running"] is False, "failed start should not report running")
    assert_true(
        system.access_logger.contains_event("recognition_loop", "failed"),
        "missing recognition loop failed event"
    )


def test_manual_open_local_only_guard(log_dir: str) -> None:
    system = make_system(log_dir, open_score=3, send_ok=True)
    service = DoorSystemService(system)

    result = service.manual_open(client_host="192.168.1.10")

    assert_true(result["success"] is False, "remote manual open should be blocked")
    assert_true(
        result["message"] == "Manual open is not allowed from this client",
        "remote manual open should explain block reason"
    )


def test_access_log_schema_and_rotation(log_dir: str) -> None:
    log_file = "access_rotation_test.log"
    base_path = os.path.join(log_dir, log_file)
    test_paths = [base_path + suffix for suffix in ("", ".1", ".2")]

    def cleanup() -> None:
        for path in test_paths:
            if os.path.exists(path):
                os.remove(path)

    cleanup()
    try:
        access_logger = AccessLogger(
            log_dir=log_dir,
            log_file=log_file,
            max_bytes=240,
            backup_count=2,
        )

        access_logger.write_event(
            event_type="open",
            person_name="Alice",
            similarity=0.91,
            result="success",
            state_from=SystemState.CONFIRMED,
            state_to=SystemState.OPEN,
            door_result="opened",
            source="test",
        )
        for index in range(6):
            access_logger.write_event(
                event_type="deny",
                person_name="Unknown",
                similarity=0.1,
                result="denied",
                failure_reason={
                    "code": "TEST_DENY",
                    "message": f"deny event {index}"
                },
                source="test",
            )

        assert_true(os.path.exists(base_path), "access log should exist")
        assert_true(os.path.exists(base_path + ".1"), "access log should rotate")

        with open(base_path, "r", encoding="utf-8") as f:
            last_record = json.loads(f.read().splitlines()[-1])

        assert_true(last_record["timestamp"], "access event should include timestamp")
        assert_true(last_record["log_type"] == "access", "access event should include log_type")
        assert_true("failure_reason" in last_record, "access event should include failure reason")
    finally:
        cleanup()


def main() -> None:
    tests = [
        test_open_success_enters_cooldown,
        test_match_interrupted_resets_waiting,
        test_person_switch_resets_count,
        test_serial_failure_returns_waiting,
        test_cooldown_blocks_recognition,
        test_cooldown_expiry_returns_waiting,
        test_status_refresh_expires_cooldown,
        test_status_snapshot,
        test_manual_open_success_enters_cooldown,
        test_manual_open_blocked_by_cooldown,
        test_manual_open_serial_failure_returns_code,
        test_service_status_and_config,
        test_service_recognition_loop_start_stop,
        test_service_recognition_loop_start_failure,
        test_manual_open_local_only_guard,
        test_access_log_schema_and_rotation,
    ]

    base_log_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "logs")
    )
    os.makedirs(base_log_dir, exist_ok=True)

    for test in tests:
        test(base_log_dir)
        print(f"PASS {test.__name__}")

    print("All state machine regression tests passed.")


if __name__ == "__main__":
    main()
