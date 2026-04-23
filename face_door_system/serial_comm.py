from typing import Optional

try:
    import serial
except ImportError:
    serial = None


class SerialErrorCode:
    NOT_ENABLED = "SERIAL_NOT_ENABLED"
    NOT_AVAILABLE = "SERIAL_NOT_AVAILABLE"
    PORT_NOT_FOUND = "SERIAL_PORT_NOT_FOUND"
    NOT_CONNECTED = "SERIAL_NOT_CONNECTED"
    WRITE_FAILED = "SERIAL_WRITE_FAILED"


class SerialManager:
    def __init__(
        self,
        enabled: bool,
        port: str,
        baudrate: int,
        timeout: float,
        logger=None,
    ):
        self.enabled = enabled
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.logger = logger
        self.ser: Optional["serial.Serial"] = None
        self.last_error: Optional[dict] = None

    def connect(self) -> bool:
        if not self.enabled:
            self._set_error(
                SerialErrorCode.NOT_ENABLED,
                "Serial is disabled in config"
            )
            if self.logger:
                self.logger.warning("Serial is disabled in config.")
            return False

        if serial is None:
            self._set_error(
                SerialErrorCode.NOT_AVAILABLE,
                "pyserial is not installed"
            )
            if self.logger:
                self.logger.error("pyserial is not installed.")
            return False

        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
            )
            if self.logger:
                self.logger.info(
                    f"Serial connected: port={self.port}, baudrate={self.baudrate}"
                )
            self._clear_error()
            return True
        except Exception as e:
            self._set_error(SerialErrorCode.PORT_NOT_FOUND, str(e))
            if self.logger:
                self.logger.exception(f"Failed to open serial port: {e}")
            return False

    def send_command(self, command: str) -> dict:
        if not self.enabled:
            result = self._set_error(
                SerialErrorCode.NOT_ENABLED,
                "Serial is disabled in config"
            )
            if self.logger:
                self.logger.warning(f"Serial disabled, skip sending: {command.strip()}")
            return result

        if serial is None:
            result = self._set_error(
                SerialErrorCode.NOT_AVAILABLE,
                "pyserial is not installed"
            )
            if self.logger:
                self.logger.error("pyserial is not installed.")
            return result

        if self.ser is None or not self.ser.is_open:
            result = self._set_error(
                SerialErrorCode.NOT_CONNECTED,
                "Serial port is not open"
            )
            if self.logger:
                self.logger.error("Serial port is not open.")
            return result

        try:
            self.ser.write(command.encode("utf-8"))
            if self.logger:
                self.logger.info(f"Serial command sent: {command.strip()}")
            self._clear_error()
            return {"ok": True}
        except Exception as e:
            result = self._set_error(SerialErrorCode.WRITE_FAILED, str(e))
            if self.logger:
                self.logger.exception(f"Failed to send serial command: {e}")
            return result

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()
            if self.logger:
                self.logger.info("Serial port closed.")

    def _set_error(self, code: str, message: str) -> dict:
        self.last_error = {
            "code": code,
            "message": message,
        }
        return {
            "ok": False,
            "code": code,
            "message": message,
        }

    def _clear_error(self) -> None:
        self.last_error = None
