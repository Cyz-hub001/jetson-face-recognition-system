from typing import Optional

try:
    import serial
except ImportError:
    serial = None


class SerialManager:
    def __init__(
        self,
        enabled: bool,
        port: str,
        baudrate: int,
        timeout: float,
        logger=None
    ):
        self.enabled = enabled
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.logger = logger
        self.ser: Optional["serial.Serial"] = None

    def connect(self) -> bool:
        if not self.enabled:
            if self.logger:
                self.logger.warning("Serial is disabled in config.")
            return False

        if serial is None:
            if self.logger:
                self.logger.error("pyserial is not installed.")
            return False

        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            if self.logger:
                self.logger.info(
                    f"Serial connected: port={self.port}, baudrate={self.baudrate}"
                )
            return True
        except Exception as e:
            if self.logger:
                self.logger.exception(f"Failed to open serial port: {e}")
            return False

    def send_command(self, command: str) -> bool:
        """
        发送字符串命令，比如 OPEN\\n
        """
        if not self.enabled:
            if self.logger:
                self.logger.warning(f"Serial disabled, skip sending: {command.strip()}")
            return False

        if self.ser is None or not self.ser.is_open:
            if self.logger:
                self.logger.error("Serial port is not open.")
            return False

        try:
            self.ser.write(command.encode("utf-8"))
            if self.logger:
                self.logger.info(f"Serial command sent: {command.strip()}")
            return True
        except Exception as e:
            if self.logger:
                self.logger.exception(f"Failed to send serial command: {e}")
            return False

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()
            if self.logger:
                self.logger.info("Serial port closed.")
                
