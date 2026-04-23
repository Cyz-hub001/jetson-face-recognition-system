import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

if cv2 is not None:
    if not hasattr(cv2, "INTER_NEAREST_EXACT"):
        cv2.INTER_NEAREST_EXACT = cv2.INTER_NEAREST
    if not hasattr(cv2, "INTER_LINEAR_EXACT"):
        cv2.INTER_LINEAR_EXACT = cv2.INTER_LINEAR

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class RecognitionEngine:
    def __init__(
        self,
        known_faces_dir: str,
        camera_sensor_id: int,
        camera_width: int,
        camera_height: int,
        camera_fps: int,
        camera_backend: str,
        camera_flip_method: int,
        camera_capture_width: int,
        camera_capture_height: int,
        det_size: tuple,
        model_root: str,
        logger=None
    ):
        self.known_faces_dir = Path(os.path.expanduser(known_faces_dir))
        self.camera_sensor_id = camera_sensor_id
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.camera_backend = camera_backend
        self.camera_flip_method = camera_flip_method
        self.camera_capture_width = camera_capture_width
        self.camera_capture_height = camera_capture_height
        self.det_size = tuple(det_size)
        self.model_root = Path(os.path.expanduser(model_root))
        self.logger = logger

        self.app = None
        self.capture = None
        self.known_embeddings = []
        self.running = False

    def start(self) -> bool:
        if self.running:
            return True

        if cv2 is None:
            self._error("OpenCV is not installed.")
            return False

        if FaceAnalysis is None:
            self._error("insightface is not installed.")
            return False

        try:
            self._prepare_model()
            self._load_known_faces()
        except Exception as e:
            self._error(f"Recognition engine startup failed: {e}")
            self.stop()
            return False

        if not self.known_embeddings:
            self._warning(f"No known faces loaded from: {self.known_faces_dir}")

        self.capture = self._open_camera()
        if self.capture is None:
            self._error(f"Failed to open camera: sensor_id={self.camera_sensor_id}")
            self.stop()
            return False

        self.running = True
        self._info("Recognition engine started")
        return True

    def stop(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.running = False
        self._info("Recognition engine stopped")

    def recognize_once(self) -> Optional[Tuple[str, float]]:
        if not self.running or self.capture is None:
            return None

        ok, frame = self.capture.read()
        if not ok or frame is None:
            self._warning("Camera frame read failed")
            return None

        faces = self.app.get(frame)
        if not faces:
            return None

        face = self._largest_face(faces)
        embedding = self._normalize(face.embedding)

        if not self.known_embeddings:
            return ("Unknown", 0.0)

        best_name = "Unknown"
        best_score = -1.0
        for person_name, known_embedding in self.known_embeddings:
            score = float(np.dot(embedding, known_embedding))
            if score > best_score:
                best_name = person_name
                best_score = score

        return best_name, best_score

    def _prepare_model(self) -> None:
        if self.app is not None:
            return

        self.model_root.mkdir(parents=True, exist_ok=True)
        self.app = FaceAnalysis(name="buffalo_l", root=str(self.model_root))
        self.app.prepare(ctx_id=0, det_size=self.det_size)
        self._info(
            f"InsightFace model prepared: det_size={self.det_size}, "
            f"model_root={self.model_root}"
        )

    def _open_camera(self):
        if str(self.camera_backend or "").lower() == "csi":
            return self._open_csi_camera()

        for backend_name, backend_value in self._camera_backend_candidates():
            capture = cv2.VideoCapture(self.camera_sensor_id, backend_value)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            capture.set(cv2.CAP_PROP_FPS, self.camera_fps)

            if capture.isOpened():
                self._info(
                    f"Camera opened: sensor_id={self.camera_sensor_id}, "
                    f"backend={backend_name}"
                )
                return capture

            capture.release()
            self._warning(
                f"Camera backend failed: sensor_id={self.camera_sensor_id}, "
                f"backend={backend_name}"
            )

        return None

    def _open_csi_camera(self):
        pipeline = self._csi_gstreamer_pipeline()
        capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if capture.isOpened():
            self._info(
                f"Camera opened: sensor_id={self.camera_sensor_id}, "
                "backend=csi"
            )
            return capture

        capture.release()
        self._warning(
            f"Camera backend failed: sensor_id={self.camera_sensor_id}, "
            "backend=csi"
        )
        self._warning(f"CSI pipeline: {pipeline}")
        return None

    def _csi_gstreamer_pipeline(self) -> str:
        return (
            f"nvarguscamerasrc sensor-id={self.camera_sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){self.camera_capture_width}, "
            f"height=(int){self.camera_capture_height}, "
            f"framerate=(fraction){self.camera_fps}/1 ! "
            f"nvvidconv flip-method={self.camera_flip_method} ! "
            f"video/x-raw, width=(int){self.camera_width}, "
            f"height=(int){self.camera_height}, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        )

    def _camera_backend_candidates(self):
        backend_map = {
            "default": cv2.CAP_ANY,
            "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
            "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
        }

        configured = str(self.camera_backend or "auto").lower()
        if configured != "auto":
            return [(configured, backend_map.get(configured, cv2.CAP_ANY))]

        return [
            ("dshow", backend_map["dshow"]),
            ("msmf", backend_map["msmf"]),
            ("default", backend_map["default"]),
        ]

    def _load_known_faces(self) -> None:
        self.known_embeddings = []

        if not self.known_faces_dir.exists():
            self._warning(f"Known faces dir does not exist: {self.known_faces_dir}")
            return

        image_paths = [
            path for path in self.known_faces_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for image_path in image_paths:
            image = self._read_image(image_path)
            if image is None:
                self._warning(f"Skip unreadable known face image: {image_path}")
                continue

            faces = self.app.get(image)
            if not faces:
                self._warning(f"No face found in known image: {image_path}")
                continue

            face = self._largest_face(faces)
            person_name = self._person_name_from_path(image_path)
            self.known_embeddings.append(
                (person_name, self._normalize(face.embedding))
            )

        self._info(
            f"Known faces loaded: count={len(self.known_embeddings)}, "
            f"dir={self.known_faces_dir}"
        )

    def _read_image(self, image_path: Path):
        data = np.fromfile(str(image_path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def _person_name_from_path(self, image_path: Path) -> str:
        if image_path.parent != self.known_faces_dir:
            return image_path.parent.name
        return image_path.stem

    def _largest_face(self, faces):
        return max(
            faces,
            key=lambda face: (
                (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
            )
        )

    def _normalize(self, embedding):
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def _info(self, message: str) -> None:
        if self.logger:
            self.logger.info(message)

    def _warning(self, message: str) -> None:
        if self.logger:
            self.logger.warning(message)

    def _error(self, message: str) -> None:
        if self.logger:
            self.logger.error(message)
