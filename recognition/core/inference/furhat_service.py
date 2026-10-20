import asyncio
import base64
import inspect
import os
import tempfile
import time
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:
    cv2 = None

from furhat_realtime_api import Events
from PIL import Image

from recognition.config import SIMILARITY_THRESHOLD
from recognition.core.inference.identity_service import identify_user
from recognition.core.utils.database import IdentityClass


class FurhatIdentitySystem:
    def __init__(
        self,
        model_name: str,
        model_path: Path | str,
        database: list[IdentityClass],
        use_webcam=False,
        callback: Callable[[dict[str, Any]], None] | None = None,
        furhat_client=None,
        similarity_threshold=SIMILARITY_THRESHOLD,
    ):
        # Shared or standalone client instance
        self.furhat = furhat_client
        self.model_name = model_name
        self.model_path = model_path
        self.gallery = database
        self.use_webcam = use_webcam
        self.threshold = similarity_threshold
        self.callback = callback

        # Session State
        self.active_user: dict[str, Any] | None = None
        self.current_user: dict[str, Any] | None = None
        self.current_role_mode: str | None = None
        self.unknown_count: int = 0
        self.max_unknown_count: int = 3
        self.is_monitoring: bool = True
        self.candidate_user = None
        self.candidate_count = 0
        self.required_confirmations = 3

        self.last_seen = time.time()
        self.user_timeout = 60

        # Buffers
        self.camera_frame_base64: str | None = None
        self.heard_text: str | None = None
        self.processed_image: str | None = None
        self._cap = None

        # Synchronization Events
        self.speaking_done = asyncio.Event()
        self.heard_event = asyncio.Event()

        # Register event handlers
        self.furhat.add_handler(
            Events.response_camera_data, self.on_camera_data
        )

    async def notify_callback(self, data: dict):
        if self.callback:
            if inspect.iscoroutinefunction(self.callback):
                await self.callback(data)
            else:
                self.callback(data)

    async def on_camera_data(self, response):
        """Called every time a camera frame arrives from Furhat."""
        self.camera_frame_base64 = response.get("image")

    async def listen(self, timeout=8.0) -> str | None:
        """Listen for user speech with timeout protection."""
        self.heard_event.clear()
        self.heard_text = None
        await self.furhat.request_listen_start()
        try:
            await asyncio.wait_for(self.heard_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self.heard_text = ""
        finally:
            await self.furhat.request_listen_stop()
        return self.heard_text

    def _init_webcam(self):
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is required for webcam mode. Install opencv-python or set use_webcam=False."
            )

        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(0)
            time.sleep(0.5)

    def _webcam_capture(self):
        self._init_webcam()
        ret, frame = self._cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    async def capture_frame(self) -> Image:
        """Fetch frame from either local webcam or Furhat camera stream."""
        if self.use_webcam:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._webcam_capture)
        else:
            # Poll for incoming frame with a 5-second timeout
            for _ in range(50):
                if self.camera_frame_base64:
                    img_bytes = base64.b64decode(self.camera_frame_base64)
                    return Image.open(BytesIO(img_bytes)).convert("RGB")
                await asyncio.sleep(0.1)
            return None

    async def identify_from_camera(self) -> dict:
        """Capture single frame and run facial identification."""
        image = await self.capture_frame()
        if image is None:
            return {"name": "unknown", "role": None, "score": 0.0}

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name
            self.processed_image = tmp_path
            print(f"Processed image: {self.processed_image}")
            result = identify_user(
                image_path=tmp_path,
                model_name=self.model_name,
                gallery=self.gallery,
                threshold=self.threshold,
                model_path=self.model_path,
            )

            print(result)
            return (
                result
                if result
                else {"name": "unknown", "role": None, "score": 0.0}
            )

        except Exception as e:
            print(f"Error during identification: {e}")
            raise e
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def handle_unknown_user(self):
        """Prompt and handle unregistered users."""
        print("[VISION] Unknown user detected. Prompting for registration...")
        await self.notify_callback({"event": "unknown_user_detected"})

    async def stop_monitoring(self):
        print("[VISION] Stopping background monitoring...")
        self.is_monitoring = False

    def is_same_user(self, identity: dict) -> bool:
        if self.active_user is None:
            return False

        return self.active_user.get("name") == identity.get(
            "name"
        ) and self.active_user.get("role") == identity.get("role")

    async def run(self):
        """Main lifecycle and tracking loop."""
        try:
            # Connect only if client isn't already connected
            if not getattr(self.furhat, "is_connected", False):
                await self.furhat.connect()
                print("[VISION] Connected to Furhat.")
        except Exception as e:
            print(f"Failed to connect to Furhat  {e}")
            return

        if self.use_webcam and cv2 is None:
            print("OpenCV is not installed. Vision webcam mode cannot start.")
            return

        if not self.use_webcam:
            print("Starting Furhat camera feed...")
            await self.furhat.request_camera_start()
            await self.furhat.request_attend_user()

        try:
            while self.is_monitoring:
                identity = await self.identify_from_camera()
                name = identity.get("name", "unknown")
                role = identity.get("role")
                score = identity.get("score", 0.0)
                print(
                    f"Detected: {name} | Role: {role} | Confidence: {score:.2f}"
                )

                now = time.time()
                # ---------------------------------------------------
                # FACE FOUND
                # ---------------------------------------------------
                if name != "unknown" and name != "Face not detected":
                    self.last_seen = now
                    self.unknown_count = 0

                    # Same user already active
                    if self.active_user and name == self.active_user["name"]:
                        self.candidate_user = None
                        self.candidate_count = 0
                        print(f"[VISION] {name} still present.")

                    # Possible NEW user
                    else:
                        if self.candidate_user == name:
                            self.candidate_count += 1
                        else:
                            self.candidate_user = name
                            self.candidate_count = 1

                        print(
                            f"[VISION] Candidate {name} ({self.candidate_count}/{self.required_confirmations})"
                        )

                        # Only switch after seeing SAME person multiple times
                        if self.candidate_count >= self.required_confirmations:
                            previous = self.active_user

                            self.active_user = identity
                            self.current_user = identity

                            self.candidate_user = None
                            self.candidate_count = 0

                            print(f"[VISION] Switching active user to {name}")

                            await self.notify_callback(
                                {
                                    "event": "user_changed",
                                    "name": name,
                                    "role": role,
                                    "score": score,
                                    "previous_user": previous,
                                }
                            )

                # ---------------------------------------------------
                # NO FACE FOUND
                # ---------------------------------------------------
                elif name == "Face not detected":
                    #
                    # Ignore temporary tracking losses
                    #
                    if self.active_user:
                        elapsed = now - self.last_seen
                        print(
                            f"[VISION] No face ({elapsed:.1f}/{self.user_timeout}s)"
                        )

                        if elapsed >= self.user_timeout:
                            print(
                                f"[VISION] {self.active_user['name']} has left."
                            )

                            await self.notify_callback(
                                {
                                    "event": "user_left",
                                    "name": self.active_user["name"],
                                    "role": self.active_user["role"],
                                }
                            )
                            self.active_user = None
                            self.current_user = None
                            self.current_role_mode = None

                            self.candidate_user = None
                            self.candidate_count = 0

                    else:
                        print("[VISION] No user present.")
                # ---------------------------------------------------
                # FACE FOUND BUT UNABLE TO RECOGNIZE THE FACE
                # ---------------------------------------------------
                else:
                    self.unknown_count += 1
                    print(
                        f"[VISION] Unknown face detected ({self.unknown_count}/{self.max_unknown_count})"
                    )

                    if self.unknown_count >= self.max_unknown_count:
                        await self.handle_unknown_user()
                        self.unknown_count = 0

                await asyncio.sleep(0.5)
        except KeyboardInterrupt:
            print(
                "[VISION] Monitoring loop terminated. Cleaning up resources..."
            )
        finally:
            if not self.use_webcam:
                await self.furhat.request_camera_stop()
            if self._cap and self._cap.isOpened():
                self._cap.release()
