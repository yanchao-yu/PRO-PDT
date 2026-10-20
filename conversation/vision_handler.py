"""
Vision Integration Module
Handles face recognition events and integrates them with the Rasa/Furhat dialogue system.
"""

import asyncio
from typing import Optional

from recognition.core.utils.database import Database
from recognition.interaction.vision import Vision


class VisionIntegration:
    """
    Integrates vision/face recognition with the Rasa chatbot system.
    Handles user identification and role-based dialogue routing.
    """

    def __init__(self, furhat_bridge=None, use_webcam: bool = False):
        """
        Initialize Vision Integration.

        Args:
            furhat_bridge: Reference to RasaAsyncFurhatBridge instance
            host: Furhat host address
            use_webcam: Whether to use webcam for face recognition
        """
        self.furhat_bridge = furhat_bridge
        self.use_webcam = use_webcam
        self.vision_system = None
        self.monitoring_task = None
        self.current_user = None
        self.user_session_map = {}  # Map user names to session IDs

    async def initialize(self):
        """Initialize the vision system."""
        if Vision is None:
            print("Vision class not available. Vision integration disabled.")
            return False

        identities = Database(db_path="faces.db").get_all()
        try:
            self.vision_system = Vision(
                callback=self.handle_vision_event,
                use_webcam=self.use_webcam,
                furhat_client=self.furhat_bridge.furhat,
                model_name="inception_resnet_v1",
                weight_path="resnet_best.pth",
                similarity_threshold=0.40,
                database=identities,
            )
            print("[Vision] System initialized successfully")
            return True
        except Exception as e:
            print(f"[Vision] Failed to initialize: {e}")
            return False

    async def handle_vision_event(self, event_data: dict):
        """
        Callback triggered whenever user state changes (enters/leaves frame, role changes).

        Args:
            event_data: Dictionary containing event type, name, role, etc.
        """
        print(f"[Vision] Received event: {event_data}")
        if not event_data:
            return

        event_type = event_data.get("event")
        name = event_data.get("name", "unknown")
        role = event_data.get("role")

        print(f"[VISION EVENT] Type: {event_type} | Name: {name} | Role: {role}")

        if event_type == "user_changed":
            await self._handle_user_changed(name, role)
        elif event_type == "user_left":
            await self._handle_user_left(name)
        elif event_type == "unknown_user_detected":
            await self.furhat_bridge.furhat.request_speak_text(
                "Hello there! I don't believe we've met yet. Could you tell me your name?"
            )
            # TODO add the code completion of collecting the user name and role and sending it back to the recogntion model

    async def _handle_user_changed(self, name: str, role: Optional[str]):
        """Handle when a user enters or changes."""
        if not self.furhat_bridge:
            print("[Vision] No furhat_bridge available, skipping dialogue")
            return

        self.current_user = name

        if name == "unknown":
            # Unknown user - ask for name
            print("[Vision] Unknown user detected")
            await self.furhat_bridge.furhat.request_speak_text(
                "Hello there! I don't believe we've met yet. Could you tell me your name?"
            )

        elif role and role == "student":
            # Student user
            print(f"[Vision] Student detected: {name}")
            self.furhat_bridge.chatbot.set_session_id(f"student_{name}")
            self.furhat_bridge.chatbot.set_user(name=name, role="student")
            print("[Vision] Sending identity to Rasa...")

            response = await self.furhat_bridge.chatbot.send_user_identity(
                name, "staff"
            )
            print(f"[Vision] Rasa response: {response}")

            for message in response:
                if message.get("text"):
                    await self.furhat_bridge.on_chatbot_response_ready(message["text"])

        elif role and role == "staff":
            # Staff user
            print(f"[Vision] Staff detected: {name}")
            self.furhat_bridge.chatbot.set_session_id(f"staff_{name}")
            self.furhat_bridge.chatbot.set_user(name=name, role="staff")
            # Update voice for staff interaction
            await self.furhat_bridge.furhat.request_voice_config(
                language="en-US", gender="male"
            )

            # Trigger Rasa to set role mode for staff
            response = await self.furhat_bridge.chatbot.send_user_identity(
                name, "student"
            )

            for message in response:
                if message.get("text"):
                    await self.furhat_bridge.on_chatbot_response_ready(message["text"])

    async def _handle_user_left(self, name: str):
        """Handle when a user leaves."""
        if not self.furhat_bridge:
            return

        print(f"[Vision] User left: {name}")

        if name and name != "unknown":
            farewell = f"Goodbye {name}, have a nice day!"
            await self.furhat_bridge.furhat.request_speak_text(farewell)

        self.current_user = None

    async def start(self):
        """Start vision monitoring."""
        if not self.vision_system:
            if not await self.initialize():
                return

        print("[Vision] Starting monitoring...")
        self.monitoring_task = asyncio.create_task(
            self.vision_system.start_monitoring()
        )
        return

    async def stop(self):
        """Stop vision monitoring."""
        if self.monitoring_task and not self.monitoring_task.done():
            print("[Vision] Stopping monitoring...")
            self.monitoring_task.cancel()
            if self.vision_system:
                await self.vision_system.stop_monitoring()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
