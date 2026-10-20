import asyncio
import aiohttp
import argparse
import signal
from dotenv import load_dotenv
from furhat_realtime_api import AsyncFurhatClient, Events

# code largely adapted from https://github.com/FurhatRobotics/realtime-api-examples/
# changed to use HTTP requests rather than purely websockets due to version compatibility issues between furhat and rasa for websocket use

# Import vision integration (optional)
from vision_handler import VisionIntegration


class Chatbot:
    def __init__(self, rasa_url: str = "http://localhost:5005/webhooks/rest/webhook"):
        self.rasa_url = rasa_url
        self.session_id = "furhat_user_003"  # basic session ID for the one current user, can be extended to support multiple users/sessions if needed
        self.dialog_history = []
        self.current_user_utt = None
        self.rasa_task = None
        self.shutting_down = False
        self.role_mode = None
        self.user_name = None
        self.user_role = None

    def set_user(self, name: str, role: str):
        self.user_name = name
        self.user_role = role

    def commit_user(self):
        if self.current_user_utt is None:
            return
        self.dialog_history.append({"role": "user", "content": self.current_user_utt})
        self.current_user_utt = None

    def commit_robot(self, message: str):
        self.dialog_history.append({"role": "assistant", "content": message})

    def initiate_request(self, text, callback):
        if self.shutting_down:
            return
        self.current_user_utt = text
        self.rasa_task = asyncio.create_task(self.make_rasa_request(callback))

    def cancel_request(self):
        self.current_user_utt = None
        if self.rasa_task and not self.rasa_task.done():
            print("[Rasa] Cancelling request...")
            self.rasa_task.cancel()

    async def send_user_identity(self, name: str, role: str):

        payload = {
            "sender_id": self.session_id,
            "message": "/user_detected",
            "metadata": {"name": name, "role": role},
        }
        print("[Rasa] Sending identity:")
        print(payload)

        async with aiohttp.ClientSession() as session:
            async with session.post(self.rasa_url, json=payload) as resp:
                return await resp.json()

    async def make_rasa_request(self, callback, is_starter=False):
        # send message to Rasa and handle multiple response messages
        try:
            payload = {
                "message": self.current_user_utt if not is_starter else "/greet",
                "sender_id": self.session_id,
            }

            if is_starter:
                print(f"[Rasa] Fetching conversation starter...")
            else:
                print(f"[Rasa] Sending: {self.current_user_utt}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.rasa_url, json=payload, timeout=10
                ) as resp:
                    if resp.status == 200:
                        response_data = await resp.json()

                        # rasa returns a list of responses
                        if (
                            response_data
                            and isinstance(response_data, list)
                            and len(response_data) > 0
                        ):
                            # print(f"{response_data}")
                            # extract text from each response item
                            response_texts = []
                            for item in response_data:
                                if item.get("text"):
                                    # append text to list
                                    response_texts.append(item.get("text"))
                                elif item.get("custom"):
                                    if item["custom"].get("text"):
                                        response_texts.append(
                                            item["custom"].get("text")
                                        )

                                    if item["custom"].get("role_mode"):
                                        self.role_mode = item["custom"].get("role_mode")
                            # join all response texts into one string for furhat to speak, or handle empty responses
                            if response_texts:
                                print(
                                    f"[Rasa] {len(response_texts)} response(s): {response_texts}"
                                )

                                response_text = ". ".join(response_texts)

                                if not self.shutting_down:
                                    # Pass the entire list to callback
                                    await callback(response_text)
                                return response_text
                            else:
                                print("[Rasa] Empty responses")
                                fallback = [
                                    "I received an empty response. Please try again."
                                ]
                                if not self.shutting_down:
                                    await callback(fallback)
                                return fallback

        except aiohttp.ClientConnectorError as e:
            print(f"[Rasa] Failed to connect to Rasa server {e}")
            fallback = "I can't connect to my conversation server. Is Rasa running?"
            if not self.shutting_down:
                await callback(fallback)
            return fallback
        except asyncio.TimeoutError:
            print("[Rasa] Request timed out")
            fallback = "The server is taking too long to respond."
            if not self.shutting_down:
                await callback(fallback)
            return fallback
        except asyncio.CancelledError:
            print("[Rasa] Request was cancelled")
            raise
        except Exception as e:
            print(f"[Rasa] Unexpected error: {e}")
            fallback = "An error occurred while processing your message."
            if not self.shutting_down:
                await callback(fallback)
            return fallback

    # get PDT starter message from RASA to avoid default furhat chatbot message
    async def get_conversation_starter(self, callback):
        try:
            async with aiohttp.ClientSession() as session:
                init_payload = {
                    "message": "/greet",  # trigger rasa greet message on startup
                    "sender_id": self.session_id,
                }
                await session.post(self.rasa_url, json=init_payload, timeout=5)
        except Exception as e:
            print(f"[Rasa] Starter message unavailable: {e}")
            fallback: str = "Hello, I am Furhat. How can I help you today?"
            if callback:
                await callback(fallback)
            return fallback

        self.current_user_utt = None
        return await self.make_rasa_request(callback, is_starter=True)

    def set_rasa_url(self, rasa_url):
        self.rasa_url = rasa_url

    def set_session_id(self, session_id):
        self.session_id = session_id

    def set_shutting_down(self, value):
        self.shutting_down = value


class RasaAsyncFurhatBridge:
    def __init__(
        self,
        host: str = "127.0.0.1",
        auth_key=None,
        rasa_url: str = "http://localhost:5005/webhooks/rest/webhook",
        enable_vision: bool = True,
        use_webcam: bool = False,
    ):
        load_dotenv(override=True)

        self.rasa_url = rasa_url
        self.conversation_starter = None  # this will be fetched from rasa
        self.stop_event = asyncio.Event()
        self.shutting_down = False
        self.host = host

        # connect to furhat realtime API (currently only local)
        self.furhat = AsyncFurhatClient(host, auth_key=auth_key)

        # initialize chatbot with rasa connection
        self.chatbot = Chatbot(rasa_url=self.rasa_url)

        # initialize vision integration (optional)
        self.vision_integration = None
        if enable_vision and VisionIntegration:
            print("Enabling vison...")
            self.vision_integration = VisionIntegration(
                furhat_bridge=self, use_webcam=use_webcam
            )

    def setup_signal_handlers(self):
        # setup signal handlers for graceful shutdown on SIGINT and SIGTERM
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, shutting down gracefully...")
            asyncio.create_task(self.shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, signal_handler)

    # shutdown everything when finished
    async def shutdown(self):
        if self.shutting_down:
            return

        self.shutting_down = True
        self.chatbot.set_shutting_down(True)
        print("Initiating shutdown...")

        try:
            # stop vision monitoring if active
            if self.vision_integration:
                await self.vision_integration.stop()

            # cancel any ongoing rasa request
            self.chatbot.cancel_request()
            # stop furhat from listening and talking
            await self.furhat.request_listen_stop()
            await self.furhat.request_speak_stop()
        except Exception as e:
            print(f"Error during shutdown: {e}")

        self.stop_event.set()

    # the user has started speaking, so cancel any rasa/deepseek requests currently ongoing
    async def on_hear_start(self, event):
        await self.furhat.request_led_set(
            color="#0000FF"
        )  # set LED to blue when furhat listening
        if not self.shutting_down:
            self.chatbot.cancel_request()

    # user has stopped speaking, send request to rasa/deepseek
    async def on_hear_end(self, event):
        if not self.shutting_down:
            user_text = event.get("text", "")
            if user_text and user_text.strip():
                self.chatbot.initiate_request(user_text, self.on_chatbot_response_ready)
            else:
                # Handle empty transcription
                await self.on_chatbot_response_ready(
                    "Sorry, I didn't catch that. Could you please repeat?"
                )

    # rasa/deepseek has sent back a response, get furhat ready to say it
    async def on_chatbot_response_ready(self, text: str):
        if not self.shutting_down and text:
            # Try SSML first (may provide better timing for lipsync). If the
            # AsyncFurhatClient implementation doesn't expose SSML support,
            # fall back to plain text speak.
            ssml = f"<speak>{text}</speak>"
            try:
                if hasattr(self.furhat, "request_speak_ssml"):
                    await self.furhat.request_speak_ssml(ssml)
                    return
            except Exception as e:
                print(f"[Furhat] request_speak_ssml failed, falling back to text: {e}")

            # Fallback to plain text speak
            await self.furhat.request_speak_text(text)

    # furhat started speaking, commit user response to chat history
    async def on_speak_start(self, event):
        await self.furhat.request_led_set(
            color="#00FF00"
        )  # set LED to green when furhat speaking
        if not self.shutting_down:
            self.chatbot.commit_user()

    # furhat finished speaking, commit furhat response to chat history
    async def on_speak_end(self, event):
        if not self.shutting_down:
            spoken_text = event.get("text", "")
            if spoken_text:
                self.chatbot.commit_robot(spoken_text)

    # main run loop
    async def run(self):
        self.setup_signal_handlers()
        print("Starting Rasa-Furhat dialog...")
        print(f"Rasa URL: {self.rasa_url}")
        print("Press Ctrl+C to stop gracefully")

        try:
            await self.furhat.connect()
            print(f"Connected to Furhat at {self.host}")
        except Exception as e:
            print(f"Failed to connect to Furhat on {self.host}: {e}")
            print("\nTroubleshooting:")
            print("1. Is the Furhat SDK running?")
            print("2. Is the Realtime API enabled?")
            exit(0)

        # await self.furhat.request_voice_config(language="en-US", gender="male")

        # register event handlers for furhat hearing & talking
        self.furhat.add_handler(Events.response_hear_start, self.on_hear_start)
        self.furhat.add_handler(Events.response_hear_end, self.on_hear_end)
        self.furhat.add_handler(Events.response_speak_start, self.on_speak_start)
        self.furhat.add_handler(Events.response_speak_end, self.on_speak_end)

        # start furhat attending user
        await self.furhat.request_attend_user()

        # grab conversation starter from rasa scripts
        print("Fetching conversation starter from Rasa...")

        # wait for starter message as a rasa takes a moment to connect to furhat
        starter_ready = asyncio.Event()
        starter_text = None

        async def set_starter(text):
            nonlocal starter_text
            starter_text = text
            starter_ready.set()

            # start vision monitoring if enabled

        print(f"vision enabled {self.vision_integration}")
        if self.vision_integration:
            print("[Vision] Starting vision monitoring...")
            await self.vision_integration.start()

        # grab starter message
        # await self.chatbot.get_conversation_starter(set_starter)

        # wait for starter, if none then timeout after 10 seconds
        # try:
        #     await asyncio.wait_for(starter_ready.wait(), timeout=10.0)
        #     self.conversation_starter = starter_text
        # except asyncio.TimeoutError:
        #     print("Timeout getting starter from Rasa, using default greeting")
        #     self.conversation_starter = "Hello, I am Furhat. How can I help you today?"

        print(f"{self.chatbot.role_mode}")
        # print(f"{self.conversation_starter}")

        if self.chatbot.role_mode in ["student", "staff"]:
            await self.furhat.request_voice_config(language="en-US", gender="female")
        elif self.chatbot.role_mode is None:
            await self.furhat.request_voice_config(language="en-US", gender="male")

        # greet the user with Rasa's starter message
        # await self.furhat.request_speak_text(self.conversation_starter)

        # start listening with smart parameters
        await self.furhat.request_listen_start(
            concat=True,  # Concatenate user speech into a single utterance
            stop_no_speech=False,  # Don't stop if no speech detected
            stop_user_end=False,  # Don't stop when user stops speaking
            stop_robot_start=True,  # Stop listening when robot starts speaking
            resume_robot_end=True,  # Resume listening after robot finishes
            end_speech_timeout=1,  # Timeout for end of speech detection
        )

        # wait for event to shut down
        await self.stop_event.wait()

        print("Shutting down...")
        await self.furhat.disconnect()
        print("Disconnected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        default="172.20.10.3",
        help="Furhat robot IP address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--auth_key", type=str, default=None, help="Authentication key for Realtime API"
    )
    parser.add_argument(
        "--rasa_url",
        type=str,
        default="http://localhost:5005/webhooks/rest/webhook",
        help="Rasa webhook URL",
    )
    parser.add_argument(
        "--session_id",
        type=str,
        default="furhat_user_001",
        help="Session ID for conversation memory",
    )
    parser.add_argument(
        "--enable_vision",
        action="store_true",
        help="Enable vision/face recognition integration",
    )
    parser.add_argument(
        "--use_webcam",
        action="store_true",
        help="Use webcam for vision integration (default: False)",
    )

    args = parser.parse_args()

    bridge = RasaAsyncFurhatBridge(
        host=args.host,
        auth_key=args.auth_key,
        rasa_url=args.rasa_url,
        enable_vision=args.enable_vision,
        use_webcam=args.use_webcam,
    )
    bridge.chatbot.set_session_id(args.session_id)

    asyncio.run(bridge.run())
