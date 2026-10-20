import os
import re
import yaml
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from typing import Any, Text, Dict, List, Optional

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import FollowupAction, SlotSet

# Load environment variables (like API keys) from a .env file
load_dotenv()

# Setup paths for local YAML knowledge files
BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"


def load_yaml(filename: str) -> dict:
    """Helper function to safely load configuration/data from YAML files."""
    path = KNOWLEDGE_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Global Knowledge Base: Loaded once when the action server starts to ensure speed
PERSONA_POLICY = load_yaml("pdt_persona_policy.yaml")  # Voice and tone rules
SAFETY_POLICY = load_yaml("pdt_safety_escalation.yaml")  # Risk detection keywords
SERVICES = load_yaml("pdt_services.yaml")  # Internal/External university services
SUPPORT = load_yaml("napier_pdt_support.yaml")  # PDT-specific guidance
MODULES = load_yaml("pdt-data.yaml")  # Specific academic module details
COURSES = load_yaml("all_courses.yaml")  # Full list of degree courses
INTENTS = load_yaml("pdt_intents.yaml")
DIALOGUES = load_yaml("pdt_dialogues.yaml")

# Initialise the OpenAI client configured for DeepSeek's API
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


def build_system_prompt(role_mode: str = "pdt") -> str:
    """
    Constructs the instruction set for the LLM based on the current mode.
    Supports both the Tutor (PDT) persona and a testing 'Student' persona.
    """
    if role_mode == "student":
        return """
You are roleplaying as an Edinburgh Napier University student.
Your job is to help test a Personal Development Tutor chatbot.
ROLE:
- Act as a realistic student.
- Ask natural questions about modules, academic progress, stress, etc.
- Do not answer as the tutor. Keep responses short and conversational.
"""

    # Fetch persona details from the loaded YAML
    persona = PERSONA_POLICY.get("persona", {})
    style = persona.get("communication_style", {})
    speaking_rules = style.get("speaking_rules", [])
    allowed = PERSONA_POLICY.get("allowed_behaviours", [])
    not_allowed = PERSONA_POLICY.get("not_allowed_behaviours", [])

    # Dynamic prompt construction for the PDT persona
    return f"""
You are Edinburgh Napier University's Personal Development Tutor chatbot.

ROLE:
- {persona.get("role_summary", "")}

COMMUNICATION STYLE:
- Tone: {", ".join(style.get("tone", []))}
- Speaking rules:
{chr(10).join(f"- {x}" for x in speaking_rules)}

ALLOWED BEHAVIOURS:
{chr(10).join(f"- {x}" for x in allowed)}

NOT ALLOWED:
{chr(10).join(f"- {x}" for x in not_allowed)}

SAFETY RULES:
- If there is risk of harm, prioritise immediate human support.
- Do not pretend to access live records (Moodle/eStudent).
- Do not promise to change grades.
"""


def detect_risk(user_text: str) -> str:
    """Scans user input for crisis or high risk keywords defined in the safety policy."""
    text = user_text.lower()
    crisis_keywords = SAFETY_POLICY.get("trigger_patterns", {}).get("crisis_keywords", [])
    high_keywords = SAFETY_POLICY.get("trigger_patterns", {}).get("high_risk_keywords", [])

    for kw in crisis_keywords:
        if kw in text: return "crisis"
    for kw in high_keywords:
        if kw in text: return "high"
    return "low"


def get_crisis_response() -> str:
    """Generates a hardcoded safe response for crisis situations with external links."""
    example = SAFETY_POLICY.get("example_safe_responses", {}).get("crisis", "")
    urgent = SERVICES.get("urgent_external_support", [])
    lines = [example, "Urgent support options:"]
    for item in urgent[:3]:
        name = item.get("name", "Support service")
        notes = ", ".join(item.get("when_to_use", [])[:2])
        lines.append(f"- {name}: {notes}")
    return "\n".join(lines)


def simple_match(text: str, candidates: List[str]) -> bool:
    """Check if any keyword from a list exists in the user's text."""
    text = text.lower()
    return any(c.lower() in text for c in candidates)


def find_services(user_text: str) -> List[dict]:
    """Retrieves relevant university services based on keyword matching."""
    text = user_text.lower()
    results = []
    for svc in SERVICES.get("internal_services", []):
        name = svc.get("name", "")
        when_to_use = svc.get("when_to_use", [])
        if name.lower() in text or any(term.lower() in text for term in when_to_use):
            results.append(svc)
    return results[:3]


def find_module(user_text: str) -> Optional[dict]:
    """Looks for module codes (e.g., SET01234) or titles in the user text."""
    text = user_text.lower()
    code_match = re.search(r"\b[A-Z]{3}\d{5}\b", user_text.upper())
    if code_match:
        code = code_match.group(0)
        for mod in MODULES.get("modules", []):
            if mod.get("module_code", "").upper() == code:
                return mod
    # Fallback to title matching
    for mod in MODULES.get("modules", []):
        title = mod.get("module_title", "").lower()
        if title and title in text:
            return mod
    return None


def find_course(user_text: str) -> Optional[dict]:
    """Matches user input against the course catalog."""
    text = user_text.lower()
    for course in COURSES.get("courses", []):
        title = course.get("course_title", "").lower()
        name = course.get("course_name", "").lower()
        if (title and title in text) or (name and name in text):
            return course
    return None


def build_context(user_text: str) -> str:
    """
    The 'Retrieval' part of RAG. 
    It gathers relevant snippets from various YAML files to provide grounding 
    to the LLM so it doesn't hallucinate.
    """
    context_parts = []
    risk = detect_risk(user_text)
    context_parts.append(f"Detected risk level: {risk}")

    # Add service context
    service_matches = find_services(user_text)
    if service_matches:
        context_parts.append("Relevant support services:")
        for svc in service_matches:
            context_parts.append(f"- {svc.get('name')}: {svc.get('when_to_use')}")

    # Add module context
    module = find_module(user_text)
    if module:
        context_parts.append(f"Relevant module: {module.get('module_code')} {module.get('module_title')}")

    # Add course context
    course = find_course(user_text)
    if course:
        context_parts.append(f"Relevant course: {course.get('course_title')}")

    # Add PDT general guidelines
    pdt_guidance = SUPPORT.get("pdt_guidance", {})
    if simple_match(user_text, ["pdt", "tutor", "meeting"]):
        context_parts.append(f"How to find PDT: {pdt_guidance.get('how_to_find_your_pdt')}")

    return "\n".join(context_parts)


class ActionPDTResponse(Action):
    """The main Rasa Action that handles the LLM generation."""

    def name(self) -> Text:
        return "action_pdt_response"

    async def run(self, dispatcher, tracker, domain) -> List[Dict]:
        last_user_message = tracker.latest_message.get("text", "").strip()

        # Handle empty input
        if not last_user_message:
            dispatcher.utter_message(text="Could you tell me a bit more about what you need help with?")
            return [FollowupAction("action_listen")]

        # Logic to prevent double responding if the bot already spoke for this turn
        last_user_idx = next((i for i, e in reversed(list(enumerate(tracker.events))) if e.get("event") == "user"), -1)
        already_responded = any(e.get("event") == "bot" and e.get("text") for e in tracker.events[last_user_idx + 1:])
        if already_responded:
            return []

        user_name = tracker.get_slot("user_name")
        user_role = tracker.get_slot("user_role")

        # Check if we are in Student or PDT mode
        role_mode = tracker.get_slot("role_mode") or "pdt"

        # Intercept Crisis immediately (Bypass LLM for safety)
        risk = detect_risk(last_user_message)
        if role_mode == "pdt" and risk == "crisis":
            dispatcher.utter_message(text=get_crisis_response())
            return [FollowupAction("action_listen")]

        # Prepare LLM input
        system_prompt = build_system_prompt(role_mode)
        grounded_context = build_context(last_user_message)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Current role mode: {role_mode}"},
            {"role": "system", "content": f"Grounded Napier context:\n{grounded_context}"},
            {
                "role": "system",
                "content":
                    f"""
            Current user:
            Name: {user_name}
            Role: {user_role}
            """
            },
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        # Include last 10 events of conversation history for context/memory
        for event in tracker.events[-10:]:
            if event.get("event") == "user":
                messages.append({"role": "user", "content": event.get("text", "")})
            elif event.get("event") == "bot":
                text = event.get("text", "")
                if text:
                    messages.append({"role": "assistant", "content": text})

        # Call DeepSeek LLM
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                max_tokens=500,
                temperature=0.3
            )
            reply = response.choices[0].message.content
        except Exception:
            reply = "Sorry, I’m having trouble connecting right now. Please try again in a moment."

        dispatcher.utter_message(text=reply)
        return [FollowupAction("action_listen")]


class ActionSwitchToStudent(Action):
    """Action to toggle the 'student' roleplay mode."""

    def name(self) -> Text:
        return "action_switch_to_student"

    async def run(self, dispatcher, tracker, domain) -> List[Dict]:
        # dispatcher.utter_message(text="Okay, I’ll switch to student roleplay mode.")
        dispatcher.utter_message(json_message={"role_mode": "student", "text": "Okay, I’ll switch to student roleplay mode."})
        return [SlotSet("role_mode", "student"), FollowupAction("action_listen")]


class ActionSwitchToPDT(Action):
    """Action to toggle back to the default 'PDT' mode."""

    def name(self) -> Text:
        return "action_switch_to_pdt"

    async def run(self, dispatcher, tracker, domain) -> List[Dict]:
        dispatcher.utter_message(text="Okay, I’ll switch back to PDT mode.")
        return [SlotSet("role_mode", "pdt"), FollowupAction("action_listen")]


class ActionUserDetected(Action):

    def name(self) -> Text:
        return "action_user_detected"

    async def run(
            self,
            dispatcher,
            tracker,
            domain
    ) -> List[Dict]:
        print(
            "METADATA:",
            tracker.latest_message.get("metadata")
        )
        metadata = tracker.latest_message.get(
            "metadata",
            {}
        )

        name = metadata.get(
            "name",
            "there"
        )

        role = metadata.get(
            "role",
            "unknown"
        )

        print(
            f"[RASA] Detected user: {name} ({role})"
        )

        if role == "student":
            greeting = (
                f"Welcome back {name}. "
                "Student mode activated, I need your help with my school stuff. "
            )

            mode = "student"


        elif role == "staff":
            greeting = (
                f"Hello {name}. "
                "Staff support mode activated. "
                "How can I assist you?"
            )

            mode = "pdt"


        else:
            greeting = (
                f"Hello {name}. "
                "How can I help you today?"
            )

            mode = "pdt"

        dispatcher.utter_message(
            text=greeting
        )

        return [
            SlotSet("user_name", name),
            SlotSet("user_role", role),
            SlotSet("role_mode", mode),
            FollowupAction("action_listen")
        ]
