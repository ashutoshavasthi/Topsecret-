"""The conversational drone assistant.

    you: "fly to the door and tell me what you see"
     -> planner (local LLM, or built-in rules) -> [go_to(door), describe()]
     -> skills run through the live session (safety layer always on)
     -> a short spoken answer, summarised by the LLM from what actually happened

Design rules:
* "stop" (and the app's STOP button) never waits for the model. It cancels whatever
  is running and holds position immediately.
* The model only chooses from the listed skills. Arguments are validated, and the
  flight brain's failsafes still decide what is safe.
* If the model is missing, slow or returns nonsense, the built-in parser keeps the
  basic commands working, and the assistant says so.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Callable

from skymind.assistant import rules
from skymind.assistant.llm import LLMError, OllamaClient
from skymind.assistant.skills import DroneSkills, SkillResult
from skymind.config import SkyMindConfig
from skymind.live import LiveSession

MAX_ACTIONS = 6
READ_ONLY = {"status", "describe", "photo", "room_report"}  # answered without interrupting what the drone is doing


@dataclass
class Message:
    id: int
    role: str  # "user" | "assistant" | "system"
    text: str
    t: float
    speak: bool = False

    def to_dict(self) -> dict:
        return {"id": self.id, "role": self.role, "text": self.text, "t": self.t, "speak": self.speak}


def plan_schema(tool_names: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "reply": {"type": "string"},
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"tool": {"type": "string", "enum": tool_names}, "args": {"type": "object"}},
                    "required": ["tool"],
                },
            },
        },
        "required": ["reply", "actions"],
    }


SUMMARY_SCHEMA = {"type": "object", "properties": {"reply": {"type": "string"}}, "required": ["reply"]}


class DroneAssistant:
    def __init__(self, session: LiveSession, cfg: SkyMindConfig, llm: OllamaClient | None = None,
                 on_message: Callable[[Message], None] | None = None):
        self.session = session
        self.cfg = cfg
        self.a = cfg.assistant
        self.cancel = threading.Event()  # cancels the current *motion* task
        self.skills = DroneSkills(session, self.cancel)  # catalogue (names, descriptions)
        self.messages: list[Message] = []
        self._lock = threading.Lock()
        self._task_lock = threading.Lock()
        self._motion: threading.Thread | None = None
        self._threads: list[threading.Thread] = []
        self.on_message = on_message
        self.llm = llm
        self.llm_ok = False
        self.vision_ok = False
        if self.a.use_llm:
            self.llm = llm or OllamaClient(self.a.ollama_url, self.a.timeout)
            self.llm_ok = self.llm.has_model(self.a.model)
            self.vision_ok = self.llm.has_model(self.a.vision_model)
        if self.llm_ok:
            vision = f" and {self.a.vision_model} for vision" if self.vision_ok else " (no vision model: I'll describe what my detector sees)"
            self._say(f"{self.a.name} online. Using {self.a.model}{vision}. How can I help?", role="system")
        else:
            why = "AI disabled in config" if not self.a.use_llm else f"local model '{self.a.model}' not found (is Ollama running?)"
            self._say(f"{self.a.name} online with built-in commands only: {why}. Try 'take off', 'go to the door', 'look around', 'come back'.", role="system")

    # ------------------------------------------------------------ messages
    def _say(self, text: str, role: str = "assistant", speak: bool = True) -> Message:
        with self._lock:
            msg = Message(len(self.messages) + 1, role, text, round(time.time(), 2), speak and role == "assistant")
            self.messages.append(msg)
        if self.on_message:
            self.on_message(msg)
        return msg

    def messages_after(self, after_id: int) -> list[dict]:
        with self._lock:
            return [m.to_dict() for m in self.messages if m.id > after_id]

    @property
    def busy(self) -> bool:
        self._threads = [t for t in self._threads if t.is_alive()]
        return bool(self._threads)

    def wait_idle(self, timeout: float = 120.0) -> None:
        end = time.time() + timeout
        while self.busy and time.time() < end:
            time.sleep(0.02)

    # ------------------------------------------------------------ entry points
    def stop(self) -> None:
        """Emergency stop: never goes through the model."""
        self.cancel.set()
        self.session.hold()
        with self._task_lock:
            motion = self._motion
        if motion is not None and motion is not threading.current_thread():
            motion.join(timeout=3.0)
        self._say("Stopped. Holding position.")

    def _spawn(self, target, *args) -> None:
        t = threading.Thread(target=target, args=args, name="skymind-assistant", daemon=True)
        self._threads.append(t)
        t.start()

    def handle(self, text: str, wait: bool = False) -> None:
        text = text.strip()
        if not text:
            return
        self._say(text, role="user", speak=False)
        if rules.is_stop(text) and len(text.split()) <= 4:
            self.stop()
            return
        self._spawn(self._run, text)
        if wait:
            self.wait_idle()

    def run_tool(self, tool: str, args: dict | None = None, wait: bool = False) -> None:
        """Direct button actions from the app (no model involved)."""
        if tool == "hold":
            self.stop()
            return
        self._spawn(self._execute_task, [{"tool": tool, "args": args or {}}], "", True)
        if wait:
            self.wait_idle()

    # ------------------------------------------------------------ planning
    def _system_prompt(self) -> str:
        tools = "\n".join(
            f"- {s.name}({', '.join(f'{k}: {v}' for k, v in s.args.items())}): {s.description}" for s in self.skills.skills.values()
        )
        st = self.session.status()
        status = {k: st.get(k) for k in ("mode", "position", "yaw_deg", "battery", "min_range", "detections", "failsafe")}
        places = ", ".join(self.cfg.mission.places) or "none"
        return f"""You are {self.a.name}, the onboard assistant of a small camera drone. You talk to your
operator like a calm, capable co-pilot and reply in at most two short sentences (your reply is spoken aloud).

You act ONLY through these tools:
{tools}

Known places: {places}
Current status (JSON): {json.dumps(status)}

Rules:
- Output JSON: {{"reply": what you say now, "actions": list of tool calls to run in order}}.
- Your reply describes what you are ABOUT to do. Never claim results before the tools run.
- If the request is unclear, ask a short question and use no actions.
- If you're on the ground and asked to fly somewhere, the go_to tool takes off by itself.
- Refuse anything harmful: weapons, dropping things on people, chasing, spying on or secretly
  recording people, flying outside the allowed area. Refuse politely, with no actions.
- Never invent tools or places. At most {MAX_ACTIONS} actions."""

    def _plan(self, text: str) -> rules.Plan:
        quick = rules.parse(text, list(self.cfg.mission.places))
        if not self.llm_ok:
            example = next((p for p in self.cfg.mission.places if p != "home"), "door")
            return quick or rules.Plan(
                f"Sorry, without my AI model I only understand simple commands like 'take off', 'go to the {example}', "
                "'look around', 'find the red marker' or 'come back'."
            )
        # Simple, safety-relevant commands skip the model for speed and reliability.
        if quick and quick.actions and quick.actions[0]["tool"] in ("takeoff", "land", "return_home", "hold"):
            return quick
        history = []
        for m in self.messages[-(2 * self.a.history_turns + 1) : -1]:
            if m.role in ("user", "assistant"):
                history.append({"role": m.role, "content": m.text})
        msgs = [{"role": "system", "content": self._system_prompt()}, *history, {"role": "user", "content": text}]
        try:
            data = self.llm.chat_json(self.a.model, msgs, plan_schema(list(self.skills.skills)))
        except LLMError as exc:
            if quick:
                quick.reply = (quick.reply + " (AI unavailable, used built-in command)").strip()
                return quick
            return rules.Plan(f"My AI model didn't answer ({exc}). Try a simple command.")
        actions = [a for a in data.get("actions", []) if isinstance(a, dict) and a.get("tool") in self.skills.skills][:MAX_ACTIONS]
        for a in actions:
            if not isinstance(a.get("args"), dict):
                a["args"] = {}
        return rules.Plan(str(data.get("reply", "")).strip(), actions, source="llm")

    def _summarise(self, request: str, results: list[tuple[str, SkillResult]]) -> str:
        facts = "\n".join(f"- {tool}: {'OK' if r.ok else 'FAILED'}: {r.text}" for tool, r in results)
        fallback = " ".join(r.text for _, r in results)
        if not self.llm_ok:
            return fallback
        msgs = [
            {"role": "system", "content": f"You are {self.a.name}, a drone assistant. Answer the operator in at most two short "
                                          "spoken sentences, using ONLY the facts given. Do not invent details."},
            {"role": "user", "content": f"Operator asked: {request}\nWhat happened:\n{facts}"},
        ]
        try:
            return str(self.llm.chat_json(self.a.model, msgs, SUMMARY_SCHEMA).get("reply", fallback)).strip() or fallback
        except LLMError:
            return fallback

    # ------------------------------------------------------------ execution
    def _run(self, text: str) -> None:
        plan = self._plan(text)
        if plan.reply:
            self._say(plan.reply)
        if plan.actions:
            self._execute_task(plan.actions, text, False, plan.source)

    def _execute_task(self, actions: list[dict], request: str, announce: bool, source: str = "rules") -> None:
        read_only = all(a["tool"] in READ_ONLY for a in actions)
        if read_only:
            cancel = threading.Event()  # questions don't interrupt motion, and motion doesn't cancel them
        else:
            with self._task_lock:  # a new motion request replaces the one in progress
                previous = self._motion
                self.cancel.set()
                self._motion = threading.current_thread()
            if previous is not None and previous is not threading.current_thread():
                previous.join(timeout=5.0)
            cancel = self.cancel = threading.Event()
        results = self._execute(actions, announce, DroneSkills(self.session, cancel), cancel)
        if not read_only:
            with self._task_lock:
                if self._motion is threading.current_thread():
                    self._motion = None
        if cancel.is_set() or announce:
            return
        if any(r.observation for _, r in results) or any(not r.ok for _, r in results):
            self._say(self._summarise(request, results) if source == "llm" else " ".join(r.text for _, r in results))
        elif results:
            self._say(results[-1][1].text)

    def _execute(self, actions: list[dict], announce: bool, skills: DroneSkills, cancel: threading.Event) -> list[tuple[str, SkillResult]]:
        results = []
        for a in actions:
            if cancel.is_set():
                break
            tool, args = a["tool"], a.get("args") or {}
            if tool == "describe" and self.vision_ok:
                r = self._describe_with_vision(skills)
            else:
                r = skills.call(tool, args)
            self._say(f"[{tool}] {r.text}", role="system", speak=False)
            results.append((tool, r))
            if announce:
                self._say(r.text)
            if not r.ok:
                break  # don't continue a plan whose earlier step failed
        return results

    def _describe_with_vision(self, skills: DroneSkills) -> SkillResult:
        jpeg = self.session.latest_jpeg
        base = skills.describe()
        if not jpeg:
            return base
        prompt = ("This is the live camera view of a small indoor drone. In one or two sentences, describe what is "
                  "in front of it: obstacles, open space, notable objects and any people. Be factual.")
        try:
            text = self.llm.describe_image(self.a.vision_model, jpeg, prompt)
        except LLMError:
            return base
        return SkillResult(True, f"{text} (Detector: {base.text})", observation=True)
