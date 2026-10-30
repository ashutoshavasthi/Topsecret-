import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from skymind.app.server import SkyMindApp
from skymind.assistant import rules
from skymind.assistant.agent import DroneAssistant
from skymind.assistant.llm import OllamaClient
from skymind.assistant.skills import DroneSkills
from skymind.config import load_config
from skymind.live import SimLiveSession

PLACES = ["home", "door", "table"]


@pytest.mark.parametrize(
    "text,tool,args",
    [
        ("Hey SkyMind, take off", "takeoff", {}),
        ("please land", "land", {}),
        ("come back", "return_home", {}),
        ("STOP", "hold", {}),
        ("go to the door", "go_to", {"place": "door"}),
        ("fly forward 2 metres", "move", {"direction": "forward", "meters": 2.0}),
        ("turn right 45 degrees", "turn", {"degrees": -45.0}),
        ("look around", "scan", {}),
        ("what do you see?", "describe", {}),
        ("find the red ball", "find", {"object": "red ball"}),
        ("go up to the chair", "approach", {"object": "chair"}),
        ("go up a bit", "climb", {"meters": 0.5}),
        ("how's the battery", "status", {}),
    ],
)
def test_rules(text, tool, args):
    plan = rules.parse(text, PLACES)
    assert plan is not None and plan.actions[0]["tool"] == tool
    assert plan.actions[0].get("args", {}) == args


def test_rules_leave_open_questions_to_the_model():
    assert rules.parse("is it safe to fly through that gap?", PLACES) is None
    assert rules.parse("fly to the moon", PLACES) is None


def test_object_matching():
    assert DroneSkills._matches("red ball", "red_marker")
    assert DroneSkills._matches("marker", "yellow_marker")
    assert DroneSkills._matches("person", "person")
    assert not DroneSkills._matches("chair", "red_marker")
    assert not DroneSkills._matches("yellow thing", "red_marker")


@pytest.fixture
def session(tmp_path):
    cfg = load_config("config/nano_room.yaml", {"assistant": {"use_llm": False}})
    s = SimLiveSession(cfg, tmp_path, speed=0)
    s.start()
    yield s
    s.stop()


def test_question_does_not_cancel_flight_but_stop_does(session):
    a = DroneAssistant(session, session.cfg)
    a.handle("take off", wait=True)
    assert session.status()["mode"] == "hold"
    a.handle("go to the corridor")
    assert session.wait_for(lambda st: st["mode"] == "mission", 5)
    a.handle("how's the battery?")  # read-only: answered while flying
    assert session.wait_for(lambda st: st["mode"] == "hold", 40)
    a.wait_idle()
    texts = [m["text"] for m in a.messages_after(0)]
    assert any("Battery" in t for t in texts) and any("arrived at the corridor" in t for t in texts)
    a.handle("go to the home")
    assert session.wait_for(lambda st: st["mode"] == "mission", 5)
    a.handle("stop")
    assert session.wait_for(lambda st: st["mode"] == "hold", 1.0)  # applied on the very next tick
    a.wait_idle()
    assert session.collisions == 0


def test_skills_refuse_outside_geofence_and_unknown_places(session):
    sk = DroneSkills(session)
    assert not sk.call("go_to_xy", {"x": 50.0, "y": 0.0}).ok
    r = sk.call("go_to", {"place": "kitchen"})
    assert not r.ok and "Known places" in r.text
    assert not sk.call("nonexistent").ok


class _FakeOllama(BaseHTTPRequestHandler):
    calls: list = []

    def log_message(self, *a):
        pass

    def _reply(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._reply({"models": [{"name": "qwen2.5:3b"}, {"name": "moondream:latest"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeOllama.calls.append(req)
        msgs = req["messages"]
        if any("images" in m for m in msgs):
            content = "A grey partition wall with an open doorway; nobody is in view."
        elif "actions" in json.dumps(req.get("format", {})):
            content = json.dumps({"reply": "On my way to the door, then I'll have a look.",
                                  "actions": [{"tool": "go_to", "args": {"place": "door"}}, {"tool": "describe"},
                                              {"tool": "launch_missiles", "args": {}}]})
        else:
            content = json.dumps({"reply": "I'm at the door. I see a doorway and nobody there."})
        self._reply({"message": {"role": "assistant", "content": content}})


def test_llm_plans_vision_and_summary(session):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cfg = session.cfg
        cfg.assistant.use_llm = True
        llm = OllamaClient(f"http://127.0.0.1:{srv.server_address[1]}", timeout=5)
        a = DroneAssistant(session, cfg, llm=llm)
        assert a.llm_ok and a.vision_ok
        a.handle("take off", wait=True)  # safety-critical commands skip the model
        n_calls = len(_FakeOllama.calls)
        a.handle("check whether anyone is at the door", wait=True)
        texts = [m["text"] for m in a.messages_after(0)]
        assert "On my way to the door, then I'll have a look." in texts
        assert any("nobody is in view" in t for t in texts)  # vision model was used
        assert texts[-1] == "I'm at the door. I see a doorway and nobody there."  # grounded summary
        assert not any("launch_missiles" in t for t in texts)  # unknown tools are dropped
        assert len(_FakeOllama.calls) == n_calls + 3  # plan, image, summary
        assert "door" in json.dumps(_FakeOllama.calls[n_calls]["messages"][0])  # places are in the prompt
    finally:
        srv.shutdown()


def test_missing_model_falls_back_to_rules(session):
    cfg = session.cfg
    cfg.assistant.use_llm = True
    a = DroneAssistant(session, cfg, llm=OllamaClient("http://127.0.0.1:9", timeout=1))
    assert not a.llm_ok
    a.handle("take off", wait=True)
    assert session.status()["mode"] == "hold"


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read()


def test_app_server_routes_and_token(session):
    a = DroneAssistant(session, session.cfg)
    app = SkyMindApp(session, a, host="0.0.0.0", port=0)
    httpd = app.serve()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(base + "/api/status")
        assert e.value.code == 403  # reachable from the network -> token required
        t = f"?t={app.token}"
        code, body = _get(base + "/api/status" + t)
        st = json.loads(body)
        assert code == 200 and st["assistant"]["name"] == "SkyMind" and "door" in st["places"]
        assert _get(base + "/" + t)[1].startswith(b"<!doctype html>")
        assert _get(base + "/map.png" + t)[1][:4] == b"\x89PNG"
        req = urllib.request.Request(base + "/api/chat" + t, data=json.dumps({"text": "take off"}).encode(),
                                     headers={"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req, timeout=5).read())["ok"]
        assert session.wait_for(lambda s: s["mode"] == "hold", 20)
        msgs = json.loads(_get(base + "/api/messages" + t + "&after=0")[1])
        assert any(m["role"] == "assistant" and m["speak"] for m in msgs)
        assert _get(base + "/frame.jpg" + t)[1][:2] == b"\xff\xd8"
    finally:
        httpd.shutdown()
