"""Hermes plugin: lazy router for ig-autobot social-auto-agency agents."""
import json, math, re
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "data" / "agents.json"
_AGENTS = None
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9+.#_-]*", re.I)

def _load_agents():
    global _AGENTS
    if _AGENTS is None:
        _AGENTS = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    return _AGENTS

def _tokens(text):
    return {t.lower() for t in _WORD_RE.findall(text or "")}

def _lookup(identifier):
    n = (identifier or "").strip().lower()
    if not n: return None
    s = re.sub(r"[^a-z0-9]+", "-", n).strip("-")
    for a in _load_agents():
        if a["slug"] == s or a["name"].lower() == n: return a
    return None

def _score(a, qt, qtext):
    fields = [a.get("name",""), a.get("description",""), a.get("division",""), a.get("vibe",""), a.get("body","")[:8000]]
    hay = chr(10).join(fields).lower()
    tokens = _tokens(hay)
    overlap = qt & tokens
    score = float(len(overlap))
    if qtext and qtext in hay: score += 5.0
    for t in qt:
        if t in a.get("name","").lower(): score += 3.0
        if t in a.get("description","").lower(): score += 1.5
    return score + (1.0 / math.sqrt(max(len(tokens),1))) if score > 0 else 0.0

def _summary(a, score=None):
    item = {"slug":a["slug"],"name":a["name"],"division":a.get("division",""),"description":a.get("description",""),"vibe":a.get("vibe",""),"source_path":a.get("source_path","")}
    if score: item["score"] = round(score, 3)
    return item

def _prompt(a, task=""):
    tb = chr(10)*2 + "## User task" + chr(10) + task.strip() + chr(10) if task.strip() else ""
    return f"Use the following specialist context for this turn." + chr(10)*2 + f"# {a['name']} ({a['slug']})" + chr(10)*2 + f"Division: {a.get('division','')}" + chr(10) + f"Description: {a.get('description','')}" + chr(10) + f"Source: {a.get('source_path','')}" + chr(10) + tb + chr(10) + "## Specialist instructions" + chr(10) + a.get("body","")

def _json(p): return json.dumps(p, ensure_ascii=False, indent=2)

def register(ctx):
    def search(args, **kw):
        q = str(args.get("query","")).strip()
        if not q: return _json({"success":False,"error":"query is required"})
        div = str(args.get("division","")).strip().lower()
        lim = min(max(int(args.get("limit",8)),1),25)
        qt = _tokens(q); qt_text = q.lower()
        matches = []
        for a in _load_agents():
            if div and a.get("division","").lower() != div: continue
            s = _score(a, qt, qt_text)
            if s > 0: matches.append((s,a))
        matches.sort(key=lambda x: (-x[0], x[1]["division"], x[1]["slug"]))
        return _json({"success":True,"query":q,"count":len(matches),"results":[_summary(a,s) for s,a in matches[:lim]]})

    def read(args, **kw):
        ident = str(args.get("agent") or args.get("slug") or "").strip()
        a = _lookup(ident)
        if not a: return _json({"success":False,"error":"agent not found","agent":ident})
        p = {"success":True,"agent":_summary(a)}
        if args.get("include_body"): p["body"] = a.get("body","")
        return _json(p)

    def prompt(args, **kw):
        ident = str(args.get("agent") or args.get("slug") or "").strip()
        a = _lookup(ident)
        if not a: return _json({"success":False,"error":"agent not found","agent":ident})
        return _json({"success":True,"agent":_summary(a),"prompt":_prompt(a, str(args.get("task","")))})

    def delegate(args, **kw):
        ident = str(args.get("agent") or args.get("slug") or "").strip()
        a = _lookup(ident)
        task = str(args.get("task","")).strip()
        if not a: return _json({"success":False,"error":"agent not found","agent":ident})
        if not task: return _json({"success":False,"error":"task is required"})
        return _json({"success":True,"agent":_summary(a),"delegated":False,"warning":"stub","prompt":_prompt(a,task)})

    for name, schema, fn in [
        ("ig_auto_agency_search", {"name":"ig_auto_agency_search","description":"Search ig-autobot specialist agents.","parameters":{"type":"object","properties":{"query":{"type":"string"},"division":{"type":"string"},"limit":{"type":"integer"}},"required":["query"]}}, search),
        ("ig_auto_agency_inspect", {"name":"ig_auto_agency_inspect","description":"Read one specialist agent.","parameters":{"type":"object","properties":{"agent":{"type":"string"},"slug":{"type":"string"},"include_body":{"type":"boolean"}},"required":[]}}, read),
        ("ig_auto_agency_load", {"name":"ig_auto_agency_load","description":"Compose one specialist prompt.","parameters":{"type":"object","properties":{"agent":{"type":"string"},"slug":{"type":"string"},"task":{"type":"string"}},"required":[]}}, prompt),
        ("ig_auto_agency_delegate", {"name":"ig_auto_agency_delegate","description":"Delegate to one specialist agent.","parameters":{"type":"object","properties":{"agent":{"type":"string"},"slug":{"type":"string"},"task":{"type":"string"}},"required":["task"]}}, delegate),
    ]:
        ctx.register_tool(name=name, toolset="ig_auto_agency", schema=schema, handler=fn, description=schema["description"])
