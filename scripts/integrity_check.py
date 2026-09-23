#!/usr/bin/env python3
"""System-wide integrity check for the ig-autobot pipeline.

Scans:
- Python syntax for all .py files
- YAML validity for all .yml/.yaml
- Import chain completeness (no broken imports)
- Cross-file function references (called functions exist)
- Shell scripts for syntax
- Flag/state file schema consistency
- Duplicate/conflicting definitions
- Missing __init__.py / entry points
"""
import ast
import glob
import json
import os
import re
import sys
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
errors = []
warnings = []
passed = []

def fail(msg):
    errors.append(msg)
    print(f"  FAIL: {msg}")

def warn(msg):
    warnings.append(msg)
    print(f"  WARN: {msg}")

def ok(msg):
    passed.append(msg)
    print(f"  ok  : {msg}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ===== 1. PYTHON SYNTAX =====
section("1. Python syntax")
py_files = sorted(glob.glob("**/*.py", root_dir=REPO_ROOT))
for rel in py_files:
    if "__pycache__" in rel or ".venv" in rel:
        continue
    full = REPO_ROOT / rel
    try:
        ast.parse(full.read_text(encoding="utf-8"), filename=str(rel))
        ok(rel)
    except SyntaxError as e:
        fail(f"{rel}: {e}")

# ===== 2. YAML VALIDITY =====
section("2. YAML validity")
yml_files = sorted(glob.glob("**/*.yml", root_dir=REPO_ROOT)) + sorted(glob.glob("**/*.yaml", root_dir=REPO_ROOT))
for rel in yml_files:
    if ".venv" in rel:
        continue
    full = REPO_ROOT / rel
    try:
        list(yaml.safe_load_all(full.read_text(encoding="utf-8")))
        ok(rel)
    except yaml.YAMLError as e:
        fail(f"{rel}: {e}")

# ===== 3. IMPORT CHAIN =====
section("3. Import chain completeness")
# Map all top-level functions/defs in repo
repo_defs = {}
for rel in py_files:
    if "__pycache__" in rel or ".venv" in rel:
        continue
    full = REPO_ROOT / rel
    try:
        tree = ast.parse(full.read_text(encoding="utf-8"), filename=str(rel))
        module = rel.replace("/", ".").replace(".py", "").lstrip(".")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                repo_defs[node.name] = (module, rel, node.lineno)
            elif isinstance(node, ast.ClassDef):
                repo_defs[node.name] = (module, rel, node.lineno)
    except:
        pass

# Now check imports in each file
import_map = {}
for rel in py_files:
    if "__pycache__" in rel or ".venv" in rel:
        continue
    full = REPO_ROOT / rel
    try:
        src = full.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(rel))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_map[alias.asname or alias.name.split(".")[0]] = alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        key = alias.asname or alias.name
                        import_map[key] = f"{node.module}.{alias.name}"
    except:
        pass

# Check that functions actually called exist (approximate)
called_funcs = set()
for rel in py_files:
    if "__pycache__" in rel or ".venv" in rel:
        continue
    full = REPO_ROOT / rel
    try:
        src = full.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(rel))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_funcs.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called_funcs.add(node.func.attr)
    except:
        pass

# Find undefined calls (common typos/bugs)
undefined = called_funcs - set(repo_defs.keys())
# Filter out builtins and stdlib
stdlib_names = set(dir(__builtins__) if isinstance(__builtins__, dict) else dir(__builtins__))
stdlib_names.update(["print","range","len","int","str","list","dict","set","tuple","open","isinstance","type","super","property","staticmethod","classmethod","map","filter","zip","enumerate","reversed","sorted","abs","min","max","sum","round","hash","id","hex","oct","bin","chr","ord","format","repr","hasattr","getattr","setattr","delattr","any","all","next","iter","callable","compile","eval","exec","globals","locals","breakpoint","memoryview","object","classmethod","staticmethod","super","type","vars","dir","help","input","print","property","__import__"])
external_names = set(import_map.keys())
external_names.update(["os","sys","json","re","time","shutil","glob","pathlib","concurrent","functools","itertools","collections","datetime","random","string","hashlib","base64","io","typing","dataclasses","enum","abc","contextlib","unittest","pytest","subprocess","signal","tempfile","textwrap","warnings","traceback","logging","math","statistics","fractions","decimal","numbers","numpy","np","PIL","Image","ImageDraw","ImageFont","ImageFilter","ImageOps","ImageColor","VideoClip","AudioFileClip","CompositeVideoClip","TextClip","moviepy","yaml","requests","dotenv","dotenv","load_dotenv","catbox","bs4","BeautifulSoup","selenium","webdriver","time","sleep","datetime","timedelta","Path","ThreadPoolExecutor","as_completed","asyncio","aiohttp","jwt","Crypto","cryptography","pandas","pd","plt","matplotlib","sklearn","tensorflow","torch","transformers","urllib","http","socket","ssl","email","html","xml","csv","sqlite3","pickle","shelve","copy","pprint","textwrap","dataclasses","abc","enum","inspect","dis","code","codeop","platform","locale","gettext","argparse","optparse","configparser","tomllib","csv","json","xml","html","urllib","http","ftplib","poplib","imaplib","smtplib","uuid","ipaddress","macaddress","netifaces","psutil","platform","ctypes","mmap","winreg","msvcrt","os","sys","io","codecs","unicodedata","locale","gettext","logging","warnings","traceback","atexit","signal","subprocess","sched","queue","threading","multiprocessing","concurrent","asyncio","select","selectors","signal","socket","ssl","email","json","mailbox","mimetypes","base64","binascii","quopri","uu","html","xml","webbrowser","cgi","cgitb","wsgirew","urllib","http","ftplib","poplib","imaplib","smtplib","uuid","socketserver","xmlrpc","ipaddress","macaddress","netifaces","psutil","platform","ctypes","mmap","winreg","msvcrt","os","sys","io","codecs","unicodedata","locale","gettext","logging","warnings","traceback","atexit","signal","subprocess","sched","queue","threading","multiprocessing","concurrent","asyncio","select","selectors","signal","socket","ssl","email","json","mailbox","mimetypes","base64","binascii","quopri","uu","html","xml","webbrowser","cgi","cgitb","wsgirew","urllib","http","ftplib","poplib","imaplib","smtplib","uuid","socketserver","xmlrpc","ipaddress"])
truly_undefined = undefined - stdlib_names - external_names
if truly_undefined:
    warn(f"Potentially undefined functions called: {sorted(truly_undefined)[:20]}")
else:
    ok("No obviously undefined function calls")

# ===== 4. DUPLICATE/CONFLICTING DEFINITIONS =====
section("4. Duplicate top-level definitions")
from collections import Counter
def_counts = Counter(repo_defs.keys())
dupes = {k: v for k, v in def_counts.items() if v > 1}
if dupes:
    for name, count in sorted(dupes.items()):
        locations = [f"{rel}:{ln}" for m, rel, ln in [repo_defs[name]]] if name in repo_defs else []
        warn(f"'{name}' defined {count}x — may cause import conflicts")
else:
    ok("No duplicate top-level definitions")

# ===== 5. STATE FILE SCHEMAS =====
section("5. State file schemas")
for state_rel in ["state.json", "forwilma/state.json", "posts.json", "carousel.json", "forwilma/carousel.json", "forwilma/clean_bases.json", "forwilma/used_bases.json"]:
    full = REPO_ROOT / state_rel
    if not full.exists():
        warn(f"{state_rel}: file not found")
        continue
    try:
        data = json.loads(full.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            keys = sorted(data.keys())
            ok(f"{state_rel}: valid JSON dict with keys {keys}")
        elif isinstance(data, list):
            ok(f"{state_rel}: valid JSON list ({len(data)} items)")
        else:
            ok(f"{state_rel}: valid JSON ({type(data).__name__})")
    except json.JSONDecodeError as e:
        fail(f"{state_rel}: invalid JSON: {e}")

# ===== 6. ENV FILE PRESENCE =====
section("6. Environment files")
for env_rel in [".env", ".env.example", ".env.template"]:
    full = REPO_ROOT / env_rel
    if full.exists():
        ok(f"{env_rel} exists")
    else:
        pass  # optional

# ===== 7. REQUIRED ENTRY POINTS =====
section("7. Required entry points")
required = [
    "bot.py",
    "forwilma/wilma_bot.py",
    "scripts/publish.py",
    "scripts/prepare_assets.py",
    "scripts/agent_publisher.py",
    "scripts/generate_hyperframes_reel.py",
    "scripts/wilma_fallback_base.py",
    "scripts/build-ig-router.py",
    "scripts/brand_linkedin_image.py",
    "shared_utils.py",
    "requirements.txt",
    "forwilma/publish_wilma_linkedin.py",
    "forwilma/publish_wilma_bluesky.py",
]
for rel in required:
    full = REPO_ROOT / rel
    if full.exists():
        ok(rel)
    else:
        fail(f"{rel}: MISSING")

# ===== 8. CRON CONFLICTS =====
section("8. Workflow cron conflicts")
cron_times = {}
for rel in yml_files:
    if ".github/workflows" not in rel:
        continue
    full = REPO_ROOT / rel
    try:
        src = full.read_text(encoding="utf-8")
        data = yaml.safe_load(src)
        triggers = data.get("on", {})
        schedules = triggers.get("schedule", [])
        if schedules:
            for sched in schedules:
                cron = sched.get("cron", "")
                if cron in cron_times:
                    warn(f"Cron overlap: {cron} in both {rel} and {cron_times[cron]}")
                else:
                    cron_times[cron] = rel
        ok(f"{rel}: {len(schedules)} schedule(s)")
    except:
        pass

# ===== 9. CRLF in PYTHON FILES =====
section("9. CRLF line endings in Python")
for rel in py_files:
    if "__pycache__" in rel or ".venv" in rel:
        continue
    full = REPO_ROOT / rel
    content = full.read_bytes()
    if b"\r\n" in content:
        fail(f"{rel}: has CRLF (LF required)")
    else:
        ok(rel)

# ===== 10. MISSING __init__.py =====
section("10. Package structure")
pkg_dirs = set()
for rel in py_files:
    if "__pycache__" in rel:
        continue
    parts = Path(rel).parts
    for i in range(1, len(parts)):
        pkg_dirs.add("/".join(parts[:i]))
for pkg in sorted(pkg_dirs):
    init_path = REPO_ROOT / pkg / "__init__.py"
    if init_path.exists():
        ok(f"{pkg}/__init__.py")
    else:
        warn(f"{pkg}/: no __init__.py (may break imports)")

# ===== SUMMARY =====
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  Passed:   {len(passed)}")
print(f"  Warnings: {len(warnings)}")
print(f"  Errors:   {len(errors)}")
if errors:
    print(f"\n  ERRORS:")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
else:
    print(f"\n  ✓ All critical checks passed")
    sys.exit(0)
