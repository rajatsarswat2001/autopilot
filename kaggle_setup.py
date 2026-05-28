"""
kaggle_setup.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bulletproof environment setup for Kaggle T4 GPU.

Run this ONCE per Kaggle session:
    !python /kaggle/working/autopilot/kaggle_setup.py

Install order (each step is independent — one failure won't block others):
  0. Pin numpy==1.26.4 (prevents PyTorch binary incompatibility)
  1. System packages (ffmpeg, libsndfile)
  2a. Core utilities (structlog, requests, pillow, etc.)
  2b. Core APIs (groq, openai, tavily, etc.)
  2c. LangChain ecosystem (pinned with upper bounds to avoid resolver hell)
  2d. Edge TTS + nest_asyncio
  3. GPU packages (diffusers>=0.33.0 force-reinstalled to overwrite Kaggle's 0.29.0)
  4. Chatterbox TTS (with --no-deps to prevent transformers downgrade)
  5. Force-reinstall numpy==1.26.4 + scipy (post-chatterbox lock)
  6. Output directories
  7. Import verification

Known-good version matrix (tested on Kaggle T4, 2026-05):
  numpy          1.26.4
  torch          2.x (pre-installed on Kaggle, do not reinstall)
  transformers   4.44.2
  diffusers      >=0.33.0 (WanPipeline)
  accelerate     0.34.2
  langchain      >=0.3.0,<1.0.0
  langgraph      >=0.3.0,<1.0.0
  chatterbox-tts latest (MIT, Resemble AI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations
import subprocess, sys, os, importlib, time

# Pass this env var to all subprocesses so chatterbox loads correctly
_ENV = {**os.environ, "TRANSFORMERS_ATTN_IMPLEMENTATION": "eager"}

PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "autopilot_pipeline")

_results: list[tuple[str, bool, str]] = []


def _run(label: str, args: list[str], critical: bool = True) -> bool:
    """Run a subprocess with merged env, log result, return success."""
    t0 = time.time()
    r  = subprocess.run(args, capture_output=True, text=True, env=_ENV)
    ok = r.returncode == 0
    elapsed = time.time() - t0
    icon = "✅" if ok else ("❌" if critical else "⚠️ ")
    msg  = r.stdout.strip()[-200:] if ok else r.stderr.strip()[-300:]
    print(f"  {icon}  {label}  ({elapsed:.0f}s)")
    if not ok:
        # Show only the key error line to keep output readable
        for line in msg.splitlines():
            if "ERROR" in line or "Conflict" in line or "conflict" in line:
                print(f"       {line.strip()[:120]}")
                break
        else:
            print(f"       {msg[:120]}")
    _results.append((label, ok, msg))
    return ok


def _pip(*pkgs: str, flags: list[str] | None = None) -> list[str]:
    """Build a pip install command (no -U by default — use explicit flags)."""
    base = [sys.executable, "-m", "pip", "install", "-q"]
    if flags:
        base += flags
    return base + list(pkgs)


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 62)
print("AutoPilot — Kaggle T4 Environment Setup")
print("═" * 62)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — numpy pin FIRST (before anything else)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[0/7] Pinning numpy==1.26.4 (prevents PyTorch binary mismatch)...")
_run("numpy==1.26.4 (initial pin)", _pip("numpy==1.26.4", "--force-reinstall"))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — System packages
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/7] System packages (ffmpeg, libsndfile)...")
subprocess.run(
    ["apt-get", "install", "-y", "-q",
     "ffmpeg", "libsndfile1", "libportaudio2", "libasound2-dev"],
    capture_output=True, env=_ENV
)
r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
ffmpeg_ver = r.stdout.split("\n")[0] if r.returncode == 0 else "NOT FOUND"
print(f"  ✅  {ffmpeg_ver}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Core pipeline packages (split into small groups to avoid resolver conflicts)
# Each group is installed independently so one failure doesn't block others.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] Core pipeline packages (split install)...")

# 2a. Utilities — no complex dependency conflicts
_run("utilities", _pip(
    "numpy==1.26.4",
    "structlog>=24.4.0",
    "python-dotenv>=1.0.1",
    "requests>=2.32.3",
    "pillow>=10.4.0",
    "pyyaml>=6.0.2",
    "httpx>=0.27.0",
    "pydantic>=2.7.0,<3.0.0",
    "nest_asyncio>=1.6.0",
))

# 2b. API clients — independent of langchain
_run("API clients", _pip(
    "groq>=0.9.0",
    "openai>=1.35.0",
    "tavily-python>=0.5.0",
    "pytrends>=4.9.2",
    "google-api-python-client>=2.130.0",
    "google-auth-oauthlib>=1.2.0",
))

# 2c. TTS runtime (edge-tts is small, separate to avoid conflicts)
_run("edge-tts", _pip("edge-tts>=6.1.0"))

# 2d. LangChain ecosystem — strict upper bounds prevent resolver from choosing 1.x
# langchain 1.x requires different pydantic/langgraph versions than 0.3.x ecosystem
_run("langchain-core", _pip(
    "langchain-core>=0.3.0,<0.4.0",
))
_run("langchain", _pip(
    "langchain>=0.3.0,<0.4.0",
    "langchain-community>=0.3.0,<0.4.0",
))
_run("langgraph", _pip("langgraph>=0.3.0,<1.0.0"))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — GPU packages
# Force-reinstall diffusers to overwrite Kaggle's pre-installed 0.29.0
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/7] GPU packages (force-reinstalling diffusers to get >=0.33.0)...")

# First install accelerate/sentencepiece/imageio (no conflicts)
_run("accelerate + imageio", _pip(
    "numpy==1.26.4",
    "accelerate>=0.34.0",
    "sentencepiece>=0.2.0",
    "safetensors>=0.4.5",
    "imageio>=2.34.0",
    "imageio-ffmpeg>=0.5.1",
))

# Force-reinstall diffusers specifically to upgrade from 0.29 → 0.33+
_run("diffusers>=0.33.0 (force-reinstall)", _pip(
    "diffusers>=0.33.0",
    flags=["--force-reinstall", "--no-deps"],
))
# Now install diffusers WITH deps (without force-reinstall) to fill any gaps
_run("diffusers deps", _pip("diffusers>=0.33.0"))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Chatterbox TTS
# Install WITHOUT --no-deps first for completeness, but we'll repin numpy after
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] Chatterbox TTS (MIT, best free voice)...")
_run("chatterbox-tts", _pip("chatterbox-tts"), critical=False)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Force-reinstall numpy + scipy AFTER chatterbox
# Chatterbox may bump numpy to 2.x. Also re-pin transformers to our tested version.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/7] Locking numpy==1.26.4 & scipy, re-pinning transformers...")

# Repin transformers to version that works with chatterbox
_run("transformers==4.44.2 (repin)", _pip(
    "transformers==4.44.2",
    flags=["--force-reinstall", "--no-deps"],
))

# Lock numpy+scipy together (binary compatibility)
_run("numpy & scipy (force reinstall)",
     _pip("numpy==1.26.4", "scipy>=1.12.0", flags=["--force-reinstall"]))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Output directories
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/7] Creating output directories...")
DIRS = [
    "outputs/video", "outputs/audio", "outputs/visual",
    "outputs/video/scratch", "data/clip_cache", "data/assets/music",
]
for d in DIRS:
    path = os.path.join(PIPELINE_DIR, d)
    os.makedirs(path, exist_ok=True)
print(f"  ✅  {len(DIRS)} directories created")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Verification
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/7] Verifying imports...")

def _check_wan(m):
    if not hasattr(m, "WanPipeline"):
        raise ImportError("WanPipeline not found — diffusers upgrade may not have taken effect. Restart kernel!")
    return f"✅ WanPipeline OK  (diffusers {m.__version__})"

def _check_scipy_ufunc(m):
    m.sph_legendre_p(0, 0, 0)
    return "✅ sph_legendre_p functional"

def _check_chatterbox(m):
    # Just importing the module is enough — full load happens in Cell 2
    return "✅ installed"

CHECK_MODULES = [
    ("numpy",         "numpy",          lambda m: m.__version__),
    ("scipy ufuncs",  "scipy.special",  _check_scipy_ufunc),
    ("torch",         "torch",          lambda m: (
        m.__version__
        + f" | CUDA: {m.cuda.is_available()}"
        + (f" | GPU: {m.cuda.get_device_name(0)}" if m.cuda.is_available() else "")
    )),
    ("diffusers",     "diffusers",      _check_wan),
    ("transformers",  "transformers",   lambda m: m.__version__),
    ("accelerate",    "accelerate",     lambda m: m.__version__),
    ("chatterbox",    "chatterbox.tts", _check_chatterbox),
    ("groq",          "groq",           lambda m: m.__version__),
    ("langchain",     "langchain",      lambda m: m.__version__),
    ("langgraph",     "langgraph",      lambda m: getattr(m, "__version__", "✅")),
    ("edge_tts",      "edge_tts",       lambda m: getattr(m, "__version__", "✅")),
    ("structlog",     "structlog",      lambda m: m.__version__),
]

pass_count = 0
fail_count = 0
numpy_ok   = False

for label, mod_name, get_ver in CHECK_MODULES:
    try:
        m   = importlib.import_module(mod_name)
        ver = get_ver(m)
        print(f"  ✅  {label:<18} {ver}")
        if label == "numpy":
            major = int(m.__version__.split(".")[0])
            if major >= 2:
                print(f"       ⚠️  numpy {m.__version__} is 2.x — RESTART KERNEL then re-run Cell 1!")
                fail_count += 1
                continue
            numpy_ok = True
        pass_count += 1
    except (ImportError, ValueError, AttributeError) as e:
        es = str(e)
        if "dtype size" in es or "numpy" in es.lower():
            print(f"  ⚠️   {label:<18} numpy binary mismatch — restart kernel!")
        else:
            print(f"  ❌  {label:<18} {es[:80]}")
        fail_count += 1

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 62)
total = pass_count + fail_count
status = "PASSED ✅" if fail_count == 0 else f"COMPLETED WITH WARNINGS ⚠️  ({fail_count}/{total} failed)"
print(f"SETUP {status}")

if not numpy_ok:
    print("\n⚠️  NUMPY NOT CONFIRMED GOOD")
    print("   → Restart kernel, then re-run Cell 1")
elif fail_count == 0:
    print("\n✅ All checks passed! Proceed to Cell 2 to pre-load GPU models.")
else:
    print(f"\n⚠️  {fail_count} checks failed — see above.")
    print("   If diffusers or chatterbox failed: restart kernel and re-run Cell 1.")
    print("   The kernel restart ensures newly installed packages are loaded fresh.")

print("═" * 62)
