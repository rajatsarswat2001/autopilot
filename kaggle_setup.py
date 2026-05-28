"""
kaggle_setup.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bulletproof environment setup for Kaggle T4 GPU.

Run this ONCE per Kaggle session:
    !python /kaggle/working/autopilot/kaggle_setup.py

What it does (in exact safe order):
  1. Pin numpy 1.26.4 FIRST (prevents PyTorch binary incompatibility)
  2. System packages (ffmpeg, libsndfile)
  3. Core pipeline packages (pinned versions)
  4. GPU packages: transformers, diffusers, accelerate (pinned)
  5. Chatterbox TTS (MIT)
  6. Force-reinstall numpy 1.26.4 AGAIN (chatterbox may try to bump it)
  7. Verify every critical import works
  8. Create output directories
  9. Print pass/fail summary

Why pinned versions instead of >= ?
  Kaggle images update weekly. A new transformers release can break
  chatterbox or diffusers overnight. Pinned = reproducible = no surprises.

Known-good version matrix (tested on Kaggle T4, 2026-05):
  numpy          1.26.4
  torch          2.x (pre-installed on Kaggle, do not reinstall)
  transformers   4.44.2
  diffusers      0.31.0
  accelerate     0.34.2
  chatterbox-tts latest (MIT, Resemble AI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations
import subprocess, sys, os, importlib, time
os.environ['TRANSFORMERS_ATTN_IMPLEMENTATION'] = 'eager'

PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "autopilot_pipeline")
LOG_PATH     = os.path.join(PIPELINE_DIR, "setup_log.txt")

_results: list[tuple[str, bool, str]] = []


def _run(label: str, args: list[str], critical: bool = True) -> bool:
    """Run a subprocess, log result, return success."""
    t0 = time.time()
    r  = subprocess.run(args, capture_output=True, text=True)
    ok = r.returncode == 0
    elapsed = time.time() - t0
    icon = "✅" if ok else ("❌" if critical else "⚠️ ")
    msg  = r.stdout.strip()[-200:] if ok else r.stderr.strip()[-300:]
    print(f"  {icon}  {label}  ({elapsed:.0f}s)")
    if not ok:
        print(f"       {msg}")
    _results.append((label, ok, msg))
    return ok


def _pip(*pkgs: str, flags: list[str] | None = None) -> list[str]:
    base = [sys.executable, "-m", "pip", "install", "-q", "-U"]
    if flags:
        base += flags
    return base + list(pkgs)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — numpy pin (MUST be first — before anything else)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 62)
print("AutoPilot — Kaggle T4 Environment Setup")
print("═" * 62)

print("\n[0/7] Pinning numpy==1.26.4 (prevents PyTorch binary mismatch)...")
_run("numpy==1.26.4 (initial pin)",
     _pip("numpy==1.26.4"))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — System packages
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/7] System packages (ffmpeg, libsndfile)...")
subprocess.run(
    ["apt-get", "install", "-y", "-q",
     "ffmpeg", "libsndfile1", "libportaudio2", "libasound2-dev"],
    capture_output=True
)
r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
ffmpeg_ver = r.stdout.split("\n")[0] if r.returncode == 0 else "NOT FOUND"
print(f"  ✅  {ffmpeg_ver}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Core pipeline packages (pinned)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] Core pipeline packages...")
CORE_PKGS = [
    "numpy==1.26.4",                 # keep pinned in resolver
    "structlog==24.4.0",
    "python-dotenv==1.0.1",
    "edge-tts==6.1.12",
    "nest_asyncio==1.6.0",
    "openai==1.35.0",
    "groq==0.9.0",
    "requests==2.32.3",
    "pillow==10.4.0",
    "pydantic==2.7.4",
    "httpx==0.27.0",
    "pyyaml==6.0.2",
    "langgraph>=0.3.0",
    "langchain>=0.3.0",
    "langchain-community>=0.3.0",
    "tavily-python==0.5.0",
    "pytrends==4.9.2",
    "google-api-python-client==2.137.0",
    "google-auth-oauthlib==1.2.1",
]
_run("core packages", _pip(*CORE_PKGS))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — GPU packages: transformers, diffusers, accelerate (pinned)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/7] GPU packages (transformers, diffusers, accelerate)...")
GPU_PKGS = [
    "numpy==1.26.4",                 # prevent resolver from bumping
    "transformers==4.44.2",          # pinned — works with Wan2.1 + chatterbox
    "diffusers>=0.33.0",             # Wan2.1 WanPipeline support
    "accelerate==0.34.2",            # pinned — model CPU offloading
    "sentencepiece==0.2.0",          # Wan2.1 tokenizer
    "safetensors==0.4.5",            # model loading
    "imageio==2.35.1",
    "imageio-ffmpeg==0.5.1",
]
_run("GPU packages", _pip(*GPU_PKGS))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Chatterbox TTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] Chatterbox TTS (MIT, best free voice)...")
_run("chatterbox-tts", _pip("chatterbox-tts"), critical=False)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Force-reinstall numpy + scipy AFTER chatterbox
# Chatterbox's dependency resolver may have bumped numpy to 2.x, which breaks scipy.
# Reinstalling them together guarantees binary compatibility.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/7] Force-reinstalling numpy==1.26.4 & scipy>=1.12.0...")
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
# STEP 7 — Verification (import every critical module)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/7] Verifying imports...")

def _check_wan(m):
    if not hasattr(m, "WanPipeline"):
        raise ImportError("WanPipeline missing from diffusers! Ensure diffusers>=0.33.0 is installed.")
    return "✅ WanPipeline available"

def _check_scipy_ufunc(m):
    # Verify ufuncs can be executed without ValueError
    res = m.sph_legendre_p(0, 0, 0)
    return "✅ sph_legendre_p functional"

CHECK_MODULES = [
    ("numpy",         "numpy",          lambda m: m.__version__),
    ("scipy ufuncs",  "scipy.special",  _check_scipy_ufunc),
    ("torch",         "torch",          lambda m: m.__version__
                                        + f" | CUDA: {m.cuda.is_available()}"
                                        + (f" | GPU: {m.cuda.get_device_name(0)}" if m.cuda.is_available() else "")),
    ("diffusers",     "diffusers",      lambda m: m.__version__),
    ("WanPipeline",   "diffusers",      _check_wan),
    ("transformers",  "transformers",   lambda m: m.__version__),
    ("accelerate",    "accelerate",     lambda m: m.__version__),
    ("chatterbox",    "chatterbox.tts", lambda m: "✅ installed"),
    ("groq",          "groq",           lambda m: m.__version__),
    ("langgraph",     "langgraph",      lambda m: getattr(m, "__version__", "✅ installed")),
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
                print(f"       ⚠️  numpy {m.__version__} is 2.x — should be 1.26.4!")
                print("       Run: !pip install --force-reinstall numpy==1.26.4")
                print("       Then restart kernel and run this script again.")
                fail_count += 1
                continue
            numpy_ok = True
        pass_count += 1
    except (ImportError, ValueError) as e:
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
print(f"SETUP {'PASSED ✅' if fail_count == 0 else 'COMPLETED WITH WARNINGS ⚠️'}")
print(f"  {pass_count} checks passed  |  {fail_count} failed")

if not numpy_ok:
    print("\n⚠️  NUMPY ISSUE DETECTED")
    print("   1. Restart the Kaggle kernel (Run → Restart & Clear Output)")
    print("   2. Run this cell again")
    print("   3. If it persists: Factory Reset → run !pip install numpy==1.26.4 first")
else:
    print("\n✅ Run Cell 2 to set API keys, then Cell 3 to generate a video")

print("═" * 62)
