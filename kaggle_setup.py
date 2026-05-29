"""
kaggle_setup.py
================================================================================
Bulletproof environment setup for Kaggle T4 GPU.

Run ONCE per Kaggle session:
    !python /kaggle/working/autopilot/kaggle_setup.py

Research-backed version matrix (Kaggle T4, May 2026):
  torch          2.6.0+cu124   DO NOT reinstall
  numpy          1.26.4        pinned LAST after every other install
  transformers   4.46.3        downgraded from Kaggle 5.x (chatterbox requires)
  diffusers      0.34.0        upgraded from Kaggle 0.29.0 (WanPipeline requires)
  accelerate     >=0.34.2
  langchain      >=1.0.0       1.x stack resolves cleanly on Kaggle Python 3.12
  langchain-community >=0.4.0
  langgraph      >=1.0.0
  chatterbox-tts latest        installed --no-deps to avoid transformers 5.x clash
  kokoro         >=0.9.4       Apache 2.0, 82M params, 210x RT — best free TTS
  onnx           1.16.0        pre-built wheel (avoids CMake build requirement)

Key design decisions from research:
  - diffusers: must pip uninstall FIRST, then reinstall. pip skip-upgrades 0.29.0.
  - chatterbox: --no-deps required, then manually pin transformers==4.46.3
  - langchain: pin to >=1.0.0 (NOT 0.3.x) to avoid langgraph resolver conflicts
  - numpy: MUST be the absolute final step — every package above may bump it
  - Wan2.1 model ID: Wan-AI/Wan2.1-T2V-1.3B-Diffusers (not Wan-Video/...)
================================================================================
"""
from __future__ import annotations
import subprocess, sys, os, importlib, time

# Passed to every subprocess so chatterbox/transformers use eager attention
_ENV = {**os.environ, "TRANSFORMERS_ATTN_IMPLEMENTATION": "eager"}

PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "autopilot_pipeline")

_results: list[tuple[str, bool, str]] = []


def _run(label: str, args: list[str], critical: bool = True) -> bool:
    """Run a subprocess with merged env, log pass/fail, return ok."""
    t0  = time.time()
    r   = subprocess.run(args, capture_output=True, text=True, env=_ENV)
    ok  = r.returncode == 0
    elapsed = time.time() - t0
    icon = "OK  " if ok else ("FAIL" if critical else "WARN")
    print(f"  [{icon}]  {label}  ({elapsed:.0f}s)")
    if not ok:
        for line in r.stderr.strip().splitlines():
            if "ERROR" in line or "conflict" in line.lower() or "Resolution" in line:
                print(f"         {line.strip()[:120]}")
                break
        else:
            print(f"         {r.stderr.strip()[-120:]}")
    _results.append((label, ok, r.stderr.strip()[-200:]))
    return ok


def _pip(*pkgs: str, flags: list[str] | None = None) -> list[str]:
    """Build pip install command. No -U flag by default (use explicit flags)."""
    base = [sys.executable, "-m", "pip", "install", "-q"]
    if flags:
        base += flags
    return base + list(pkgs)


# ============================================================================
print("\n" + "=" * 64)
print("AutoPilot -- Kaggle T4 Environment Setup")
print("=" * 64)

# ============================================================================
# STEP 0: Pin numpy FIRST (prevent any early bump to 2.x)
# ============================================================================
print("\n[0/7] Pre-pinning numpy==1.26.4 ...")
_run("numpy==1.26.4 (pre-pin)", _pip("numpy==1.26.4", flags=["--force-reinstall"]))

# ============================================================================
# STEP 1: System packages (including espeak-ng for Kokoro TTS)
# ============================================================================
print("\n[1/7] System packages (ffmpeg, libsndfile, espeak-ng) ...")
subprocess.run(
    ["apt-get", "install", "-y", "-q",
     "ffmpeg", "libsndfile1", "libportaudio2", "libasound2-dev", "espeak-ng"],
    capture_output=True, env=_ENV
)
r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
print(f"  [OK ]  {r.stdout.split(chr(10))[0]}")

# ============================================================================
# STEP 2: Core packages — each group installed independently
# (one group failing will NOT block others)
# ============================================================================
print("\n[2/7] Core pipeline packages (independent install groups) ...")

# 2a. Utilities
_run("utilities", _pip(
    "numpy==1.26.4",
    "structlog>=24.4.0",
    "python-dotenv>=1.0.1",
    "requests>=2.32.0",
    "pillow>=10.3.0",
    "pyyaml>=6.0.1",
    "httpx>=0.27.0",
    "pydantic>=2.7.0,<3.0.0",
    "nest_asyncio>=1.6.0",
    "ftfy>=6.1.1",      # required by transformers CLIP tokenizer (Wan2.1 dep)
))

# 2b. API clients (groq, openai, tavily — independent of langchain)
_run("API clients", _pip(
    "groq>=0.9.0",
    "openai>=1.35.0",
    "tavily-python>=0.5.0",
    "pytrends>=4.9.2",
    "google-api-python-client>=2.130.0",
    "google-auth-oauthlib>=1.2.0",
))

# 2c. Edge TTS (small, isolated)
_run("edge-tts", _pip("edge-tts>=6.1.0"))

# 2d. LangChain 1.x stack
# Research finding: 0.3.x + langgraph 1.x = ResolutionImpossible
# Solution: use 1.x stack which resolves cleanly together
_run("langchain-core 1.x", _pip("langchain-core>=1.0.0"))
_run("langchain 1.x",      _pip("langchain>=1.0.0"))
_run("langchain-community", _pip("langchain-community>=0.4.0"))
_run("langgraph 1.x",      _pip("langgraph>=1.0.0"))

# ============================================================================
# STEP 3: GPU packages — upgrade diffusers from 0.29.0 to 0.34.0
# Research finding: must pip uninstall first; otherwise pip sees 0.29 as "satisfied"
# ============================================================================
print("\n[3/7] GPU packages (upgrading diffusers 0.29 -> 0.34) ...")

# Uninstall stale diffusers so pip doesn't skip the upgrade
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "diffusers", "-y"],
    capture_output=True, env=_ENV
)

# Install diffusers 0.34.0 without deps first (keeps torch safe)
_run("diffusers==0.34.0 --no-deps",
     _pip("diffusers==0.34.0", flags=["--no-deps"]))

# Now install diffusers' own deps (excludes torch which is pre-installed)
_run("diffusers deps", _pip(
    "huggingface-hub>=0.23.0",
    "accelerate>=0.34.0",
    "safetensors>=0.4.5",
    "sentencepiece>=0.2.0",
    "imageio>=2.34.0",
    "imageio-ffmpeg>=0.5.1",
))

# ============================================================================
# STEP 4: Chatterbox TTS — --no-deps, then pin transformers==4.46.3
# Research finding: chatterbox requires transformers==4.46.3
# Kaggle pre-installs 5.x which breaks chatterbox's LlamaModel import
# --no-deps prevents chatterbox from re-specifying the full dep graph
# ============================================================================
print("\n[4/7] Chatterbox TTS (--no-deps + manual dep pinning) ...")

_run("chatterbox-tts --no-deps",
     _pip("chatterbox-tts", flags=["--no-deps"]), critical=False)

# Manual Chatterbox deps in safe install order
_run("conformer==0.3.2",      _pip("conformer==0.3.2"),          critical=False)
_run("resemble-perth==1.0.1", _pip("resemble-perth==1.0.1"),     critical=False)
_run("librosa",               _pip("librosa>=0.10.0"),           critical=False)
_run("s3tokenizer --no-deps", _pip("s3tokenizer", flags=["--no-deps"]), critical=False)
_run("onnx==1.16.0",          _pip("onnx==1.16.0"),              critical=False)
_run("torchaudio --no-deps",  _pip("torchaudio", flags=["--no-deps"]), critical=False)

# CRITICAL: Downgrade transformers from Kaggle 5.x to 4.46.3
# This must come AFTER chatterbox install (chatterbox may re-install transformers)
_run("transformers==4.46.3 (downgrade from 5.x, force-reinstall)",
     _pip("transformers==4.46.3", flags=["--force-reinstall", "--no-deps"]))

# ============================================================================
# STEP 4b: Kokoro TTS — Apache 2.0 fallback
# Research recommendation: best free TTS, 82M params, 210x RT, <2GB VRAM
# ============================================================================
print("  Installing Kokoro TTS fallback (Apache 2.0, 82M params) ...")
_run("kokoro>=0.9.4 + soundfile", _pip("kokoro>=0.9.4", "soundfile"), critical=False)

# ============================================================================
# STEP 4c: ACE-Step — AI music generator (copyright-free, unique per video)
# Prevents falling back to generic SoundHelix MP3 tracks
# ============================================================================
print("  Installing ACE-Step music generator ...")
_run("acestep>=0.1.0", _pip("acestep>=0.1.0"), critical=False)

# ============================================================================
# STEP 5: FINAL numpy + scipy lock
# MUST be the absolute last step — every package above may have bumped numpy
# ============================================================================
print("\n[5/7] Final numpy==1.26.4 + scipy lock (must be LAST) ...")
_run("numpy==1.26.4 + scipy (final force-reinstall)",
     _pip("numpy==1.26.4", "scipy>=1.12.0", flags=["--force-reinstall"]))

# ============================================================================
# STEP 6: Output directories
# ============================================================================
print("\n[6/7] Creating output directories ...")
DIRS = [
    "outputs/video", "outputs/audio", "outputs/visual",
    "outputs/video/scratch", "data/clip_cache", "data/assets/music",
]
for d in DIRS:
    os.makedirs(os.path.join(PIPELINE_DIR, d), exist_ok=True)
print(f"  [OK ]  {len(DIRS)} directories ready")

# ============================================================================
# STEP 7: Import verification
# ============================================================================
print("\n[7/7] Verifying imports ...")


def _check_wan(m):
    if not hasattr(m, "WanPipeline"):
        raise ImportError(
            f"WanPipeline missing (diffusers {m.__version__}) -- restart kernel!"
        )
    return f"OK  (diffusers {m.__version__}, WanPipeline found)"


def _check_scipy(m):
    m.sph_legendre_p(0, 0, 0)
    return "OK  (ufuncs functional)"


CHECK_MODULES = [
    ("numpy",         "numpy",          lambda m: m.__version__),
    ("scipy ufuncs",  "scipy.special",  _check_scipy),
    ("torch",         "torch",          lambda m: (
        m.__version__
        + " | CUDA: " + str(m.cuda.is_available())
        + (" | GPUs: " + str(m.cuda.device_count()) if m.cuda.is_available() else "")
        + (" | " + m.cuda.get_device_name(0) if m.cuda.is_available() else "")
    )),
    ("diffusers",     "diffusers",      lambda m: (
        f"OK  (diffusers {m.__version__}"
        + (", LTXPipeline ✓" if hasattr(m, "LTXPipeline") else ", LTXPipeline ✗ — upgrade diffusers!")
        + ")"
    )),
    ("transformers",  "transformers",   lambda m: m.__version__),
    ("accelerate",    "accelerate",     lambda m: m.__version__),
    ("chatterbox",    "chatterbox.tts", lambda m: "OK"),
    ("kokoro",        "kokoro",         lambda m: getattr(m, "__version__", "OK")),
    ("groq",          "groq",           lambda m: m.__version__),
    ("langchain",     "langchain",      lambda m: m.__version__),
    ("langgraph",     "langgraph",      lambda m: getattr(m, "__version__", "OK")),
    ("edge_tts",      "edge_tts",       lambda m: getattr(m, "__version__", "OK")),
    ("structlog",     "structlog",      lambda m: m.__version__),
]

pass_count = 0
fail_count = 0
numpy_ok   = False

for label, mod_name, get_ver in CHECK_MODULES:
    try:
        m   = importlib.import_module(mod_name)
        ver = get_ver(m)
        print(f"  [OK ]  {label:<18} {ver}")
        if label == "numpy":
            if int(m.__version__.split(".")[0]) >= 2:
                print(f"         WARN: numpy {m.__version__} is 2.x -- restart kernel!")
                fail_count += 1
                continue
            numpy_ok = True
        pass_count += 1
    except (ImportError, ValueError, AttributeError) as e:
        es = str(e)
        if "dtype size" in es or "numpy" in es.lower():
            print(f"  [WARN] {label:<18} numpy binary mismatch -- restart kernel!")
        else:
            print(f"  [FAIL] {label:<18} {es[:80]}")
        fail_count += 1

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 64)
total = pass_count + fail_count
if fail_count == 0:
    print(f"SETUP PASSED ({pass_count}/{total} checks passed)")
else:
    print(f"SETUP WARNINGS ({fail_count}/{total} checks failed -- see above)")

if not numpy_ok:
    print("\nACTION: numpy 2.x detected or numpy check failed.")
    print("  -> Restart kernel, then re-run Cell 1.")
elif fail_count == 0:
    print("\nAll clear! Proceed to Cell 2 to pre-load GPU models.")
else:
    print(f"\n{fail_count} check(s) failed.")
    print("  If diffusers/chatterbox failed: restart kernel, re-run Cell 1.")

print("=" * 64)
