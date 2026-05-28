"""
patch_notebook.py — Fixes numpy binary incompatibility in autopilot_kaggle.ipynb
Run: python patch_notebook.py
"""
import json

NB_PATH = "autopilot_kaggle.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

# ── Find Cell 1 ──────────────────────────────────────────────────────────────
cell1 = next(c for c in nb["cells"] if c.get("id") == "cell-1-setup")

# Rebuild source as a single string for easier editing
src_lines = cell1["source"]
if isinstance(src_lines, list):
    src = "".join(src_lines)
else:
    src = src_lines

# ── PATCH 1: Insert numpy pin step right after PIPELINE_DIR line ─────────────
NUMPY_STEP = """

# -- 0. Pin numpy FIRST to 1.26.4 ──────────────────────────────────────────
# Kaggle's pre-installed PyTorch is compiled against numpy 1.x.
# chatterbox-tts and transformers>=4.42 pull in numpy 2.x which causes:
#   ValueError: numpy.dtype size changed, may indicate binary incompatibility
# We must pin numpy BEFORE those installs happen.
print('[0/6] Pinning numpy==1.26.4 (prevents numpy 2.x binary mismatch)...')
r_np = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', 'numpy==1.26.4'],
    capture_output=True, text=True
)
print('  ' + ('OK  numpy pinned to 1.26.4' if r_np.returncode == 0 else 'WARN: ' + r_np.stderr[-200:]))

"""
ANCHOR1 = "PIPELINE_DIR = '/kaggle/working/autopilot/autopilot_pipeline'"
if ANCHOR1 in src and "[0/6]" not in src:
    src = src.replace(ANCHOR1, ANCHOR1 + NUMPY_STEP)
    print("PATCH 1 applied: numpy pre-pin step added")
else:
    print("PATCH 1 skipped (already applied or anchor not found)")

# ── PATCH 2: Add numpy==1.26.4 to core_pkgs list ────────────────────────────
ANCHOR2 = "    'structlog',"
REPLACE2 = "    'numpy==1.26.4',       # keep pinned in combined install\n    'structlog',"
if ANCHOR2 in src and "numpy==1.26.4" not in src.split("core_pkgs")[1].split("]")[0]:
    src = src.replace(ANCHOR2, REPLACE2, 1)
    print("PATCH 2 applied: numpy added to core_pkgs")
else:
    print("PATCH 2 skipped (already applied)")

# ── PATCH 3: Re-pin numpy after chatterbox install (step 4) ─────────────────
ANCHOR3 = "print('[4/6] Chatterbox TTS (MIT, best free voice)...')"
REPIN_BLOCK = """
# -- Re-pin numpy AFTER chatterbox install (chatterbox deps may bump it) ──
r_repin = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall', 'numpy==1.26.4'],
    capture_output=True, text=True
)
try:
    import importlib, numpy as _np
    importlib.reload(_np)
except Exception:
    pass

"""
# Insert the re-pin block before the verification section
ANCHOR3B = "# -- Verification"
if ANCHOR3B not in src:
    ANCHOR3B = "# ── Verification"
if ANCHOR3B in src and "force-reinstall" not in src:
    src = src.replace(ANCHOR3B, REPIN_BLOCK + ANCHOR3B)
    print("PATCH 3 applied: numpy re-pin after chatterbox")
else:
    print("PATCH 3 skipped (already applied or anchor not found)")

# ── PATCH 4: Fix verification loop to catch ValueError too ──────────────────
OLD_EXCEPT = "    except ImportError:\n        checks.append((mods[1], '\u274c NOT INSTALLED'))"
NEW_EXCEPT = (
    "    except (ImportError, ValueError) as _e:\n"
    "        _es = str(_e)\n"
    "        if 'dtype size' in _es or 'numpy' in _es.lower():\n"
    "            checks.append((mods[1], '\u26a0\ufe0f numpy mismatch \u2014 restart kernel, re-run Cell 1'))\n"
    "        else:\n"
    "            checks.append((mods[1], '\u274c ' + _es[:60]))"
)
if OLD_EXCEPT in src:
    src = src.replace(OLD_EXCEPT, NEW_EXCEPT)
    print("PATCH 4 applied: ValueError catch added to verification loop")
else:
    # Try alternate form
    OLD_EXCEPT2 = "    except ImportError:\n        checks.append((mods[1], '\u274c NOT INSTALLED'))\n"
    if OLD_EXCEPT2 in src:
        src = src.replace(OLD_EXCEPT2, NEW_EXCEPT + "\n")
        print("PATCH 4 applied (alt form)")
    else:
        print("PATCH 4 skipped (already applied or pattern not found)")

# ── Write back ───────────────────────────────────────────────────────────────
cell1["source"] = src
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Validate
with open(NB_PATH, encoding="utf-8") as f:
    nb2 = json.load(f)
print(f"\nNotebook OK: {len(nb2['cells'])} cells, nbformat {nb2['nbformat']}")
print("numpy patch complete - push to GitHub to apply on Kaggle")
