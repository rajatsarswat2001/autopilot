"""
patch_cell3.py — Rewrite Cell 3 (API keys) to ONLY use hardcoded keys.
"""
import json

NB_PATH = "autopilot_kaggle.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

# Find Cell 3
cell3 = next(c for c in nb["cells"] if c.get("id") == "cell-3-keys")

NEW_CELL_3 = [
    "# ═══════════════════════════════════════════════════════════════════════════\n",
    "# CELL 3 — API KEYS & SETTINGS (Hardcoded)\n",
    "# ═══════════════════════════════════════════════════════════════════════════\n",
    "import os\n",
    "PIPELINE_DIR = '/kaggle/working/autopilot/autopilot_pipeline'\n",
    "\n",
    "# ── API Keys (paste yours here) ──────────────────────────────────────────────\n",
    "keys = {\n",
    "    # Groq (Free at console.groq.com). Add multiple separated by commas for rotation.\n",
    "    'GROQ_API_KEYS': 'YOUR_GROQ_KEY_1,YOUR_GROQ_KEY_2',\n",
    "\n",
    "    # Gemini (Free at aistudio.google.com). Add multiple separated by commas.\n",
    "    'GEMINI_API_KEYS': 'YOUR_GEMINI_KEY_1,YOUR_GEMINI_KEY_2',\n",
    "\n",
    "    # Pexels (Free at pexels.com/api)\n",
    "    'PEXELS_API_KEYS': 'YOUR_PEXELS_KEY',\n",
    "\n",
    "    # Tavily (Free 1000/month at app.tavily.com)\n",
    "    'TAVILY_API_KEY': 'YOUR_TAVILY_KEY',\n",
    "}\n",
    "\n",
    "# ── Pipeline settings ────────────────────────────────────────────────────────\n",
    "import torch\n",
    "has_gpu = torch.cuda.is_available()\n",
    "\n",
    "settings = {\n",
    "    'AUTOPILOT_AUTO_APPROVE': '1',\n",
    "    'AUDIO_PARALLEL_WORKERS': '1' if has_gpu else '4',   # sequential on GPU to avoid VRAM contention\n",
    "    'VISUAL_PARALLEL_WORKERS': '1' if has_gpu else '4',  # sequential on GPU for Wan2.1\n",
    "    'VIDEO_GEN_ENABLED':          '1' if has_gpu else '0',   # disable on CPU-only\n",
    "    'LOG_LEVEL':              'INFO',\n",
    "    'FORMAT':                 'short',    # 'short' = 9:16 vertical (YouTube Shorts)\n",
    "}\n",
    "keys.update(settings)\n",
    "\n",
    "env_path = os.path.join(PIPELINE_DIR, '.env')\n",
    "with open(env_path, 'w') as f:\n",
    "    for k, v in keys.items():\n",
    "        if v and 'YOUR_' not in v:\n",
    "            f.write(f'{k}={v}\\n')\n",
    "        os.environ[k] = str(v)\n",
    "\n",
    "print(f'.env written: {env_path}')\n",
    "print(f'GPU mode: {\"ON (Wan2.1 + Chatterbox active)\" if has_gpu else \"OFF (Pexels + Edge TTS)\"}')\n",
    "print('\\n✅ Run Cell 4 to generate a video')\n"
]

cell3["source"] = NEW_CELL_3

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Patched Cell 3 to use hardcoded API keys only.")
