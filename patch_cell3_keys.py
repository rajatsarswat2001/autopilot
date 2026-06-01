import json

def patch():
    with open("autopilot_kaggle.ipynb", "r", encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and any("CELL 3" in line for line in cell['source']):
            new_source = [
                "# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\n",
                "# CELL 3 \u2014 API KEYS & SETTINGS\n",
                "# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\n",
                "import os\n",
                "PIPELINE_DIR = '/kaggle/working/autopilot/autopilot_pipeline'\n",
                "\n",
                "# \u2500\u2500 API Keys (Loaded from Kaggle Secrets OR paste below) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
                "keys = {\n",
                "    'GROQ_API_KEYS': 'YOUR_GROQ_KEY_1,YOUR_GROQ_KEY_2',\n",
                "    'GEMINI_API_KEYS': 'YOUR_GEMINI_KEY_1,YOUR_GEMINI_KEY_2',\n",
                "    'NVIDIA_API_KEY': 'YOUR_NVIDIA_KEY',\n",
                "    'DEEPSEEK_API_KEY': 'YOUR_DEEPSEEK_KEY',\n",
                "    'PEXELS_API_KEYS': 'YOUR_PEXELS_KEY',\n",
                "    'TAVILY_API_KEY': 'YOUR_TAVILY_KEY',\n",
                "}\n",
                "\n",
                "# Attempt to override with Kaggle Secrets if available\n",
                "try:\n",
                "    from kaggle_secrets import UserSecretsClient\n",
                "    user_secrets = UserSecretsClient()\n",
                "    for k in keys.keys():\n",
                "        try:\n",
                "            secret_val = user_secrets.get_secret(k)\n",
                "            if secret_val:\n",
                "                keys[k] = secret_val\n",
                "        except:\n",
                "            pass\n",
                "except ImportError:\n",
                "    pass\n",
                "\n",
                "# \u2500\u2500 Pipeline settings \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
                "import torch\n",
                "has_gpu = torch.cuda.is_available()\n",
                "\n",
                "settings = {\n",
                "    'AUTOPILOT_AUTO_APPROVE': '1',\n",
                "    'AUDIO_PARALLEL_WORKERS': '1' if has_gpu else '4',\n",
                "    'VISUAL_PARALLEL_WORKERS': '1' if has_gpu else '4',\n",
                "    'VIDEO_GEN_ENABLED':          '1' if has_gpu else '0',\n",
                "    'VIDEO_GEN_WAN_STEPS':    '50',\n",
                "    'VIDEO_GEN_LTX_STEPS':    '40',\n",
                "    'VIDEO_GEN_COG_STEPS':    '50',\n",
                "    'DISABLE_STOCK':          '1',\n",
                "    'LOG_LEVEL':              'INFO',\n",
                "    'FORMAT':                 'short',\n",
                "}\n",
                "keys.update(settings)\n",
                "\n",
                "env_path = os.path.join(PIPELINE_DIR, '.env')\n",
                "with open(env_path, 'w') as f:\n",
                "    for k, v in keys.items():\n",
                "        # Only write/export valid keys\n",
                "        if v and 'YOUR_' not in v:\n",
                "            f.write(f'{k}={v}\\n')\n",
                "            os.environ[k] = str(v)\n",
                "        elif k in os.environ and 'YOUR_' in v:\n",
                "            # Do not overwrite valid env vars with placeholders\n",
                "            pass\n",
                "\n",
                "print(f'.env written: {env_path}')\n",
                "print(f'GPU mode: {\"ON (Wan2.1 + Chatterbox active)\" if has_gpu else \"OFF (Pexels + Edge TTS)\"}')\n",
                "print('\\n\u2705 Run Cell 4 to generate a video')\n"
            ]
            cell['source'] = new_source
            break

    with open("autopilot_kaggle.ipynb", "w", encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    patch()
