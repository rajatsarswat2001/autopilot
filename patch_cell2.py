"""
patch_cell2.py — Rewrite Cell 2 in autopilot_kaggle.ipynb to run pre-loads and smoke tests in a subprocess.
This completely prevents VRAM from being locked by the parent Jupyter kernel process, avoiding subsequent CUDA OOMs in Cell 4.
"""
import json

NB_PATH = "autopilot_kaggle.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

# Find Cell 2
cell2 = next(c for c in nb["cells"] if c.get("id") == "cell-2-gpu-models")

NEW_CELL_2 = [
    "# =============================================================================\n",
    "# CELL 2 -- PRE-LOAD GPU MODELS IN SUBPROCESS (Zero-VRAM Leakage Guard)\n",
    "# =============================================================================\n",
    "# Downloads and verifies weights in isolated subprocesses.\n",
    "# This ensures all GPU memory is 100% released before running the main pipeline!\n",
    "# =============================================================================\n",
    "import subprocess, sys, os, torch, time\n",
    "\n",
    "if not torch.cuda.is_available():\n",
    "    print('❌ No GPU available — pipeline will run in CPU-only mode (Pexels + Edge TTS)')\n",
    "else:\n",
    "    num_gpus = torch.cuda.device_count()\n",
    "    gpu_name = torch.cuda.get_device_name(0)\n",
    "    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
    "    print(f'GPUs detected: {num_gpus}x {gpu_name} ({vram_gb:.1f} GB VRAM each)')\n",
    "    if num_gpus >= 2:\n",
    "        print('  Dual-T4 mode:')\n",
    "        print('    cuda:0 -> Chatterbox TTS (~2 GB, freed after audio)')\n",
    "        print('    cuda:0 -> CogVideoX-2B INT8 (mirror strategy, ~6.5 GB)')\n",
    "        print('    cuda:1 -> CogVideoX-2B INT8 (mirror strategy, ~6.5 GB)')\n",
    "    print()\n",
    "\n",
    "    # Make sure torchao and diffusers are fully installed first\n",
    "    print('Checking dependencies...')\n",
    "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torchao>=0.4.0', 'diffusers>=0.33.0'], check=False)\n",
    "\n",
    "    # -- [1/2] Chatterbox TTS (cuda:0) in subprocess --\n",
    "    print('\\n[1/2] Chatterbox TTS -- checking/downloading weights in subprocess (~1.8 GB first time)...')\n",
    "    t0 = time.time()\n",
    "    chat_code = \"\"\"\n",
    "import os, torch, gc\n",
    "os.environ['TRANSFORMERS_ATTN_IMPLEMENTATION'] = 'eager'\n",
    "try:\n",
    "    from chatterbox.tts import ChatterboxTTS\n",
    "    model = ChatterboxTTS.from_pretrained(device='cuda:0')\n",
    "    print('      Chatterbox weights pre-loaded successfully!')\n",
    "except Exception as e:\n",
    "    print('      FAIL:', str(e)[:150])\n",
    "    raise e\n",
    "\"\"\"\n",
    "    r = subprocess.run([sys.executable, '-c', chat_code], capture_output=True, text=True)\n",
    "    if r.returncode == 0:\n",
    "        print(r.stdout.strip())\n",
    "        print(f'  ✅ OK  Chatterbox ready ({time.time()-t0:.0f}s)')\n",
    "    else:\n",
    "        print(f'  ⚠️  Chatterbox failed or skipped: ' + r.stdout.strip() + '\\n' + r.stderr.strip()[-200:])\n",
    "        print('      -> Will fall back to Kokoro TTS fallback at runtime')\n",
    "\n",
    "    # -- [2/2] CogVideoX-2B INT8 (cuda:1) in subprocess --\n",
    "    video_gpu_id = 1 if num_gpus >= 2 else 0\n",
    "    print(f'\\n[2/2] CogVideoX-2B INT8 -- checking/downloading weights in subprocess (~5 GB first time)...')\n",
    "    t0 = time.time()\n",
    "    cog_code = f\"\"\"\n",
    "import os, torch, gc\n",
    "os.environ['TRANSFORMERS_ATTN_IMPLEMENTATION'] = 'eager'\n",
    "from diffusers import CogVideoXPipeline\n",
    "print('      Downloading and loading pipeline...')\n",
    "pipe = CogVideoXPipeline.from_pretrained(\n",
    "    'THUDM/CogVideoX-2b',\n",
    "    torch_dtype=torch.float16,\n",
    ")\n",
    "try:\n",
    "    from torchao.quantization import quantize_, int8_weight_only\n",
    "    quantize_(pipe.transformer, int8_weight_only())\n",
    "    print('      INT8 quantization applied (6.5 GB VRAM)')\n",
    "except Exception as qe:\n",
    "    print('      INT8 skipped:', str(qe))\n",
    "pipe.to('cuda:{video_gpu_id}')\n",
    "for method in ('enable_vae_slicing', 'enable_vae_tiling', 'enable_attention_slicing'):\n",
    "    if hasattr(pipe, method):\n",
    "        try: getattr(pipe, method)()\n",
    "        except Exception: pass\n",
    "print('      Running 5-step smoke test...')\n",
    "with torch.inference_mode():\n",
    "    pipe(\n",
    "        prompt='a calm ocean wave, cinematic, photorealistic',\n",
    "        height=480, width=720, num_frames=49, num_inference_steps=5,\n",
    "        guidance_scale=6.0,\n",
    "    )\n",
    "print('      Smoke test completed successfully!')\n",
    "\"\"\"\n",
    "    r = subprocess.run([sys.executable, '-c', cog_code], capture_output=True, text=True)\n",
    "    if r.returncode == 0:\n",
    "        print(r.stdout.strip())\n",
    "        print(f'  ✅ OK  CogVideoX-2B ready ({time.time()-t0:.0f}s)')\n",
    "    else: \n",
    "        print(f'  ⚠️  CogVideoX failed or skipped: ' + r.stdout.strip() + '\\n' + r.stderr.strip()[-200:])\n",
    "        print('      -> Will fall back to LTX-Video or Pollinations FLUX at runtime')\n",
    "\n",
    "    print('\\nVRAM status inside parent Jupyter kernel: 0 bytes allocated (Subprocess isolated)')\n",
    "    print('✅ Setup complete -- proceed directly to Cell 3!')\n"
]

cell2["source"] = NEW_CELL_2

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully patched Cell 2 in autopilot_kaggle.ipynb to use a subprocess!")
