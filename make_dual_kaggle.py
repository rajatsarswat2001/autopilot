import json

def code_cell(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

def md_cell(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

cells = []

cells.append(md_cell(
    "# AutoPilot — Ultimate Dual-GPU Pipeline\n"
    "Run the **entire** AutoPilot pipeline purely inside Kaggle!\n\n"
    "- Uses **both T4 GPUs** simultaneously (2x generation speed)\n"
    "- No local execution required\n"
    "- Outputs are saved straight to Kaggle's `/output` directory"
))

# ── Cell 1: Clone Repo & Install dependencies ────────────────────────────────
cells.append(code_cell(
    "# ── Cell 1: Environment Setup & Clone Repo ────────────────────────────────\n"
    "import os, sys, subprocess\n\n"
    "CLONE_DIR = '/kaggle/working/autopilot'\n\n"
    "print('[1/3] Pinning numpy...\\n')\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy==1.26.4'])\n\n"
    "print('[2/3] Cloning AutoPilot repo...\\n')\n"
    "if not os.path.exists(CLONE_DIR):\n"
    "    subprocess.run(['git', 'clone', 'https://github.com/rajatsarswat2001/autopilot.git', CLONE_DIR])\n\n"
    "print('[3/3] Installing Kaggle requirements (this takes a few minutes)...\\n')\n"
    "subprocess.run([sys.executable, os.path.join(CLONE_DIR, 'kaggle_setup.py')])\n\n"
    "print('\\n✅ Setup complete! Note: if this is your first run, RESTART KERNEL now to load new packages.\\n')"
))

# ── Cell 2: Download Models ──────────────────────────────────────────────────
cells.append(code_cell(
    "# ── Cell 2: Download Wan 2.2 Models & ComfyUI ────────────────────────────\n"
    "import os\n\n"
    "COMFY_DIR = '/kaggle/working/ComfyUI'\n"
    "if not os.path.exists(COMFY_DIR):\n"
    "    os.system('git clone https://github.com/comfyanonymous/ComfyUI.git /kaggle/working/ComfyUI')\n\n"
    "BASE = 'https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files'\n"
    "WAN21_BASE = 'https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files'\n\n"
    "DIRS = {\n"
    "    'diffusion': f'{COMFY_DIR}/models/diffusion_models',\n"
    "    'text_enc' : f'{COMFY_DIR}/models/text_encoders',\n"
    "    'vae'      : f'{COMFY_DIR}/models/vae',\n"
    "    'clip_vis' : f'{COMFY_DIR}/models/clip_vision',\n"
    "}\n"
    "for d in DIRS.values(): os.makedirs(d, exist_ok=True)\n\n"
    "def dl(url, dest_dir, fname):\n"
    "    dest = os.path.join(dest_dir, fname)\n"
    "    if not os.path.exists(dest):\n"
    "        print(f'📥 Downloading {fname} ...')\n"
    "        os.system(f\"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M '{url}' -d '{dest_dir}' -o '{fname}'\")\n\n"
    "dl(f'{BASE}/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors', DIRS['diffusion'], 'wan2.2_ti2v_5B_fp16.safetensors')\n"
    "dl(f'{BASE}/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors', DIRS['text_enc'], 'umt5_xxl_fp8_e4m3fn_scaled.safetensors')\n"
    "dl(f'{BASE}/vae/wan2.2_vae.safetensors', DIRS['vae'], 'wan2.2_vae.safetensors')\n"
    "dl(f'{WAN21_BASE}/clip_vision/clip_vision_h.safetensors', DIRS['clip_vis'], 'clip_vision_h.safetensors')\n\n"
    "print('✅ Models ready!')"
))

# ── Cell 3: Secrets to .env ──────────────────────────────────────────────────
cells.append(code_cell(
    "# ── Cell 3: Configure Pipeline Environment ────────────────────────────────\n"
    "import os\n"
    "try:\n"
    "    from kaggle_secrets import UserSecretsClient\n"
    "    secrets = UserSecretsClient()\n"
    "except ImportError:\n"
    "    secrets = None\n\n"
    "KEYS = ['NVIDIA_API_KEY', 'PEXELS_API_KEYS', 'GROQ_API_KEYS', 'OPENAI_API_KEY']\n"
    "env_content = []\n\n"
    "if secrets:\n"
    "    for k in KEYS:\n"
    "        try:\n"
    "            val = secrets.get_secret(k)\n"
    "            if val:\n"
    "                env_content.append(f'{k}={val}')\n"
    "        except:\n"
    "            pass\n\n"
    "# CRITICAL: Point AutoPilot to our internal Dual-GPU Load Balancer\n"
    "env_content.extend([\n"
    "    'KAGGLE_NGROK_URL=http://127.0.0.1:8080',\n"
    "    'VISUAL_PARALLEL_WORKERS=2',\n"
    "    'VIDEO_GEN_MODEL=wan22',\n"
    "    'VIDEO_GEN_ENABLED=1'\n"
    "])\n\n"
    "with open('/kaggle/working/autopilot/autopilot_pipeline/.env', 'w') as f:\n"
    "    f.write('\\n'.join(env_content))\n\n"
    "print('✅ .env written securely')"
))

# ── Cell 4: Write Worker and Balancer scripts ────────────────────────────────
cells.append(code_cell(
    "# ── Cell 4: Launch Dual-GPU ComfyUI Workers + Load Balancer ──────────────\n"
    "import os, time, subprocess, sys\n\n"
    "worker_code = \"\"\"\n"
    "import gc, os, random, sys, torch, imageio, numpy as np\n"
    "from pathlib import Path\n"
    "from fastapi import FastAPI, UploadFile, File, Form, HTTPException\n"
    "from fastapi.responses import FileResponse\n"
    "import uvicorn\n\n"
    "sys.path.insert(0, '/kaggle/working/ComfyUI')\n"
    "from nodes import NODE_CLASS_MAPPINGS\n"
    "OUTPUT_DIR = '/kaggle/working/output'\n"
    "INPUT_DIR  = '/kaggle/working/input'\n"
    "os.makedirs(OUTPUT_DIR, exist_ok=True)\n"
    "os.makedirs(INPUT_DIR, exist_ok=True)\n\n"
    "app = FastAPI()\n"
    "@app.get('/health')\n"
    "def health(): return {'status': 'ok'}\n\n"
    "@app.post('/generate_video')\n"
    "async def api_generate(\n"
    "    image: UploadFile = File(None), prompt: str = Form(...),\n"
    "    seed: int = Form(0), steps: int = Form(30),\n"
    "    width: int = Form(832), height: int = Form(480), frames: int = Form(49)\n"
    "):\n"
    "    # Load model and generate video\n"
    "    if seed == 0: seed = random.randint(0, 2**32-1)\n"
    "    with torch.inference_mode():\n"
    "        clip = NODE_CLASS_MAPPINGS['CLIPLoader']().load_clip('umt5_xxl_fp8_e4m3fn_scaled.safetensors', 'wan', 'default')[0]\n"
    "        pos_cond = NODE_CLASS_MAPPINGS['CLIPTextEncode']().encode(clip, prompt)[0]\n"
    "        neg_cond = NODE_CLASS_MAPPINGS['CLIPTextEncode']().encode(clip, 'text, watermark')[0]\n"
    "        del clip; gc.collect(); torch.cuda.empty_cache()\n\n"
    "        loaded_image, clip_vis_out = None, None\n"
    "        if image is not None and image.filename:\n"
    "            img_path = f'{INPUT_DIR}/{image.filename}'\n"
    "            with open(img_path, 'wb') as f: f.write(await image.read())\n"
    "            loaded_image = NODE_CLASS_MAPPINGS['LoadImage']().load_image(img_path)[0]\n"
    "            clip_vis = NODE_CLASS_MAPPINGS['CLIPVisionLoader']().load_clip('clip_vision_h.safetensors')[0]\n"
    "            clip_vis_out = NODE_CLASS_MAPPINGS['CLIPVisionEncode']().encode(clip_vis, loaded_image, 'none')[0]\n"
    "            del clip_vis; gc.collect(); torch.cuda.empty_cache()\n\n"
    "        vae = NODE_CLASS_MAPPINGS['VAELoader']().load_vae('wan2.2_vae.safetensors')[0]\n"
    "        wan_cls = NODE_CLASS_MAPPINGS.get('WanImageToVideo') or NODE_CLASS_MAPPINGS.get('Wan TI2V Encode')\n"
    "        if loaded_image is not None and wan_cls:\n"
    "            pos_cond, neg_cond, lat = wan_cls().encode(pos_cond, neg_cond, vae, width, height, frames, 1, loaded_image, clip_vis_out)\n"
    "        else:\n"
    "            lat = NODE_CLASS_MAPPINGS['EmptyLatentImage']().generate(width, height, 1)[0]\n\n"
    "        model = NODE_CLASS_MAPPINGS['UNETLoader']().load_unet('wan2.2_ti2v_5B_fp16.safetensors', 'default')[0]\n"
    "        sampled = NODE_CLASS_MAPPINGS['KSampler']().sample(model, seed, steps, 6.0, 'euler', 'simple', pos_cond, neg_cond, lat, 1.0)[0]\n"
    "        del model; gc.collect(); torch.cuda.empty_cache()\n\n"
    "        decoded = NODE_CLASS_MAPPINGS['VAEDecode']().decode(vae, sampled)[0]\n"
    "        del vae, sampled; gc.collect(); torch.cuda.empty_cache()\n\n"
    "        out_path = f'{OUTPUT_DIR}/wan22_{seed}.mp4'\n"
    "        frames_np = [(f.cpu().numpy() * 255).astype(np.uint8) for f in decoded]\n"
    "        with imageio.get_writer(out_path, fps=16) as writer:\n"
    "            for frame in frames_np: writer.append_data(frame)\n\n"
    "        return FileResponse(out_path, media_type='video/mp4', filename='output.mp4')\n"
    "if __name__ == '__main__':\n"
    "    uvicorn.run(app, host='127.0.0.1', port=int(sys.argv[1]))\n"
    "\"\"\"\n"
    "with open('worker.py', 'w') as f: f.write(worker_code)\n\n"
    "balancer_code = \"\"\"\n"
    "import httpx, uvicorn, asyncio, gc\n"
    "from fastapi import FastAPI, UploadFile, File, Form, HTTPException\n"
    "from fastapi.responses import FileResponse\n\n"
    "app = FastAPI()\n"
    "WORKERS = ['http://127.0.0.1:8001', 'http://127.0.0.1:8002']\n"
    "idx = 0; lock = asyncio.Lock()\n\n"
    "@app.get('/health')\n"
    "def health(): return {'status': 'ok'}\n\n"
    "@app.post('/generate_video')\n"
    "async def gen(image: UploadFile = File(None), prompt: str = Form(...), seed: int = Form(0), steps: int = Form(30), width: int = Form(832), height: int = Form(480), frames: int = Form(49)):\n"
    "    global idx\n"
    "    async with lock:\n"
    "        target = WORKERS[idx]\n"
    "        idx = (idx + 1) % len(WORKERS)\n"
    "    data = {'prompt': prompt, 'seed': str(seed), 'steps': str(steps), 'width': str(width), 'height': str(height), 'frames': str(frames)}\n"
    "    files = {}\n"
    "    if image and image.filename:\n"
    "        files = {'image': (image.filename, await image.read(), image.content_type)}\n"
    "    async with httpx.AsyncClient(timeout=3600.0) as client:\n"
    "        resp = await client.post(f'{target}/generate_video', data=data, files=files)\n"
    "        out_file = f'/kaggle/working/output/proxied_{seed}.mp4'\n"
    "        with open(out_file, 'wb') as f: f.write(resp.content)\n"
    "        gc.collect()\n"
    "        return FileResponse(out_file, media_type='video/mp4')\n"
    "if __name__ == '__main__':\n"
    "    uvicorn.run(app, host='0.0.0.0', port=8080)\n"
    "\"\"\"\n"
    "with open('balancer.py', 'w') as f: f.write(balancer_code)\n\n"
    "print('Starting Load Balancer (Port 8080)...')\n"
    "subprocess.Popen([sys.executable, 'balancer.py'])\n\n"
    "print('Starting Worker 0 on GPU 0 (Port 8001)...')\n"
    "env0 = os.environ.copy()\n"
    "env0['CUDA_VISIBLE_DEVICES'] = '0'\n"
    "subprocess.Popen([sys.executable, 'worker.py', '8001'], env=env0)\n\n"
    "print('Starting Worker 1 on GPU 1 (Port 8002)...')\n"
    "env1 = os.environ.copy()\n"
    "env1['CUDA_VISIBLE_DEVICES'] = '1'\n"
    "subprocess.Popen([sys.executable, 'worker.py', '8002'], env=env1)\n\n"
    "import requests\n"
    "for p in [8080, 8001, 8002]:\n"
    "    while True:\n"
    "        try:\n"
    "            requests.get(f'http://127.0.0.1:{p}/health')\n"
    "            print(f'✅ Port {p} is UP')\n"
    "            break\n"
    "        except:\n"
    "            time.sleep(2)\n"
))

# ── Cell 5: Run AutoPilot ────────────────────────────────────────────────────
cells.append(code_cell(
    "# ── Cell 5: Run AutoPilot Pipeline! ──────────────────────────────────────\n"
    "import os, sys, subprocess\n\n"
    "PIPELINE_DIR = '/kaggle/working/autopilot/autopilot_pipeline'\n"
    "NICHE = 'personal_finance'\n"
    "TOPIC = ''\n\n"
    "print(f'\\n🚀 Starting Pipeline (Niche: {NICHE})')\n"
    "cmd = [sys.executable, 'main.py', '--niche', NICHE, '--no-db', '--approve', '--log-format', 'console']\n"
    "if TOPIC: cmd.extend(['--topic', TOPIC])\n\n"
    "proc = subprocess.Popen(cmd, cwd=PIPELINE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
    "for line in proc.stdout: print(line, end='', flush=True)\n"
    "proc.wait()\n\n"
    "print(f'\\n✅ Pipeline Finished with code {proc.returncode}\\nCheck /kaggle/working/autopilot/autopilot_pipeline/outputs/video for MP4 files')"
))

notebook = {
    'cells': cells,
    'metadata': {'accelerator': 'GPU'},
    'nbformat': 4,
    'nbformat_minor': 5
}

with open('final_dual_kaggle.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print('Done writing final_dual_kaggle.ipynb')
