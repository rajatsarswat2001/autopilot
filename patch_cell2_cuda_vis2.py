import json

def patch():
    with open("autopilot_kaggle.ipynb", "r", encoding='utf-8') as f:
        content = f.read()
    
    # Target 2: wan_code
    old_wan = """    "    wan_code = f\\"\\"\\"\\n",
    "import os, torch, gc\\n",
    "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\\n",
    "from diffusers import WanPipeline\\n",
    "print('      Loading Wan2.1 T2V-1.3B...')\\n",
    "pipe = WanPipeline.from_pretrained(\\n",
    "    'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',\\n",
    "    torch_dtype=torch.float16,\\n",
    ")\\n",
    "pipe.to(f'cuda:{video_gpu_id}')\\n",
    "pipe.enable_attention_slicing()\\n",
    "pipe.vae.enable_slicing()\\n",
    "print('      Running 5-step smoke test...')\\n",
    "with torch.no_grad():\\n",
    "    pipe(\\n",
    "        prompt='a calm ocean wave, cinematic',\\n",
    "        width=832, height=480, num_frames=9,\\n",
    "        num_inference_steps=5, guidance_scale=5.0,\\n",
    "    )\\n",
    "print('      Smoke test completed successfully!')\\n",
    "del pipe; gc.collect(); torch.cuda.empty_cache()\\n",
    "\\"\\"\\"\\n","""

    new_wan = """    "    wan_code = f\\"\\"\\"\\n",
    "import os, torch, gc\\n",
    "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\\n",
    "from diffusers import WanPipeline\\n",
    "pipe = WanPipeline.from_pretrained(\\n",
    "    'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',\\n",
    "    torch_dtype=torch.float16,\\n",
    ")\\n",
    "pipe.to('cuda:0')\\n",
    "pipe.enable_attention_slicing()\\n",
    "pipe.vae.enable_slicing()\\n",
    "pipe.vae.enable_tiling()\\n",
    "with torch.no_grad():\\n",
    "    pipe(prompt='a calm ocean wave, cinematic',\\n",
    "         width=832, height=480, num_frames=9,\\n",
    "         num_inference_steps=5, guidance_scale=5.0)\\n",
    "print('Smoke test passed')\\n",
    "del pipe; gc.collect(); torch.cuda.empty_cache()\\n",
    "\\"\\"\\"\\n","""
    
    if old_wan in content:
        content = content.replace(old_wan, new_wan)
        with open("autopilot_kaggle.ipynb", "w", encoding='utf-8') as f:
            f.write(content)
        print("wan_code patched successfully using string replacement.")
    else:
        print("Could not find old_wan string")

if __name__ == "__main__":
    patch()
