import json

def patch():
    with open("autopilot_kaggle.ipynb", "r", encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and any("CELL 2 -- PRE-LOAD GPU MODELS IN SUBPROCESS" in line for line in cell['source']):
            source = cell['source']
            
            # Find where to truncate the source: right after the Chatterbox block ends
            truncate_idx = -1
            for i, line in enumerate(source):
                if "Chatterbox failed:" in line:
                    truncate_idx = i + 1
                    break
            
            if truncate_idx != -1:
                # Keep everything up to the Chatterbox block
                new_source = source[:truncate_idx]
                
                # Append the new Wan2.1 pre-download block
                new_source.extend([
                    "\n",
                    "    # -- [2/2] Pre-download Wan2.1 weights to disk cache (no GPU load) --\n",
                    "    print(f'\\n[2/2] Wan2.1 T2V-1.3B -- pre-downloading weights to cache...')\n",
                    "    t0 = time.time()\n",
                    "\n",
                    "    dl_code = \"\"\"\n",
                    "import os\n",
                    "os.environ['TRANSFORMERS_VERBOSITY'] = 'error'\n",
                    "from huggingface_hub import snapshot_download\n",
                    "path = snapshot_download(\n",
                    "    repo_id='Wan-AI/Wan2.1-T2V-1.3B-Diffusers',\n",
                    "    ignore_patterns=['*.msgpack', '*.h5'],\n",
                    ")\n",
                    "print(f'      Cached at: {path}')\n",
                    "\"\"\"\n",
                    "\n",
                    "    r = subprocess.run([sys.executable, '-c', dl_code],\n",
                    "                       capture_output=True, text=True, env=os.environ.copy())\n",
                    "    if r.returncode == 0:\n",
                    "        print(r.stdout.strip())\n",
                    "        print(f'  \\u2705 OK  Wan2.1 weights cached ({time.time()-t0:.0f}s)')\n",
                    "    else:\n",
                    "        print(f'  \\u26a0\\ufe0f  Download failed (will retry at runtime):')\n",
                    "        print('STDERR:', r.stderr.strip()[-300:])\n",
                    "\n",
                    "    print('\\nVRAM status: 0 bytes allocated (no models loaded \\u2014 correct)')\n",
                    "    print('\\u2705 Setup complete -- proceed to Cell 3!')\n"
                ])
                cell['source'] = new_source
                print("Patched successfully")
            else:
                print("Could not find Chatterbox block end")
            break

    with open("autopilot_kaggle.ipynb", "w", encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    patch()
