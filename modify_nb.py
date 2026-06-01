import json

path = r'C:\Users\rajat\Downloads\yt\autopilot_kaggle.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find cell 3 and cell 4 indices
cell3_idx = None
cell4_idx = None
for i, cell in enumerate(nb['cells']):
    if cell.get('id') == 'cell-3-keys':
        cell3_idx = i
    elif cell.get('id') == 'cell-4-run':
        cell4_idx = i

if cell3_idx is not None:
    # Modify cell 3 source
    source = nb['cells'][cell3_idx]['source']
    for i, line in enumerate(source):
        if line.startswith("    'AUDIO_PARALLEL_WORKERS'"):
            source[i] = "    'AUDIO_PARALLEL_WORKERS': '1',\n"
        elif line.startswith("    'VISUAL_PARALLEL_WORKERS'"):
            source[i] = "    'VISUAL_PARALLEL_WORKERS': '1',\n"
        elif line.startswith("    'VIDEO_GEN_ENABLED'"):
            source[i] = "    'VIDEO_GEN_ENABLED':          '1',\n"
        elif line.startswith("    'VIDEO_GEN_COG_STEPS'"):
            source[i] = "    'VIDEO_GEN_COG_STEPS':    '25',\n"
            # insert new variables here
            source.insert(i+1, "    'VIDEO_GEN_MODEL':        'cogvideox',\n")
            source.insert(i+2, "    'FFMPEG_PRESET':          'ultrafast',\n")

# Create new cell
new_cell = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "id": "cell-cleanup",
 "source": [
  "# =============================================================================\n",
  "# CLEANUP CACHE BEFORE RUNNING PIPELINE\n",
  "# =============================================================================\n",
  "import shutil, os, subprocess\n",
  "\n",
  "# Clear model cache of unused models\n",
  "cache_dirs_to_clean = [\n",
  "    \"/root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers\",\n",
  "    \"/root/.cache/huggingface/hub/models--Lightricks--LTX-Video\",\n",
  "]\n",
  "for d in cache_dirs_to_clean:\n",
  "    if os.path.exists(d):\n",
  "        shutil.rmtree(d)\n",
  "        print(f\"Deleted: {d}\")\n",
  "\n",
  "# Clear scratch outputs\n",
  "scratch = \"/kaggle/working/autopilot/autopilot_pipeline/outputs/video/scratch\"\n",
  "if os.path.exists(scratch):\n",
  "    shutil.rmtree(scratch)\n",
  "    os.makedirs(scratch)\n",
  "    print(\"Scratch cleared\")\n",
  "\n",
  "# Check disk\n",
  "result = subprocess.run([\"df\", \"-h\", \"/kaggle/working\"], capture_output=True, text=True)\n",
  "print(result.stdout)\n"
 ]
}

if cell4_idx is not None:
    nb['cells'].insert(cell4_idx, new_cell)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
