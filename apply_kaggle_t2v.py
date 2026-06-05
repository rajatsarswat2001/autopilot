import json

with open('final.ipynb', 'r', encoding='utf-8') as f:
    d = json.load(f)

for c in d['cells']:
    if 'source' not in c: continue
    src = "".join(c['source'])
    
    # Update Cell 2
    if 'CELL 2 -- CLONE COMFYUI & DOWNLOAD WAN 2.2' in src:
        src = src.replace('wan2.2_ti2v', 'wan2.2_t2v')
        src = src.replace('TI2V', 'T2V')
        
        # Remove clip_vision line
        lines = src.split('\n')
        new_lines = []
        for line in lines:
            if 'clip_vision_h.safetensors' in line: continue
            new_lines.append(line)
        src = '\n'.join(new_lines)
        
        c['source'] = [line + '\n' for line in src.split('\n')]
        if c['source']: c['source'][-1] = c['source'][-1].rstrip('\n')

    # Update Cell 5
    if 'CELL 5 -- LAUNCH DUAL-GPU WAN 2.2' in src:
        # We need to rewrite the gen() function inside worker_code
        
        # Rewrite the nodes
        old_nodes = '''        loaded_image, clip_vis_out = None, None
        if image and image.filename:
            img_path = f"{INPUT_DIR}/{image.filename}"
            with open(img_path, "wb") as fh: fh.write(await image.read())
            loaded_image = NODE_CLASS_MAPPINGS["LoadImage"]().load_image(img_path)[0]
            cv = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]().load_clip("clip_vision_h.safetensors")[0]
            clip_vis_out = NODE_CLASS_MAPPINGS["CLIPVisionEncode"]().encode(cv, loaded_image, "none")[0]
            del cv; gc.collect(); torch.cuda.empty_cache()

        vae = NODE_CLASS_MAPPINGS["VAELoader"]().load_vae("wan2.2_vae.safetensors")[0]
        wan_cls = NODE_CLASS_MAPPINGS.get("WanImageToVideo")
        if loaded_image is not None and wan_cls:
            pos_c, neg_c, lat = wan_cls().encode(pos_c, neg_c, vae, width, height, frames, 1, loaded_image, clip_vis_out)
        else:
            lat = NODE_CLASS_MAPPINGS["EmptyLatentImage"]().generate(width, height, 1)[0]'''
        
        new_nodes = '''        vae = NODE_CLASS_MAPPINGS["VAELoader"]().load_vae("wan2.2_vae.safetensors")[0]
        lat = NODE_CLASS_MAPPINGS["EmptyLatentImage"]().generate(width, height, 1)[0]'''

        if old_nodes in src:
            src = src.replace(old_nodes, new_nodes)
        
        c['source'] = [line + '\n' for line in src.split('\n')]
        if c['source']: c['source'][-1] = c['source'][-1].rstrip('\n')

with open('final.ipynb', 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2)

print("final.ipynb T2V architecture applied!")
