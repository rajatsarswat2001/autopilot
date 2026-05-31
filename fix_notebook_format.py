import json

with open("autopilot_kaggle.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        # If the source is a list of single characters, join it into a single string, then split by lines.
        # How to detect? If it's a list, and most elements are length 1 (except maybe '\n').
        if isinstance(cell['source'], list) and len(cell['source']) > 0:
            # Check if it was shredded
            if all(len(s) <= 2 for s in cell['source'][:50]):
                # It was shredded! Join everything back together
                full_text = "".join(cell['source'])
                # Then split into a list of lines with trailing newlines
                lines = [line + '\n' for line in full_text.split('\n')]
                # Fix the last line so it doesn't have an extra newline if it didn't originally
                if lines and lines[-1] == '\n' and not full_text.endswith('\n\n'):
                    lines.pop()
                cell['source'] = lines

with open("autopilot_kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook format fixed!")
