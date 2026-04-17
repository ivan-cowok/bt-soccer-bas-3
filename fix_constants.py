"""
Regenerates util/constants.py with correct GAMES_SNB list syntax from
data/my_league/train.json and data/my_league/val.json.
"""
import json, os, textwrap

ROOT = os.path.dirname(os.path.abspath(__file__))

def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

train_videos = [v['video'] for v in load(os.path.join(ROOT, 'data', 'my_league', 'train.json'))]
val_videos   = [v['video'] for v in load(os.path.join(ROOT, 'data', 'my_league', 'val.json'))]

def fmt_list(videos, indent=8):
    pad = ' ' * indent
    inner = ',\n'.join(f"{pad}    '{v}'" for v in videos)
    return f"[\n{inner},\n{pad}]"

constants_content = f"""\
'''
We define constants here that are used across the codebase.
'''

LABELS_SNB_PATH = 'E:\\\\Database\\\\44\\\\soccer_data'
STRIDE = 1
STRIDE_SNB = 1
EVAL_SPLITS = ['val']
OVERLAP = 0.9
F3SET_ELEMENTS = [2, 3, 3, 3, 7, 8, 2, 4]
DEFAULT_PAD_LEN = 5
FPS_SNB = 25

GAMES_SNB = {{
        'train'     : {fmt_list(train_videos)},
        'val'       : {fmt_list(val_videos)},
        'test'      : [],
        'challenge' : [],
        }}

# Evaluation
TOLERANCES = [0, 1, 2, 4]
TOLERANCES_SNB = [6, 12]
WINDOWS = [1, 2]
WINDOWS_SNB = [6, 12]
INFERENCE_BATCH_SIZE = 4
"""

out_path = os.path.join(ROOT, 'util', 'constants.py')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(constants_content)

print(f"Written {out_path}")
print(f"  train : {len(train_videos)} videos")
print(f"  val   : {len(val_videos)} videos")
print(f"  test  : [] (no test split)")

# Quick syntax check
import ast
with open(out_path, encoding='utf-8') as f:
    src = f.read()
ast.parse(src)
print("Syntax check: OK")
