"""
Build AdaSpot training data for my_league dataset.

Reads all Labels-ball.json under E:/Database/44/soccer_data,
performs a 90/10 train/val split (seeded), and writes:
  - data/my_league/class.txt
  - data/my_league/train.json
  - data/my_league/val.json
  - config/MyLeague/MyLeague_finetune.json

(No test split / test.json — validation only.)

Also patches util/constants.py with the correct LABELS_SNB_PATH and GAMES_SNB.
"""

import json, os, glob, random
from collections import Counter, defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
LABELS_ROOT   = r'E:\Database\44\soccer_data'
ADASPOT_ROOT  = os.path.dirname(os.path.abspath(__file__))
DATA_OUT_DIR  = os.path.join(ADASPOT_ROOT, 'data', 'my_league')
CFG_OUT_DIR   = os.path.join(ADASPOT_ROOT, 'config', 'MyLeague')
FRAME_DIR     = r'E:\Database\44\soccer_data_frames'   # where frames will live
SAVE_DIR      = os.path.join(ADASPOT_ROOT, 'checkpoints', 'MyLeague')

FPS           = 25          # assumed frame rate for your clips
BUFFER_SECS   = 2           # extra seconds added after last annotation
SEED          = 42
TRAIN_RATIO   = 0.9

# ── Collect all clips ─────────────────────────────────────────────────────────
all_jsons = sorted(glob.glob(os.path.join(LABELS_ROOT, '**', 'Labels-ball.json'), recursive=True))
print(f'Found {len(all_jsons)} Labels-ball.json files')

clips       = []   # list of dicts: {video, num_frames, labels_data}
label_count = Counter()
skipped     = []

for jpath in all_jsons:
    # Relative video path (forward slashes, no trailing slash)
    rel = os.path.relpath(jpath, LABELS_ROOT)           # my_league\2026-2027\xxx\Labels-ball.json
    rel = os.path.dirname(rel).replace('\\', '/')        # my_league/2026-2027/xxx

    with open(jpath, encoding='utf-8') as f:
        data = json.load(f)
    anns = data.get('annotations', [])
    if not anns:
        skipped.append(rel)
        continue

    # Estimate num_frames from max position in ms
    max_pos_ms  = max(int(a['position']) for a in anns)
    num_frames  = int((max_pos_ms / 1000 + BUFFER_SECS) * FPS)

    for a in anns:
        label_count[a.get('label', '?')] += 1

    clips.append({
        'video':      rel,
        'num_frames': num_frames,
        '_anns':      anns,
    })

print(f'Usable clips: {len(clips)}  |  Skipped (empty): {len(skipped)}')
print(f'Total annotations: {sum(label_count.values())}')
print('Label distribution:')
for lab, cnt in label_count.most_common():
    print(f'  {lab}: {cnt}')

# ── Class list (sorted by frequency, most common first) ──────────────────────
class_names = [lab for lab, _ in label_count.most_common()]
print(f'\nClasses ({len(class_names)}): {class_names}')

# ── 90/10 split (seeded) ─────────────────────────────────────────────────────
random.seed(SEED)
shuffled = clips[:]
random.shuffle(shuffled)
n_train   = round(len(shuffled) * TRAIN_RATIO)
train_clips = shuffled[:n_train]
val_clips   = shuffled[n_train:]
print(f'\nSplit -> train: {len(train_clips)}  val: {len(val_clips)}')

# ── Helper: strip internal field, write JSON ─────────────────────────────────
def to_json_entry(clip):
    return {'video': clip['video'], 'num_frames': clip['num_frames']}

def write_json(path, obj, pretty=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        else:
            json.dump(obj, f, ensure_ascii=False)

# ── Write data/my_league/ ────────────────────────────────────────────────────
os.makedirs(DATA_OUT_DIR, exist_ok=True)

# class.txt
class_txt = os.path.join(DATA_OUT_DIR, 'class.txt')
with open(class_txt, 'w', encoding='utf-8') as f:
    for name in class_names:
        f.write(name + '\n')
print(f'\nWrote {class_txt}')

# train.json / val.json only
write_json(os.path.join(DATA_OUT_DIR, 'train.json'),
           [to_json_entry(c) for c in train_clips])
write_json(os.path.join(DATA_OUT_DIR, 'val.json'),
           [to_json_entry(c) for c in val_clips])
obsolete_test = os.path.join(DATA_OUT_DIR, 'test.json')
if os.path.isfile(obsolete_test):
    os.remove(obsolete_test)
    print(f'Removed obsolete {obsolete_test}')
print(f'Wrote train.json ({len(train_clips)} videos)')
print(f'Wrote val.json  ({len(val_clips)} videos)')

# ── Write config/MyLeague/MyLeague_finetune.json ──────────────────────────────
os.makedirs(CFG_OUT_DIR, exist_ok=True)
config = {
    "paths": {
        "frame_dir": FRAME_DIR,
        "save_dir":  SAVE_DIR
    },
    "data": {
        "dataset":          "soccernetball",
        "num_classes":      len(class_names),
        "clip_len":         100,
        "epoch_num_frames": 200000,
        "mixup":            True,
        "store_mode":       "store"
    },
    "training": {
        "batch_size":       1,
        "num_epochs":       30,
        "warm_up_epochs":   3,
        "start_val_epoch":  5,
        "learning_rate":    0.0002,
        "only_test":        False,
        "criterion":        "map",
        "num_workers":      4,
        "lowres_loss":      True,
        "highres_loss":     True
    },
    "model": {
        "hr_dim":             [448, 796],
        "hr_crop":            [448, 796],
        "lr_dim":             [224, 398],
        "lr_crop":            [224, 398],
        "roi_size":           [112, 112],
        "feature_arch":       "rny004_gsf",
        "blocks_temporal":    [True, True, True, True],
        "aggregation":        "max",
        "temporal_arch":      "gru",
        "threshold":          0.0,
        "padding":            "replicate",
        "roi_channel_reduce": "mean_max",
        "roi_spatial_increase": 10,
        "roi_size_step":      28,
        "use_full_hr":        True,
        "use_cbam":           True
    }
}
cfg_path = os.path.join(CFG_OUT_DIR, 'MyLeague_finetune.json')
write_json(cfg_path, config, pretty=True)
print(f'\nWrote {cfg_path}')

# ── Patch util/constants.py ───────────────────────────────────────────────────
train_vids = [c['video'] for c in train_clips]
val_vids   = [c['video'] for c in val_clips]


def _fmt_py_list(videos, indent=8):
    pad = ' ' * indent
    inner = ',\n'.join(f"{pad}    {repr(v)}" for v in videos)
    return f"[\n{inner},\n{pad}]"


constants_path = os.path.join(ADASPOT_ROOT, 'util', 'constants.py')
new_constants = f'''\'\'\'
We define constants here that are used across the codebase.
\'\'\'

LABELS_SNB_PATH = {repr(LABELS_ROOT)}
STRIDE = 1
STRIDE_SNB = 1
EVAL_SPLITS = ['val']
OVERLAP = 0.9
F3SET_ELEMENTS = [2, 3, 3, 3, 7, 8, 2, 4]
DEFAULT_PAD_LEN = 5
FPS_SNB = {FPS}

GAMES_SNB = {{
        'train'     : {_fmt_py_list(train_vids)},
        'val'       : {_fmt_py_list(val_vids)},
        'test'      : [],
        'challenge' : [],
        }}

# Evaluation
TOLERANCES = [0, 1, 2, 4]
TOLERANCES_SNB = [6, 12]
WINDOWS = [1, 2]
WINDOWS_SNB = [6, 12]
INFERENCE_BATCH_SIZE = 4
'''

with open(constants_path, 'w', encoding='utf-8') as f:
    f.write(new_constants)
print(f'Patched {constants_path}')

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('DONE. Next steps:')
print('='*60)
print(f'1. Extract video frames to:  {FRAME_DIR}')
print( '   Naming: frame0.jpg, frame1.jpg, ... at 25 fps')
print( '   Folder per clip:  <FRAME_DIR>/<video_path>/')
print( '   (e.g., E:\\Database\\44\\soccer_data_frames\\my_league\\2026-2027\\0484fd09bc994720bf73cb35545ea9\\frame0.jpg)')
print( '2. Train:')
print(f'   python main.py --model_name MyLeague_finetune --seed 1')
print( '   (first run with store_mode="store", then switch to "load")')
print(f'3. Inference:')
print(f'   python visualize.py --video_path <your_clip.mp4>')
print( '   (update visualize.py CONFIG_PATH/CHECKPOINT_PATH if needed)')
