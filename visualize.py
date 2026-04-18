#!/usr/bin/env python3
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')

"""
AdaSpot – Soccer action detector + video visualizer.

Usage (run from the AdaSpot-main directory):
    python visualize.py --video_path D:/Data/1.mp4

Output:
  - Console table of detected actions with timestamps and confidence scores.
  - Annotated video (default: <input>_annotated.mp4 unless --output_path is set).
  - Optional: HR RoI crops as PNGs (--save_crops): default is every strided frame;
    use --crops_events_only to save only frames with a detection.

Optional flags:
  --output_path PATH       Override where the annotated video is saved.
  --save_crops             Save HR RoI crops (one per strided frame by default).
  --crops_events_only      With --save_crops, save only frames that have an event (fewer files).
  --crops_dir PATH         Folder for crops (default: <video_stem>_roi_crops).
  --threshold FLOAT        Score threshold (default 0.15).
  --batch_size INT         Inference batch size (default 2; reduce if OOM).
  --num_workers INT        DataLoader workers (default 0 for Windows).
"""

import argparse
import json
import torch
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from tabulate import tabulate

# ── Local imports ──────────────────────────────────────────────────────────────
from model.model import AdaSpot
from dataset.frame import ActionSpotInferenceDataset
from util.dataset import load_classes
from util.io import load_json
from util.constants import STRIDE_SNB, WINDOWS_SNB
from util.eval import soft_non_maximum_supression, process_frame_predictions_inference


# ── Fixed paths for the SoccerNetBall_big pretrained model ────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH     = os.path.join(_BASE, 'config', 'SoccerNetBall', 'SoccerNetBall_big.json')
CHECKPOINT_PATH = os.path.join(_BASE, 'config', 'pretrained', 'SoccernetBall_Big', 'checkpoint_best.pt')
CLASSES_PATH    = os.path.join(_BASE, 'data',   'soccernetball', 'class.txt')

# ── BGR colours for class legend / timeline (cycles with idx % len) ─────────
_COLORS_BGR = [
    (  0, 200, 255), (  0, 255, 100), (255, 100,   0), (255,   0, 200),
    (  0, 170, 255), (100, 255, 255), (255, 200,   0), (  0, 100, 255),
    (180,   0, 255), ( 50, 255,  50), (255, 128, 128), (128, 255,  64),
    ( 64, 128, 255), (255,  64, 192), (192, 255,  64), ( 64, 255, 192),
]


# ──────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(
        description='AdaSpot: detect and visualize soccer actions in a video clip'
    )
    p.add_argument('--video_path',   type=str,   default="D:/Data/224.mp4",
                   help='Path to the input soccer video (mp4 / avi / …)')
    p.add_argument('--output_path',  type=str,   default="D:/out.avi",
                   help='Output annotated video path (default: <input>_annotated.mp4)')
    p.add_argument('--threshold',    type=float, default=0.05,
                   help='Minimum confidence score to keep a detection (default 0.15)')
    p.add_argument('--batch_size',   type=int,   default=2,
                   help='Inference batch size (default 2)')
    p.add_argument('--num_workers',  type=int,   default=0,
                   help='DataLoader workers (default 0; keep 0 on Windows)')
    p.add_argument('--save_crops', action='store_false',
                   help='Save HR RoI crops (default: all strided frames; see --crops_events_only)')
    p.add_argument('--crops_events_only', action='store_true',
                   help='With --save_crops, save only strided frames that have a detection')
    p.add_argument('--crops_dir', type=str, default='D:/Data',
                   help='Directory for RoI crops (default: <video_stem>_roi_crops)')
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
def _dict_to_ns(d):
    """Recursively convert a dict to argparse.Namespace."""
    if isinstance(d, dict):
        import argparse as _ap
        return _ap.Namespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    return d


def load_model_and_classes():
    """Load config, class list, and pre-trained AdaSpot weights."""
    cfg      = load_json(CONFIG_PATH)
    classes  = load_classes(CLASSES_PATH)

    args_model    = _dict_to_ns(cfg['model'])
    args_training = _dict_to_ns(cfg['training'])
    args_model.clip_len     = cfg['data']['clip_len']
    args_model.dataset      = cfg['data']['dataset']
    args_model.num_classes  = cfg['data']['num_classes']
    args_model.lowres_loss  = cfg['training']['lowres_loss']
    args_model.highres_loss = cfg['training']['highres_loss']
    args_model.pretrained   = False   # weights come from checkpoint; skip download

    model = AdaSpot(
        args_model=args_model, args_training=args_training,
        classes=classes, elements=None
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading checkpoint from: {CHECKPOINT_PATH}')
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load(state)
    model.clean_modules()
    print(f'Model loaded on {device.upper()}.')

    return model, classes, cfg['data'], cfg['model']


# ──────────────────────────────────────────────────────────────────────────────
def run_inference(model, classes, video_path, data_cfg, model_cfg, threshold, batch_size, num_workers):
    """
    Slide a window over the video, aggregate scores, apply Soft-NMS, and
    return the final list of detected events.
    """
    clip_len = data_cfg['clip_len']
    stride   = STRIDE_SNB                       # 2 for soccernetball

    # cv2.resize expects (width, height); hr_dim is [height, width]
    hr_h, hr_w = model_cfg['hr_dim']
    frame_size = (hr_w, hr_h)                   # (width, height) for cv2

    inf_ds = ActionSpotInferenceDataset(
        video_path,
        clip_len    = clip_len,
        overlap_len = clip_len // 2,
        stride      = stride,
        dataset     = data_cfg['dataset'],
        size        = frame_size,
    )

    loader = DataLoader(
        inf_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = False,
    )

    num_classes = data_cfg['num_classes']
    video_frames = inf_ds._video_len
    pred_len     = video_frames // stride

    predictions = np.zeros((pred_len, num_classes + 1), np.float32)
    support     = np.zeros(pred_len,                    np.int32)

    print(f'\nRunning inference on: {video_path}')
    print(f'  Total frames: {video_frames}  |  Clip length: {clip_len}  |  Stride: {stride}')
    use_amp = torch.cuda.is_available()
    for frames, starts in tqdm(loader, desc='Inference'):
        _, batch_scores = model.predict(frames, use_amp=use_amp)
        for i in range(frames.shape[0]):
            scores = batch_scores[i]
            start  = starts[i].item()
            if start < 0:
                scores = scores[-start:, :]
                start  = 0
            end = start + scores.shape[0]
            if end >= pred_len:
                end    = pred_len
                scores = scores[:end - start, :]
            predictions[start:end, :] += scores
            support[start:end] += (scores.sum(axis=1) != 0).astype(np.int32)

    _, events_hr, _ = process_frame_predictions_inference(
        data_cfg['dataset'], classes, predictions, support,
        high_recall_score_threshold=threshold,
    )

    post = soft_non_maximum_supression(
        [{'events': events_hr}],
        window    = WINDOWS_SNB[1],
        threshold = threshold,
    )
    final_events = post[0]['events']

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    return final_events, fps, video_frames, stride


def _roi_tensor_to_bgr_uint8(roi_chw: torch.Tensor) -> np.ndarray:
    """Inverse ImageNet norm → uint8 BGR for cv2.imwrite (roi_chw: 3×H×W)."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    x = roi_chw.detach().float().cpu().numpy()
    x = x * std + mean
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    x = np.transpose(x, (1, 2, 0))
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def _safe_label_for_filename(label: str) -> str:
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in label.replace(' ', '_'))


def save_roi_crops(
    model,
    video_path: str,
    data_cfg: dict,
    model_cfg: dict,
    events: list,
    crops_dir: str,
    batch_size: int,
    num_workers: int,
    video_frames: int,
    stride: int,
    events_only: bool,
):
    """
    Second pass over the video: save HR RoI crops (after RoI selector) as PNGs.

    By default saves one crop per *strided* frame index in [0, pred_len), matching
    the model timeline (SoccerNet ball uses stride 2 on the video reader). This is
    one crop per strided timestep, not every raw video frame.

    If events_only=True, only strided frames that appear in ``events`` are saved.
    """
    clip_len = data_cfg['clip_len']
    hr_h, hr_w = model_cfg['hr_dim']
    frame_size = (hr_w, hr_h)
    pred_len = video_frames // stride

    os.makedirs(crops_dir, exist_ok=True)

    event_strided = {int(ev['frame']) for ev in events}
    if events_only and not event_strided:
        print('No events to export crops for (--crops_events_only).')
        return

    inf_ds = ActionSpotInferenceDataset(
        video_path,
        clip_len=clip_len,
        overlap_len=clip_len // 2,
        stride=stride,
        dataset=data_cfg['dataset'],
        size=frame_size,
    )
    loader = DataLoader(
        inf_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    ev_by_sf = {}
    for ev in sorted(events, key=lambda e: (int(e['frame']), -e['score'])):
        sf = int(ev['frame'])
        if sf not in ev_by_sf:
            ev_by_sf[sf] = ev

    saved_frames = set()
    meta = []
    use_amp = torch.cuda.is_available()

    mode = 'event frames only' if events_only else f'all strided frames (0 … {pred_len - 1})'
    print(f'\nSaving RoI crops to: {crops_dir}  ({mode})')

    for frames, starts in tqdm(loader, desc='RoI crops'):
        rois = model.predict_rois(frames, use_amp=use_amp)
        rois = rois.float()
        B, T = rois.shape[:2]
        for i in range(B):
            raw_start = starts[i].item()
            for t in range(T):
                global_sf = raw_start + t
                if global_sf < 0 or global_sf >= pred_len:
                    continue
                if events_only and global_sf not in event_strided:
                    continue
                if global_sf in saved_frames:
                    continue
                img = _roi_tensor_to_bgr_uint8(rois[i, t])
                ev = ev_by_sf.get(global_sf)
                if ev is not None:
                    label = _safe_label_for_filename(ev['label'])
                    name = f'crop_f{global_sf:06d}_{label}_{ev["score"]:.3f}.png'
                    meta.append({
                        'strided_frame': global_sf,
                        'orig_video_frame': int(global_sf * stride),
                        'label': ev['label'],
                        'score': ev['score'],
                        'file': name,
                    })
                else:
                    name = f'crop_f{global_sf:06d}.png'
                    meta.append({
                        'strided_frame': global_sf,
                        'orig_video_frame': int(global_sf * stride),
                        'label': None,
                        'score': None,
                        'file': name,
                    })
                path = os.path.join(crops_dir, name)
                cv2.imwrite(path, img)
                saved_frames.add(global_sf)

    meta.sort(key=lambda r: r['strided_frame'])
    with open(os.path.join(crops_dir, 'crops_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f'Saved {len(saved_frames)} RoI crop(s); manifest: {crops_dir}/crops_manifest.json')


# ──────────────────────────────────────────────────────────────────────────────
def print_results_table(events, fps, stride, class_names):
    rows = []
    for ev in sorted(events, key=lambda e: e['frame']):
        orig_frame = int(ev['frame']) * stride
        secs       = orig_frame / fps
        mm, ss     = divmod(secs, 60)
        rows.append([ev['label'], f'{int(mm):02d}:{ss:05.2f}', orig_frame, f"{ev['score']:.3f}"])

    print('\n' + '=' * 60)
    print('  DETECTED ACTIONS')
    print('=' * 60)
    if rows:
        print(tabulate(rows, headers=['Action', 'Time (mm:ss)', 'Frame', 'Score'],
                       tablefmt='rounded_outline'))
    else:
        print('  No actions detected above the threshold.')
    print()


# ──────────────────────────────────────────────────────────────────────────────
def _color_for(label, class_names):
    try:
        idx = class_names.index(label)
        return _COLORS_BGR[idx % len(_COLORS_BGR)]
    except ValueError:
        return (200, 200, 200)


def _draw_timeline(frame, events, cur_frame, total_frames, class_names, bar_h=28):
    """Render a thin event-timeline bar at the very bottom of the frame."""
    H, W = frame.shape[:2]
    bar_y = H - bar_h
    cv2.rectangle(frame, (0, bar_y), (W, H), (25, 25, 25), -1)

    for ev in events:
        x     = int(ev['frame_orig'] / total_frames * W)
        color = _color_for(ev['label'], class_names)
        cv2.line(frame, (x, bar_y), (x, H), color, 2)

    cx = int(cur_frame / total_frames * W)
    cv2.line(frame, (cx, bar_y), (cx, H), (255, 255, 255), 2)
    cv2.putText(frame, 'TIMELINE', (4, H - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1, cv2.LINE_AA)
    return frame


def _draw_legend(frame, class_names):
    """Small per-class colour legend in the top-right corner."""
    H, W = frame.shape[:2]
    fs, th = 0.40, 1
    pad, sq = 6, 12
    line_h  = sq + 4

    n   = len(class_names)
    bw  = 160
    bh  = n * line_h + pad * 2
    x0  = W - bw - pad
    y0  = pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 4, y0), (W - pad, y0 + bh), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, name in enumerate(class_names):
        color = _COLORS_BGR[i % len(_COLORS_BGR)]
        ry    = y0 + pad + i * line_h
        cv2.rectangle(frame, (x0, ry), (x0 + sq, ry + sq), color, -1)
        cv2.putText(frame, name, (x0 + sq + 4, ry + sq - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (220, 220, 220), th, cv2.LINE_AA)
    return frame


def annotate_video(video_path, output_path, events, fps, total_frames, stride, class_names):
    """
    Write an annotated copy of the video:
      - Coloured border + centred label on the event frame (±0.4 s flash).
      - Persistent colour-coded class legend (top-right).
      - Timeline bar at the bottom.
    """
    # Pre-compute original frame numbers once
    for ev in events:
        ev['frame_orig'] = int(ev['frame']) * stride

    flash_radius = int(fps * 0.4)

    cap    = cv2.VideoCapture(video_path)
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    frame_idx = 0
    print(f'Writing annotated video → {output_path}')
    pbar = tqdm(total=total_frames, desc='Rendering')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Events that should flash on this frame
        active = [ev for ev in events
                  if abs(frame_idx - ev['frame_orig']) <= flash_radius]

        # Coloured border
        if active:
            color     = _color_for(active[0]['label'], class_names)
            thickness = max(6, H // 55)
            cv2.rectangle(frame, (0, 0), (W, H - 29), color, thickness)

        # Centred action label(s)
        font_scale = H / 420.0
        font_thick = max(1, int(font_scale * 2))
        for k, ev in enumerate(active):
            text = f"{ev['label']}   {ev['score']:.2f}"
            color = _color_for(ev['label'], class_names)
            (tw, th_px), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX,
                                             font_scale, font_thick)
            tx = (W - tw) // 2
            ty = int(H * 0.14) + k * int(th_px * 2.4)
            # Drop shadow
            cv2.putText(frame, text, (tx + 2, ty + 2), cv2.FONT_HERSHEY_DUPLEX,
                        font_scale, (0, 0, 0), font_thick + 2, cv2.LINE_AA)
            cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX,
                        font_scale, color, font_thick, cv2.LINE_AA)

        frame = _draw_legend(frame, class_names)
        frame = _draw_timeline(frame, events, frame_idx, total_frames, class_names)

        out.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    print('Done.')


# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = get_args()

    base, _ = os.path.splitext(args.video_path)
    if args.output_path is None:
        args.output_path = base + '_annotated.mp4'

    if args.save_crops and args.crops_dir is None:
        args.crops_dir = base + '_roi_crops'

    # ── Load model ────────────────────────────────────────────────────────────
    model, classes, data_cfg, model_cfg = load_model_and_classes()
    class_names = list(classes.keys())

    # ── Inference ─────────────────────────────────────────────────────────────
    events, fps, total_frames, stride = run_inference(
        model, classes, args.video_path, data_cfg, model_cfg,
        threshold   = args.threshold,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    # ── Print table ───────────────────────────────────────────────────────────
    print_results_table(events, fps, stride, class_names)

    # ── Render video ──────────────────────────────────────────────────────────
    annotate_video(args.video_path, args.output_path,
                   events, fps, total_frames, stride, class_names)

    if args.save_crops:
        save_roi_crops(
            model, args.video_path, data_cfg, model_cfg, events,
            args.crops_dir, args.batch_size, args.num_workers,
            total_frames, stride, args.crops_events_only,
        )


if __name__ == '__main__':
    main()
