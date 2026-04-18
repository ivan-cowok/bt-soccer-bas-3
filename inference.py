# timm 0.9+ may hit HuggingFace Hub when building the backbone; on Windows that
# can trigger 0xC0000005 (same mitigation as main.py).
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')

# Global imports
import argparse
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader

# Local imports
from util.io import load_json
from model.model import AdaSpot
from util.eval import inference
from dataset.frame import ActionSpotInferenceDataset
from util.dataset import load_classes, load_elements
from util.constants import STRIDE, STRIDE_SNB
from visualize import annotate_video, print_results_table

# Default weights after training (repo root = directory containing inference.py).
DEFAULT_CHECKPOINT = os.path.join(
    'checkpoints', 'MyLeague', 'MyLeague_finetune-1', 'checkpoint_epoch_0005.pt')


def get_args():
    #Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', type=str, default='MyLeague_finetune',
        help='Config stem: config/<Dataset>/<model_name>.json (e.g. MyLeague_finetune, SoccerNetBall_big).')
    parser.add_argument(
        '--video_path', type=str, default="D:/Data/23.mp4",
        help='Path to the video file (.mp4, etc.). Required unless you set VIDEO_PATH in the environment.')
    parser.add_argument('--inference_threshold', type=float, default=0.03, help='Threshold for inference')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument(
        '--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
        help='Path to a training checkpoint (.pt state dict). Relative paths are resolved from cwd.')
    parser.add_argument(
        '--no_visualize', action='store_true',
        help='Skip console table and annotated output video.')
    parser.add_argument(
        '--output_video', type=str, default=None,
        help='Where to save the annotated video (default: <input_video>_annotated.mp4).')
    
    return parser.parse_args()

def dict_to_namespace(d):
    if isinstance(d, dict):
        return argparse.Namespace(**{
            k: dict_to_namespace(v) for k, v in d.items()
        })
    return d

def update_args(args, config):

    #Update arguments with config file
    args.paths = dict_to_namespace(config['paths'])
    args.data = dict_to_namespace(config['data'])
    args.data.store_dir = args.paths.save_dir + '/store_data'
    args.paths.save_dir = os.path.join(args.paths.save_dir, args.model_name + '-' + str(args.seed)) # allow for multiple seeds
    args.data.frame_dir = args.paths.frame_dir
    args.data.save_dir = args.paths.save_dir
    args.training = dict_to_namespace(config['training'])
    args.model = dict_to_namespace(config['model'])
    args.model.clip_len = args.data.clip_len
    args.model.dataset = args.data.dataset
    args.model.num_classes = args.data.num_classes

    return args


def main(args):

    video_arg = args.video_path or os.environ.get('VIDEO_PATH')
    if not video_arg:
        raise SystemExit(
            'Pass the movie to analyze, e.g.\n'
            '  python inference.py --video_path C:/path/to/match.mp4\n'
            'or set VIDEO_PATH in the environment.')

    config_path = args.model_name.split('_')[0] + '/' + args.model_name + '.json'
    config = load_json(os.path.join('config', config_path))
    args = update_args(args, config)

    # Same rule as dataset/datasets.get_datasets: class list lives under data_dir
    # (e.g. data/my_league/class.txt), not under the logical dataset name (soccernetball).
    data_dir = getattr(args.data, 'data_dir', None) or os.path.join(
        'data', args.data.dataset)
    classes = load_classes(os.path.join(data_dir, 'class.txt'))
    if args.data.dataset == 'f3set':
        elements = load_elements(os.path.join(data_dir, 'elements.txt'))
    else:
        elements = None

    # No ImageNet download: weights come from checkpoint below.
    args.model.pretrained = False

    # Model
    model = AdaSpot(args_model=args.model, args_training=args.training, classes=classes, elements=elements)

    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.normpath(os.path.join(os.getcwd(), ckpt_path))

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f'Checkpoint not found:\n  {ckpt_path}\n'
            f'Pass --checkpoint path/to/your.pt (e.g. checkpoint_best.pt or checkpoint_epoch_0005.pt).')

    video_path = os.path.normpath(
        video_arg if os.path.isabs(video_arg) else os.path.join(os.getcwd(), video_arg))
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f'Video not found:\n  {video_path}')

    print('START INFERENCE')
    print('  checkpoint:', ckpt_path)
    print('  video:     ', video_path)
    model.load(torch.load(ckpt_path))
    model.clean_modules() # clean modules to remove unnecessary parameters for inference and speed up evaluation

    stride = STRIDE
    if args.data.dataset == 'soccernetball':
        stride = STRIDE_SNB

    inf_dataset_kwargs = {
        'clip_len': args.data.clip_len, 'overlap_len': args.data.clip_len // 2, 'stride': stride, 'dataset': args.data.dataset, 'size': (args.model.hr_dim[0], args.model.hr_dim[1])
    }

    inference_dataset = ActionSpotInferenceDataset(video_path, **inf_dataset_kwargs)

    nw = args.training.num_workers
    # pin_memory with num_workers=0 starts a CUDA thread on Windows and can crash (0xC0000005).
    inference_loader = DataLoader(
        inference_dataset, batch_size=args.training.batch_size,
        shuffle=False, num_workers=nw,
        pin_memory=(nw > 0), drop_last=False)

    events, stride_out, video_len = inference(
        model, inference_loader, classes, threshold=args.inference_threshold)

    class_names = list(classes.keys())
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()

    out_vid = None
    if not args.no_visualize:
        print_results_table(events, fps, stride_out, class_names)
        out_vid = args.output_video
        if not out_vid:
            root, _ext = os.path.splitext(video_path)
            out_vid = root + '_annotated.mp4'
        print('Writing annotated video →', out_vid)
        annotate_video(
            video_path, out_vid, events, fps, video_len, stride_out, class_names)

    print('CORRECTLY FINISHED INFERENCE STEP')
    print('  JSON:', os.path.normpath(os.path.join(os.getcwd(), 'inference_output', 'results_inference.json')))
    if out_vid is not None:
        print('  Video:', os.path.normpath(out_vid))


if __name__ == '__main__':
    main(get_args())