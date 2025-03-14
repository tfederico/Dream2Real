import gc
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import numpy as np
import cv2
import torch
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms.functional import rotate, affine, pil_to_tensor, InterpolationMode, resize
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from clip_text_templates import CLIP_TEMPLATES
from vision_3d.geometry_utils import spatially_smooth_heatmap
from scene_model import ObjectModel
import utils.accio2ngp as accio2ngp

import pdb

from vision_3d.obj_pose_opt import sample_poses_grid

os.environ["TOKENIZERS_PARALLELISM"] = "false" # Because visualisation causes fork and tokenizers complains.
device = torch.device("cuda:0")

CLIP_RES = 336


def optimise_pose_grid(renderer, depths_gt, render_cam_pose_idx, task_model, data_dir, 
                       sample_res=None, phys_check=None, use_templates=False, scene_type=0, 
                       use_vis_pcds=False, use_cache_renders=False, smoothing=True):

    if sample_res is None:
        sample_res = [40, 40, 1, 1, 1, 1]
    pose_batch = sample_poses_grid(task_model, sample_res, scene_type=scene_type)

    renders, valid_poses, valid_idxs = prepare_renders(use_cache_renders, data_dir, pose_batch, 
                                                       task_model, phys_check, use_vis_pcds, 
                                                       renderer, render_cam_pose_idx, depths_gt)

    task_model.free_visual_models() # To save memory for CLIP.

    print('Evaluating rendered images using CLIP...')
    model, processor = load_clip_model()

    captions, goal_caption, norm_captions = prepare_captions(task_model, use_templates)    
    
    print('Computing CLIP similarity score for each render...')
    logits = calculate_clip_scores(model, processor, captions, renders, use_templates, norm_captions)

    pose_scores, best_pose, best_render = get_best_pose_and_render(pose_batch.shape[0], valid_idxs, logits, 
                                                                   smoothing, sample_res, renders, valid_poses)
    

    best_render = Image.fromarray(best_render)
    best_render.save(os.path.join(data_dir, 'best_render.png'))
    # show best render
    best_render.show()

    return best_pose.view(4, 4), pose_batch, pose_scores

def prepare_renders(use_cache_renders, data_dir, pose_batch, task_model, phys_check, 
                    use_vis_pcds, renderer, render_cam_pose_idx, depths_gt):
    if use_cache_renders:
        renders, valid_poses, valid_idxs = load_cached_renders(data_dir, pose_batch)
    else:
        valid_poses, valid_idxs = run_pre_render_checks(pose_batch, task_model, phys_check)
        renders = render_images(renderer, valid_poses, render_cam_pose_idx, depths_gt, task_model, use_vis_pcds)

    # Clipifying images.
    renders = np.rot90(np.vstack(np.expand_dims(renders, axis=0)), k=1, axes=(1, 2))
    renders = np.split(renders, renders.shape[0], axis=0)
    renders = [render.squeeze(0) for render in renders]
    
    return renders, valid_poses, valid_idxs

def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    return model, processor

def prepare_captions(task_model, use_templates):
    goal_caption = task_model.goal_caption
    norm_captions = task_model.norm_captions
    if use_templates:
        captions = [template.format(goal_caption) for template in CLIP_TEMPLATES]
        if norm_captions is not None:
            templated_norm_captions = []
            for curr_norm_caption in norm_captions:
                templated_norm_captions += [template.format(curr_norm_caption) for template in CLIP_TEMPLATES]
            captions += templated_norm_captions
    else:
        captions = [goal_caption] if norm_captions is None else [goal_caption] + norm_captions
    return captions, goal_caption, norm_captions

def calculate_clip_scores(model, processor, captions, renders, use_templates, norm_captions):
    # Split into batches of images to fit into memory.
    # OPT: optimise memory efficiency to allow for larger batch size and faster computation.
    total_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    clip_batch_size = 128 if total_memory_gb > 20 else 32
    num_batches = int(torch.ceil(torch.tensor(len(renders) / clip_batch_size)).item())
    all_logits = []

    with torch.no_grad(): # Here to save memory, but may need grad elsewhere in method.
        with tqdm(total=len(renders)) as pbar:
            for i in range(num_batches):
                # OPT: do all text preproc at start.
                inputs = processor(text=captions, images=renders[i * clip_batch_size : (i + 1) * clip_batch_size], 
                                   return_tensors="pt", padding=True).to(device)

                new_pixel_values = inputs.pixel_values
                batch_outputs = model(pixel_values=new_pixel_values, 
                                      attention_mask=inputs.attention_mask, 
                                      input_ids=inputs.input_ids)
                batch_logits = batch_outputs.logits_per_image # Has shape (num_imgs, num_captions).
                all_logits.append(batch_logits)
                pbar.update(clip_batch_size if i < num_batches - 1 else len(renders) - (i * clip_batch_size))
            pbar.close()
            all_logits = torch.cat(all_logits, dim=0).to('cpu')

        if use_templates:
            if norm_captions is None:
                # Average similarities across templates.
                logits = all_logits.mean(dim=1)
            else:
                num_templates = len(CLIP_TEMPLATES)
                goal_logits = all_logits[:, :num_templates].mean(dim=1) # Average across templates.
                norm_logits = all_logits[:, num_templates:].mean(dim=1) # Average across templates and normalising captions.
                logits = goal_logits / norm_logits
        else:
            if norm_captions is None:
                logits = all_logits.squeeze(-1)
            else:
                # Normalise by normalising caption.
                goal_logits = all_logits[:, 0]
                norm_logits = all_logits[:, 1:].mean(dim=1)
                logits = goal_logits / norm_logits

    return logits

def get_best_pose_and_render(n_pos_scores, valid_idxs, logits, smoothing, sample_res, renders, valid_poses):
    pose_scores = torch.zeros(n_pos_scores)
    pose_scores[valid_idxs] = logits
    # We want to create an array render_idxs of same shape as pose_scores which maps each valid pose back to corresponding index in renders.
    render_idxs = torch.zeros(pose_scores.shape[0], dtype=torch.long)
    render_idxs[valid_idxs] = torch.arange(valid_idxs.shape[0])

    # Apply spatial convolution / smoothing / filtering to remove outlier high-scoring poses surrounded by many low-scoring neighbours.
    if smoothing:
        print('Applying spatial smoothing...')
        with torch.no_grad():
            pose_scores = spatially_smooth_heatmap(pose_scores, sample_res)
        print('Done smoothing.')

    best_pose_idx = torch.argmax(pose_scores).item()
    best_render = renders[render_idxs[best_pose_idx]]
    best_pose = valid_poses[render_idxs[best_pose_idx]]
    return pose_scores, best_pose, best_render

def load_cached_renders(data_dir, pose_batch):
    print('Using cached renders')
    old_pose_scores = torch.from_numpy(np.loadtxt(os.path.join(data_dir, 'pose_scores.txt')))
    is_valid = old_pose_scores.bool()
    valid_idxs = torch.nonzero(old_pose_scores).squeeze(-1)
    valid_poses = pose_batch[valid_idxs]

    render_dir = os.path.join(data_dir, 'cb_render')
    print('Reading renders from disk...')
    renders = []
    for filename in tqdm(sorted(os.listdir(render_dir))):
        render_path = os.path.join(render_dir, filename)
        render = cv2.imread(render_path)
        render = cv2.cvtColor(render, cv2.COLOR_BGR2RGB)
        renders.append(render)
    assert len(renders) == valid_poses.shape[0], f'Expected {valid_poses.shape[0]} renders, got {len(renders)}. Try running without use_cache_renders.'
    
    return renders, valid_poses, valid_idxs

def run_pre_render_checks(pose_batch, task_model, phys_check):
    print('Running pre-render checks...')
    valid_so_far = torch.ones(pose_batch.shape[0]).bool().to(device)
    is_valid = phys_check(pose_batch, task_model, valid_so_far)
    valid_idxs = torch.nonzero(is_valid).squeeze(-1)
    valid_poses = pose_batch[valid_idxs]
    print(f'Of {pose_batch.shape[0]} sampled poses, {valid_idxs.shape[0]} passed pre-render checks ({100 * valid_idxs.shape[0] / pose_batch.shape[0]:.2f}%).')

    if valid_idxs.shape[0] == 0:
        print('No poses passed pre-render checks. Exiting.')
        raise Exception

    return valid_poses, valid_idxs

def render_images(renderer, valid_poses, render_cam_pose_idx, depths_gt, task_model, use_vis_pcds):
    from vision_3d.virtual_cam_pose_sample import get_virtual_cam_poses
    render_poses = get_virtual_cam_poses(task_model, render_cam_pose_idx)
    
    if use_vis_pcds:
        print('Rendering images from pcds...')
        renders = renderer.render(render_poses[0], valid_poses, task_model, hide_movable=False)
    else:
        print('Rendering images from ngp...')
        render_poses_ngp = accio2ngp.converter(render_poses)
        valid_poses_ngp = accio2ngp.converter(valid_poses.cpu().numpy().reshape(-1, 4, 4))
        renders = renderer.render(valid_poses_ngp,
                                  render_poses_ngp,
                                  render_cam_pose_idx,
                                  depths_gt,
                                  task_model.movable_masks,
                                  save=True)
    return renders