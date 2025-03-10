import gc
import numpy as np
import torch
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from vis_utils import visimg
from segmentation.seg_utils import *

import os
import pathlib
curr_dir_path = pathlib.Path(__file__).parent.absolute()
working_dir = os.path.join(curr_dir_path, '..')

import pdb


class Segmentor():
    def __init__(self, device="cuda:0"):
        # Initialize attributes as None first
        self.sam = None
        self.mask_generator = None
        self.device = device
        
        # Initialize SAM model
        self._init_sam()

    def _init_sam(self):
        """Initialize SAM model"""
        try:
            total_memory_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
            if total_memory_gb > 10:
                self.sam = sam_model_registry["vit_h"](
                    checkpoint=os.path.join(working_dir, "models/sam_vit_h_4b8939.pth")
                ).to(self.device)
            else:
                self.sam = sam_model_registry["vit_b"](
                    checkpoint=os.path.join(working_dir, "models/sam_vit_b_01ec64.pth")
                ).to(self.device)
                
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=48,
                pred_iou_thresh=0.95,
                stability_score_thresh=0.90,
                crop_n_layers=2,
                crop_n_points_downscale_factor=2,
                crop_nms_thresh=0.95,
                min_mask_region_area=120,
            )
        except Exception as e:
            print(f"Error initializing SAM model: {e}")
            raise

    def subpart_suppression(self, masks, threshold=0.1):
        # For any pair of objects, if (subpart_threshold) of one is inside the other, keep the other.
        remove_idxs = []
        for i in range(len(masks)):
            curr_mask = masks[i]
            curr_area = curr_mask.sum()
            for j in range(i + 1, len(masks)):
                other_mask = masks[j]
                other_area = other_mask.sum()
                intersection = (curr_mask & other_mask).sum()
                if intersection / curr_area > threshold or intersection / other_area > threshold:
                    # Remove the smaller one.
                    smaller_area_idx = i if curr_area < other_area else j
                    remove_idxs.append(smaller_area_idx)

        keep_idxs = [i for i in range(len(masks)) if i not in remove_idxs]
        masks = [masks[i] for i in keep_idxs]
        return masks

    def large_obj_suppression(self, masks, img, threshold=0.3):
        img_area = img.shape[0] * img.shape[1]
        masks = [mask for mask in masks if mask.sum() / img_area <= threshold]
        return masks

    def small_obj_suppression(self, masks, area_thresh=80, side_thresh=20):
        masks = [mask for mask in masks if mask.sum() >= area_thresh]
        masks = [mask for mask in masks if get_smallest_side(mask)[1] > side_thresh]
        return masks

    # Keeps only masks which are connected components (no multiple islands).
    # Dilate a bit first in case small gap between components but actually same object.
    def disconnected_components_suppression(self, masks, img):
        masks = [mask for mask in masks if cv2.connectedComponents(cv2.dilate(mask.cpu().numpy().astype(np.uint8), np.ones((5, 5), np.uint8)))[0] == 2]
        return masks

    def segment(self, img, show_masks=False, scene_bound_mask=None):
        print("SAM segmenting the image...")
        masks = self.mask_generator.generate(img) # img in HWC uint8 format. Seems like RGB (rather than BGR).

        # if show_masks:
        #     plt.figure(figsize=(20,20))
        #     plt.imshow(img)
        #     show_anns(masks)
        #     plt.axis('off')
        #     plt.show()

        masks = [torch.tensor(mask['segmentation']).to(self.device) for mask in masks]
        if scene_bound_mask is not None:
            scene_bound_mask = torch.tensor(scene_bound_mask.copy()).to(masks[0].device)
            for mask in masks:
                mask &= scene_bound_mask

        # Uncomment for debugging.
        # print(f'Number of masks from SAM after SAM post-proc + before our post-proc: {len(masks)}')
        # if os.path.exists('temp_masks'):
        #     os.system('rm -rf temp_masks')
        # os.mkdir('temp_masks')
        # for i, mask in enumerate(masks):
        #     cv2.imwrite(f'temp_masks/mask_{i:03}.png', mask.cpu().numpy().astype(np.uint8) * 255)

        masks = self.disconnected_components_suppression(masks, img)
        masks = self.large_obj_suppression(masks, img) # To remove bground objs.
        masks = self.subpart_suppression(masks)
        masks = self.small_obj_suppression(masks) # To remove small objs which cannot be grasped anyway.

        # Uncomment for debugging.
        # print(f'Number of masks from SAM after SAM post-proc + our post-proc: {len(masks)}')
        # if os.path.exists('temp_masks'):
        #     os.system('rm -rf temp_masks')
        # os.mkdir('temp_masks')
        # for i, mask in enumerate(masks):
        #     cv2.imwrite(f'temp_masks/mask_{i:03}.png', mask.cpu().numpy().astype(np.uint8) * 255)

        # Inflate object masks to remove shadows on background, which would influence inpainting of holes in background.
        inflation_factor = 1.6
        obj_masks_inflated = [rescale_mask(mask.cpu().numpy(), inflation_factor) for mask in masks]
        obj_masks_inflated = np.logical_or.reduce(obj_masks_inflated)
        obj_masks_inflated = torch.from_numpy(obj_masks_inflated).to(self.device)
        bground_mask = ~obj_masks_inflated
        masks.insert(0, bground_mask)
        print('SAM segmentation complete.')
        return masks

    # Inputs are torch tensors.
    # Output has alpha channel, so has shape (H, W, 4).
    # Returns a tight object image with no background. Used for rendering.
    def get_obj_img(self, img, obj_mask):
        obj_img = img

        row_has_obj = torch.any(obj_mask.view(obj_mask.shape[0], -1), dim=-1)
        rows_with_obj = torch.where(row_has_obj)[0]
        first_obj_row = rows_with_obj[0]
        last_obj_row = rows_with_obj[-1]

        col_has_obj = torch.any(obj_mask.permute(1, 0).reshape(obj_mask.shape[1], -1), dim=-1)
        cols_with_obj = torch.where(col_has_obj)[0]
        first_obj_col = cols_with_obj[0]
        last_obj_col = cols_with_obj[-1]

        obj_img = obj_img[first_obj_row:last_obj_row + 1, first_obj_col:last_obj_col + 1]

        # Add alpha channel.
        obj_img = torch.cat([obj_img, obj_mask.to(obj_img.device)[first_obj_row:last_obj_row + 1, first_obj_col:last_obj_col + 1].unsqueeze(-1) * 255], dim=-1)

        return obj_img

    def free(self):
        """Safely free GPU memory"""
        try:
            if hasattr(self, 'sam') and self.sam is not None:
                self.sam = self.sam.to('cpu')
                del self.sam
                self.sam = None
            
            if hasattr(self, 'mask_generator'):
                del self.mask_generator
                self.mask_generator = None
                
        except Exception as e:
            print(f"Warning: Error during SAM cleanup: {e}")
        finally:
            torch.cuda.empty_cache()


if __name__ == '__main__':
    img = cv2.imread("./data/3d/pool/manual-obj-render.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    segmentor = Segmentor()
    segmentor.segment(img, True)
