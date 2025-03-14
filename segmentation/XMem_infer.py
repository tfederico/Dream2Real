import gc
import os
import pdb
import sys
from tqdm import tqdm
import pathlib
curr_dir_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(os.path.join(curr_dir_path, "XMem"))
from model.network import XMem
from inference.inference_core import InferenceCore
from inference.data.mask_mapper import MaskMapper
from inference.interact.interactive_utils import image_to_torch, torch_prob_to_numpy_mask
from inference.interact.interaction import ClickInteraction

from segmentation.sam_seg import Segmentor

import yaml
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import imgviz
import open3d as o3d
import copy
from PIL import Image

class XMem_inference(object):
    def __init__(self, config_file=os.path.join(curr_dir_path, "XMem.yaml")):
        """Initialize XMem inference components.

        Args:
            config_file (str): Path to the configuration file.
        """
        super(XMem_inference, self).__init__()
        self.config = self.load_config(config_file)
        self.model_pth = self.config['model_pth']
        self.num_objects = self.config['num_objects']
        
        # Initialize attributes as None first
        self.processor = None
        self.sam_segmentor = None
        self.mapper = None
        
        # Initialize components
        self._init_components()
        
        print("XMem_inference initialized")

    def _init_components(self):
        """Initialize XMem components for inference."""
        try:
            torch.autograd.set_grad_enabled(False)
            network = XMem(self.config, self.model_pth).cuda().eval()
            self.mapper = MaskMapper()
            self.processor = InferenceCore(network, config=self.config)
            self.processor.set_all_labels(list(range(1, self.num_objects + 1)))
            self.first_mask_loaded = self.config['first_mask_loaded']
            self.size = self.config['size']
            self.sam_segmentor = Segmentor()
        except Exception as e:
            print(f"Error initializing XMem components: {e}")
            raise

    def load_config(self, config_file):
        """Load and process the configuration file.

        Args:
            config_file (str): Path to the configuration file.

        Returns:
            dict: Configuration settings as a dictionary.
        """
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # Expand relative paths.
        for key, value in config.items():
            if isinstance(value, str) and value.startswith('./'):
                config[key] = os.path.join(os.path.dirname(config_file), value)
        return config

    def resize_img(self, img, mask=False):
        """Resize an image while maintaining its aspect ratio.

        Args:
            img (numpy.ndarray): The image to resize.
            mask (bool): Boolean indicating if the image is a mask (affects interpolation method).

        Returns:
            numpy.ndarray: Resized image.
        """
        height, width = img.shape[:2]  # Get the current height and width of the image
        aspect_ratio = min(width, height)  # Determine the aspect ratio based on the smaller dimension

        # Calculate new dimensions while maintaining the aspect ratio
        new_width = (width * self.size) // aspect_ratio
        new_height = (height * self.size) // aspect_ratio

        # Resize the image only if the new dimensions differ from the original
        if (new_width, new_height) != (width, height):
            interpolation_method = cv2.INTER_NEAREST if mask else cv2.INTER_AREA  # Choose interpolation method based on mask
            img = cv2.resize(img, dsize=(new_width, new_height), interpolation=interpolation_method)  # Resize the image

        return img  # Return the resized image

    def inference(self, data):
        """Run inference on a single frame of RGB data.

        Args:
            data (dict): Dictionary containing 'rgb' and optional 'mask'.

        Returns:
            numpy.ndarray: The resulting mask after inference.
        """
        rgb = data['rgb']
        msk = data['mask']
        shape = rgb.shape[:2]

        if not self.first_mask_loaded:
            if msk is not None:
                self.first_mask_loaded = True
            else:
                # label the first frame
                print("label the first frame")
                raise NotImplementedError

        # Map possibly non-continuous labels to continuous ones
        if msk is not None:
            msk = self.resize_img(msk, mask=True)
            msk, labels = self.mapper.convert_mask(msk)
            msk = torch.Tensor(msk).cuda()
            self.processor.set_all_labels(list(self.mapper.remappings.values()))
        else:
            labels = None

        # preprocess
        rgb = self.resize_img(rgb)
        rgb, rgb_no_norm = image_to_torch(rgb)

        # Run the model on this frame
        prob = self.processor.step(rgb, msk, labels, end=False)

        # Upsample to original size if needed
        prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:, 0]

        # Probability mask -> index mask
        out_mask = torch_prob_to_numpy_mask(prob)
        return out_mask

    def segment(self, rgb_data, depth_data, out_dir=None, show=False, use_cache=False):
        """Segment a sequence of RGB images.

        Args:
            rgb_data (list): List of RGB images.
            depth_data (list): List of depth images.
            out_dir (str): Output directory for saving masks.
            show (bool): Whether to show visualization.
            use_cache (bool): Whether to use cached results.

        Returns:
            list: List of segmentation masks.
        """
        if use_cache:
            print("Using cached segmentations")
            refined_masks = []
            for i in range(len(rgb_data)):
                mask_dir = os.path.join(out_dir, "XMem_masks")
                refined_mask = cv2.imread(os.path.join(mask_dir, "rgb_%04d.png" % i))
                refined_mask = refined_mask[:, :, 0]
                refined_masks.append(refined_mask)
            return refined_masks

        refined_masks = []
        print('Segmenting frames...')
        for i in tqdm(range(len(rgb_data))):
            data = {}
            data['rgb'] = rgb_data[i].cpu().numpy()
            data['mask'] = None
            data['info'] = {}
            data['info']['frame'] = [i]
            data['info']['shape'] = data['rgb'].shape[:2]

            # only give the first frame a mask
            if i == 0:
                sam_masks = self.sam_segmentor.segment(data['rgb'], show_masks=show)
                scene_mask = self._integrate_masks(sam_masks)
                data['mask'] = scene_mask

            # inference
            with torch.no_grad():
                mask = self.inference(data)

            # eliminate the disconnected components
            refined_mask = mask
            refined_masks.append(refined_mask)

            # show visualised masks
            mask_vis = imgviz.label2rgb(refined_mask)
            if show:
                cv2.imshow("mask", mask_vis)
                cv2.waitKey(1)

            if out_dir is not None:
                # save visualised masks
                inst_dir = os.path.join(out_dir, "XMem_vis_masks")
                os.makedirs(inst_dir, exist_ok=True)
                cv2.imwrite(os.path.join(inst_dir, "rgb_%04d.png" % i), mask_vis * 255)

                # save real masks
                mask_dir = os.path.join(out_dir, "XMem_masks")
                os.makedirs(mask_dir, exist_ok=True)
                cv2.imwrite(os.path.join(mask_dir, "rgb_%04d.png" % i), refined_mask)

        return refined_masks

    def segment_associate(self, video_path, depth_data, T_WC_data, intrinsics,
                          out_dir, out_scene_bound_masks, scene_centre,
                          show=False, use_cache=False, debug=False):
        """Segment video frames with association.

        Args:
            video_path (str): Path to video frames.
            depth_data (list): List of depth images.
            T_WC_data (list): Camera poses.
            intrinsics (numpy.ndarray): Camera intrinsics.
            out_dir (str): Output directory for saving masks.
            out_scene_bound_masks (numpy.ndarray): Scene boundary masks.
            scene_centre (numpy.ndarray): Scene center point.
            show (bool): Whether to show visualization.
            use_cache (bool): Whether to use cached results.
            debug (bool): Whether to save debug info.

        Returns:
            list: List of processed masks.
        """
        if use_cache:
            print("Using cached segmentations")
            refined_masks = []
            for i in range(len(depth_data)):
                mask_dir = os.path.join(out_dir, "XMem_masks")
                refined_mask = cv2.imread(os.path.join(mask_dir, "rgb_%04d.png" % i))
                refined_mask = refined_mask[:, :, 0]
                refined_masks.append(refined_mask)
            return refined_masks

        # create output directories
        if out_dir is not None:
            inst_dir = os.path.join(out_dir, "XMem_vis_masks")
            mask_dir = os.path.join(out_dir, "XMem_masks")
            os.makedirs(inst_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            if debug:
                video_dir = os.path.join(out_dir, "XMem_video_masks")
                video_vis_dir = os.path.join(out_dir, "XMem_video_vis_masks")
                os.makedirs(video_dir, exist_ok=True)
                os.makedirs(video_vis_dir, exist_ok=True)

        # Associate the video frames with the rgb data
        from utils.associate import associate
        associate_list = associate(out_dir)

        files = os.listdir(video_path)
        files.sort()

        refined_masks = []
        print('Segmenting frames...')
        for i, file in enumerate(tqdm(files)):
            if i < associate_list[0]:
                continue
            data = {}
            rgb = cv2.imread(os.path.join(video_path, file)).astype(np.uint8)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            data['rgb'] = rgb
            data['mask'] = None
            data['info'] = {}
            data['info']['frame'] = [i]
            data['info']['shape'] = data['rgb'].shape[:2]

            # only give the first frame a mask
            if i == associate_list[0]:
                # Flip to be upright, so that more in-distribution for SAM.
                flipped_img = np.rot90(data['rgb'], 1)
                sam_masks = self.sam_segmentor.segment(flipped_img, show_masks=show,
                                                       scene_bound_mask=np.rot90(torch.logical_not(out_scene_bound_masks[0]).cpu().numpy()))

                scene_mask = self._integrate_masks(sam_masks)
                scene_mask = np.rot90(scene_mask, 3)
                data['mask'] = scene_mask

            # inference
            with torch.no_grad():
                mask = self.inference(data)

            if i in associate_list:
                index = associate_list.index(i)

                # eliminate the duplicated components
                pruned_mask = self._duplicate_prune(mask,
                                                    depth_data[index].cpu().numpy(),
                                                    T_WC_data[index].cpu().numpy(),
                                                    intrinsics,
                                                    scene_centre)

                # combined with out_scene_bound_masks
                refined_mask = np.where(out_scene_bound_masks[index].cpu().numpy() == 255, 255, pruned_mask)
                refined_masks.append(refined_mask)

                # show visualised masks
                mask_vis = imgviz.label2rgb(refined_mask.astype(int))
                if show:
                    cv2.imshow("mask", mask_vis)
                    cv2.waitKey(1)

                # save visualised masks
                cv2.imwrite(os.path.join(inst_dir, "rgb_%04d.png" % index), mask_vis * 255)
                # save real masks
                cv2.imwrite(os.path.join(mask_dir, "rgb_%04d.png" % index), refined_mask)

            if debug:
                # save video masks
                cv2.imwrite(os.path.join(video_dir, "rgb_%d.png" % i), mask)
                mask_ori = imgviz.label2rgb(mask)
                cv2.imwrite(os.path.join(video_vis_dir, "rgb_%d.png" % i), mask_ori * 255)

        return refined_masks
    
    def _duplicate_prune(self, mask, depth, T_WC, intrinsics, scene_centre):
        """
        Eliminate the duplicated components, keep the closest one to the scene center.
        :param mask: the mask to be refined
        :param depth: the depth image
        :param T_WC: the camera pose
        :param intrinsics: the camera intrinsics
        :param scene_centre: the scene centre
        :return:
        """
        refined_mask = np.zeros_like(mask)
        for i in np.unique(mask):
            if i == 0:
                continue

            curr_mask = np.zeros_like(mask)
            curr_mask[mask == i] = 255
            num_comps, comps_img = cv2.connectedComponents(curr_mask.astype(np.uint8))
            # cv2.imshow("curr_mask", curr_mask.astype(np.uint8))
            if num_comps > 2:
                min_distance = 10000
                keep_comp_mask = None
                for comp_idx in range(1, num_comps):
                    comp_mask = comps_img == comp_idx
                    comp_area = comp_mask.sum()
                    if comp_area < 200:
                        continue
                    comp_depth = depth.copy()
                    comp_depth[comp_mask == 0] = 0
                    # project depth to 3D points
                    depth_o3d = o3d.geometry.Image((comp_depth * 1000).astype(np.uint16))
                    T_cw = np.linalg.inv(T_WC)
                    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(depth.shape[0], depth.shape[1], intrinsics)
                    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, o3d_intrinsics, T_cw,
                                                                        project_valid_depth_only=True)
                    pcd_array = np.asarray(pcd.points)
                    avg_point = np.mean(pcd_array, axis=0)
                    distance = np.linalg.norm(avg_point - scene_centre)
                    if distance < min_distance:
                        keep_comp_mask = comp_mask
                        min_distance = distance
                if keep_comp_mask is not None:
                    refined_mask[keep_comp_mask] = i

                # cv2.imshow("comp_mask", keep_comp_mask.astype(np.uint8) * 255)
                # cv2.waitKey(0)
            else:
                refined_mask[comps_img == 1] = i

                # cv2.imshow("comp_mask", (comps_img == 1).astype(np.uint8) * 255)
                # cv2.waitKey(0)

        return refined_mask.astype(np.uint8)

    def free(self):
        """Safely free GPU memory"""
        try:
            if hasattr(self, 'sam_segmentor') and self.sam_segmentor is not None:
                self.sam_segmentor.free()
                self.sam_segmentor = None
            
            if hasattr(self, 'processor') and self.processor is not None:
                if hasattr(self.processor, 'network'):
                    self.processor.network.cpu()
                del self.processor
                self.processor = None
            
            if hasattr(self, 'mapper'):
                del self.mapper
                self.mapper = None
                
        except Exception as e:
            print(f"Warning: Error during XMem cleanup: {e}")
        finally:
            torch.cuda.empty_cache()

    def _integrate_masks(self, sam_masks):
        """Integrate a list of SAM masks into a single mask.

        This function takes multiple binary masks generated by the SAM (Segment Anything Model)
        and combines them into a single mask where each unique mask is assigned a unique index.

        Args:
            sam_masks (list): A list of binary masks from SAM segmentation, where each mask is a boolean array.

        Returns:
            numpy.ndarray: A single integrated mask where each region corresponds to a unique index.
        """
        # Initialize an output mask with the same shape as the first mask
        integrated_mask = torch.zeros_like(sam_masks[0], dtype=torch.uint8)

        # Iterate through each mask and assign a unique index
        for index, mask in enumerate(sam_masks):
            integrated_mask[mask] = index  # Assign the index to the corresponding mask region

        return integrated_mask.cpu().numpy()  # Return the integrated mask as a NumPy array




# def disconnected_prune(mask: np.array):
#     """
#     Eliminate the disconnected components, keep the largest one.
#     :param mask:
#     :param depth:
#     :return:
#     """
#     refined_mask = np.zeros_like(mask)
#     for i in np.unique(mask):
#         if i == 0:
#             # skip the background
#             continue

#         curr_mask = np.zeros_like(mask)
#         curr_mask[mask == i] = 255
#         num_comps, comps_img = cv2.connectedComponents(curr_mask.astype(np.uint8))
#         if num_comps > 2:
#             max_area = 0
#             keep_comp_mask = None
#             for comp_idx in range(1, num_comps):
#                 comp_mask = comps_img == comp_idx
#                 comp_area = comp_mask.sum()
#                 if comp_area < 200:
#                     continue
#                 if comp_area >= max_area:
#                     keep_comp_mask = comp_mask
#                     max_area = comp_area
#             if keep_comp_mask is not None:
#                 # if all components are too small, ignore this object
#                 refined_mask[keep_comp_mask] = i
#         else:
#             refined_mask[comps_img == 1] = i

#     return refined_mask.astype(np.uint8)