import copy
import os
from PIL import Image
import numpy as np
import torch
from caption import Captioner
from clip_scoring import optimise_pose_grid
from vis_utils import visimg, pastel_colors
from diffusers import StableDiffusionInpaintPipeline
from diffusion import inpaint
from torchvision.transforms.functional import pil_to_tensor
import pathlib
import sys
import open3d as o3d
import pybullet as p
import pybullet_planning as pp
import pdb
from segmentation.XMem_infer import XMem_inference
from vision_3d.geometry_utils import vis_cost_volume, vis_multiverse
from vision_3d.physics_utils import create_unsup_col_check, get_phys_models
from vision_3d.camera_info import INTRINSICS_REALSENSE_1280
from segmentation.sam_seg import get_thumbnail
from reconstruction.train_ngp import build_vis_model
from reconstruction.combined_rendering import renderer
from scene_model import ObjectModel, SceneModel, TaskModel
from vision_3d.pcd_visual_model import PointCloudRenderer, get_vis_pcds
from data_loader import d2r_dataloader
from cfg import Config
from termcolor import colored
import argparse
import json

os.nice(1) # We're here to run fast, not to make friends.

curr_dir_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(curr_dir_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
total_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
egi_gpu = total_memory_gb > 20

from lang.llm import HFLlama, OllamaLlama, OllamaDeepSeek, OpenAIAPI

class ImaginationEngine():
    """Imagination engine for generating task models from user instructions."""
    def __init__(self, cfg, embodied=False):
        self.embodied = embodied

        # Initialise configs
        self.cfg = cfg
        self.data_dir = cfg.data_dir
        self.use_phys = cfg.use_phys
        self.use_phys_tsdf = cfg.use_phys_tsdf
        self.lazy_phys_mods = cfg.lazy_phys_mods
        self.multi_view_captions = cfg.multi_view_captions
        self.use_cache_dynamic_masks = cfg.use_cache_dynamic_masks
        self.use_cache_segs = cfg.use_cache_segs
        self.use_cache_captions = cfg.use_cache_captions
        self.use_cache_phys = cfg.use_cache_phys
        self.use_cache_cam_poses = cfg.use_cache_cam_poses
        self.use_cache_renders = cfg.use_cache_renders
        self.use_cache_goal_pose = cfg.use_cache_goal_pose
        self.render_distractors = cfg.render_distractors
        self.spatial_smoothing = cfg.spatial_smoothing
        self.use_cache_vis = cfg.use_cache_vis
        self.use_vis_pcds = cfg.use_vis_pcds
        self.pcds_type = cfg.pcds_type
        self.render_cam_pose_idx = cfg.render_cam_pose_idx
        self.scene_type = cfg.scene_type
        self.topdown = cfg.scene_type in [0, 3]
        self.physics_only = cfg.physics_only
        self.single_view_idx = cfg.single_view_idx # Defaults to 0 if not specified in cfg.

        # Initialise datasets
        self.depths_gt = None

        # Allocate models to GPUs.
        self.captioner_device = "cuda:0"

        self.scene_model = None
        self.segmentor = None
        self.caption = cfg.caption
        self.inpaint = cfg.inpaint_holes
        self.visseg = cfg.visseg
        self.inpainter = None

        self.lang_model = None

        self.renderer = None
        # self.renderer = PointCloudRenderer()

        assert cfg.scene_centre is not None
        assert cfg.scene_phys_bounds is not None
        assert cfg.sample_res is not None
        self.scene_centre = cfg.scene_centre
        self.scene_phys_bounds = cfg.scene_phys_bounds  # Format: x_min, y_min, z_min, x_max, y_max, z_max
        self.sample_res = cfg.sample_res

        # Move model initialization to when they're needed
        self.segmentor = None
        self.captioner = None
        self.inpainter = None
        self.lang_model = None
        self.renderer = None

        # Add GPU memory management
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()  # Clear GPU cache at start

    def _init_segmentor(self):
        """Lazy initialization of segmentor"""
        if self.segmentor is None:
            self.segmentor = XMem_inference()

    def _init_captioner(self):
        """Lazy initialization of captioner"""
        if self.captioner is None and self.cfg.caption:
            self.captioner = Captioner(
                topdown=self.topdown, 
                device=self.captioner_device, 
                read_cache=self.use_cache_captions,
                cache_path=os.path.join(self.data_dir, 'captions.json')
            )

    def _init_inpainter(self):
        """Lazy initialization of inpainter"""
        if self.inpainter is None and self.cfg.inpaint_holes:
            self.inpainter = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                requires_safety_checker=False
            ).to(self.device)

    def _init_lang_model(self):
        """Lazy initialization of language model"""
        if self.lang_model is None:
            self.lang_model = self._choose_llm()
            
    def _choose_llm(self):
        """Choose language model based on config"""
        # TODO: Uncomment this when llm option is added to config
        # if self.cfg.llm == 'hfllama': 
        #     return HFLlama()
        # elif self.cfg.llm == 'ollama_llama':
        #     return OllamaLlama()
        # elif self.cfg.llm == 'ollama_deepseek':
        #     return OllamaDeepSeek()
        # elif self.cfg.llm == 'openai':
        #     return OpenAIAPI()
        return HFLlama()
    
    def prepare_scene_data(self, raw_data=None):
        """Prepare scene data by segmenting and captioning.
        
        Args:
            raw_data: Optional tuple of (rgbs, depths, raw_cam_poses)
            
        Returns:
            dict: Scene data including masks, objects, captions, etc.
        """
        print('Preparing scene data...')
        torch.cuda.empty_cache()

        # Load and segment data
        masks, num_objs, rgbs, depths, intrinsics = self._segment_scene(raw_data)

        # Get captions for objects
        captions, thumbnails = self._get_captions(num_objs, masks, rgbs)

        return {
            'masks': masks,
            'num_objs': num_objs,
            'rgbs': rgbs,
            'depths': depths,
            'intrinsics': intrinsics,
            'captions': captions,
            'thumbnails': thumbnails
        }

    def build_scene_model(self, scene_data=None):
        """Build scene model from prepared data.
        
        Args:
            scene_data: Dict containing scene data. If None, calls prepare_scene_data()
            
        Returns:
            None
        """
        print('Building scene model...')
        torch.cuda.empty_cache()

        # Get scene data if not provided
        if scene_data is None:
            scene_data = self.prepare_scene_data()

        # Get physical and visual models
        phys_models, init_poses, vis_models, opt_cam_poses = self._get_models(
            scene_data['masks'], 
            scene_data['num_objs'], 
            scene_data['depths'], 
            scene_data['intrinsics']
        )

        # Build final scene model
        self._build_scene_visual_models(
            scene_data['masks'], scene_data['num_objs'],
            scene_data['captions'], scene_data['thumbnails'],
            phys_models, init_poses, vis_models,
            scene_data['rgbs'], scene_data['depths'],
            opt_cam_poses, scene_data['intrinsics']
        )

    def _segment_scene(self, raw_data=None):
        """Segment the scene into objects.
        
        Args:
            raw_data: Optional tuple of (rgbs, depths, raw_cam_poses)
            
        Returns:
            tuple: (masks, num_objs, rgbs, depths, intrinsics)
        """
        self._init_segmentor()
        
        intrinsics = INTRINSICS_REALSENSE_1280
        dataloader = d2r_dataloader(self.cfg)
        rgbs, depths, raw_cam_poses = dataloader.load_rgbds() if raw_data is None else raw_data
        self.out_scene_bound_masks = dataloader.remove_background(
            intrinsics, self.scene_phys_bounds, 
            use_cache=self.use_cache_dynamic_masks
        )

        self.depths_gt = [depths[render_idx] for render_idx in self.render_cam_pose_idx]
        self.depths_gt = torch.stack(self.depths_gt, dim=0)

        # Segment scene
        video_path = os.path.join(self.data_dir, "seg_images")
        if os.path.exists(video_path):
            masks = self.segmentor.segment_associate(
                video_path, depths, dataloader.T_WC_data, intrinsics,
                self.data_dir, self.out_scene_bound_masks, self.scene_centre,
                show=self.visseg, use_cache=self.use_cache_segs, debug=False
            )
        else:
            masks = self.segmentor.segment(
                rgbs, depths, self.data_dir, 
                show=self.visseg, use_cache=self.use_cache_segs
            )

        # Free segmentor
        self.segmentor.free()
        self.segmentor = None
        torch.cuda.empty_cache()

        # Process masks
        masks = [torch.tensor(mask) for mask in masks]
        masks = torch.stack(masks, dim=0)

        # Count objects
        if 255 in torch.unique(masks):
            num_objs = torch.unique(masks).shape[0] - 1
        else:
            num_objs = torch.unique(masks).shape[0]

        return masks, num_objs, rgbs, depths, intrinsics

    def _get_models(self, masks, num_objs, depths, intrinsics):
        """Get physical and visual models for objects.
        
        Args:
            masks: Segmentation masks
            num_objs: Number of objects
            depths: Depth images
            intrinsics: Camera intrinsics
            
        Returns:
            tuple: (phys_models, init_poses, vis_models, opt_cam_poses)
        """
        if self.use_cache_cam_poses:
            print('Using cached optimised camera poses')
            opt_cam_poses = np.load(os.path.join(self.data_dir, 'opt_cam_poses.npy'))
        else:
            _, opt_cam_poses = build_vis_model(
                self.cfg, dynamic_time_extension=False, 
                render_distract=self.render_distractors
            )
        opt_cam_poses = [torch.tensor(pose) for pose in opt_cam_poses]

        if self.lazy_phys_mods:
            phys_models = [None] * num_objs
            init_poses = [None] * num_objs
        else:
            phys_models, init_poses = get_phys_models(
                depths, opt_cam_poses, intrinsics, masks, num_objs, 
                self.scene_phys_bounds,
                save_dir=os.path.join(self.data_dir, 'phys_mods/'),
                vis=not self.use_cache_phys, 
                use_cache=self.use_cache_phys,
                use_phys_tsdf=self.use_phys_tsdf
            )

        vis_models = [None] * num_objs
        return phys_models, init_poses, vis_models, opt_cam_poses

    def _get_captions(self, num_objs, masks, rgbs):
        """Get and aggregate captions for objects.
        
        Args:
            num_objs: Number of objects
            masks: Segmentation masks
            rgbs: RGB images
            
        Returns:
            tuple: (captions, thumbnails)
        """
        self._init_captioner()
        
        # Get individual captions
        all_captions, thumbnails = self.captioner.get_object_captions(
            num_objs, rgbs, masks, self.out_scene_bound_masks,
            topdown=self.topdown,
            multi_view=self.multi_view_captions,
            single_view_idx=self.single_view_idx
        )

        # Save raw captions
        all_captions_dict = {i: all_captions[i] for i in range(len(all_captions))}
        with open(os.path.join(self.data_dir, 'all_captions.json'), 'w') as f:
            json.dump(all_captions_dict, f)

        # Free captioner
        self.captioner.free()
        self._init_lang_model()

        # Aggregate captions
        captions = self.captioner.aggregate_captions(all_captions, self.lang_model, multi_view=self.multi_view_captions, silent=True)
        captions_dict = {i: captions[i] for i in range(len(captions))}
        with open(os.path.join(self.data_dir, 'captions.json'), 'w') as f:
            json.dump(captions_dict, f)

        # Free models
        del self.captioner
        self.captioner = None
        self.lang_model.free_memory()
        del self.lang_model
        self.lang_model = None
        torch.cuda.empty_cache()

        return captions, thumbnails

    def _build_scene_visual_models(self, masks, num_objs, captions, thumbnails, 
                            phys_models, init_poses, vis_models,
                            rgbs, depths, opt_cam_poses, intrinsics):
        """Build final scene model with visual models.
        
        Args:
            masks: Segmentation masks
            num_objs: Number of objects
            captions: Object captions
            thumbnails: Object thumbnails
            phys_models: Physical models
            init_poses: Initial poses
            vis_models: Visual models
            rgbs: RGB images
            depths: Depth images
            opt_cam_poses: Optimized camera poses
            intrinsics: Camera intrinsics
        """
        objs = []
        for obj_idx in range(num_objs):
            mask_idx = obj_idx
            obj = ObjectModel(
                captions[obj_idx], vis_models[obj_idx], 
                phys_models[obj_idx], init_poses[obj_idx], 
                thumbnails[obj_idx], mask_idx
            )
            objs.append(obj)

        self.scene_model = SceneModel(
            self.scene_centre, objs, objs[0], rgbs, depths, 
            opt_cam_poses, intrinsics, masks, self.scene_phys_bounds, 
            self.scene_type, device=device
        )

    def _determine_movable_obj(self, user_instr):
        """Determine which object is movable based on user instruction.

        Args:
            user_instr: user instruction string

        Returns:
            movable_obj: ObjectModel of movable object
            movable_idx: index of movable object in scene_model.objs

        """
        obj_captions = [obj.name for obj in self.scene_model.objs]
        movable_idx = self.lang_model.get_movable_obj_idx(user_instr, obj_captions)
        movable_obj = self.scene_model.objs[movable_idx]
        return movable_obj, movable_idx

    def _determine_relevant_objs(self, norm_caption, movable_obj_idx):
        """Determine which objects are relevant to the task based on the normalised caption.

        Relevant means not a distractor. For example, relevant objects could include the plate where apple is to be
        placed. Even though plate not movable.

        Args:
            norm_caption: normalised caption string
            movable_obj_idx: index of movable object in scene_model.objs

        Returns:
            relevant_objs: list of relevant ObjectModels

        """
        obj_captions = [obj.name for obj in self.scene_model.objs]
        relevant_idxs = self.lang_model.get_relevant_obj_idxs(norm_caption, obj_captions, movable_obj_idx)
        if len(relevant_idxs) == 0:
            raise RuntimeError(f'Error: None of the captioned objects were determined to be relevant.')
        relevant_objs = [self.scene_model.objs[idx] for idx in relevant_idxs]
        return relevant_objs

    def process_instruction(self, user_instr, goal_caption=None, norm_captions=None):
        """Process user instruction to determine goal caption, normalised captions, 
           movable and relevant objects.
        
        Args:
            user_instr: user instruction string
            goal_caption: goal caption string
            norm_captions: list of normalised caption strings
            
        Returns:
            tuple: (movable_obj, relevant_objs, goal_caption, norm_captions)
        """
        torch.cuda.empty_cache()
        self._init_lang_model()
        
        # Parse instruction if goal caption not provided
        if goal_caption is None:
            goal_caption, norm_caption = self.lang_model.parse_instr(user_instr)
            print(colored('Goal caption: ', 'green'), goal_caption)
            print(colored('Normalised caption: ', 'green'), norm_caption)
            norm_captions = [norm_caption]
            
        # Determine relevant objects
        movable_obj, movable_obj_idx = self._determine_movable_obj(user_instr)
        relevant_objs = self._determine_relevant_objs(goal_caption, movable_obj_idx)

        # Free language model
        self.lang_model.free_memory()
        del self.lang_model
        self.lang_model = None
        torch.cuda.empty_cache()

        return {
            'movable_obj': movable_obj,
            'relevant_objs': relevant_objs,
            'goal_caption': goal_caption,
            'norm_captions': norm_captions
        }

    def create_task_model(self, movable_obj, relevant_objs, user_instr, goal_caption, norm_captions):
        """Create task model with determined objects.
        
        Args:
            movable_obj: ObjectModel of movable object
            relevant_objs: list of relevant ObjectModels
            user_instr: user instruction string
            goal_caption: goal caption string
            norm_captions: list of normalised caption strings
            
        Returns:
            TaskModel: created task model
        """
        # Create physical models if using lazy loading
        if self.lazy_phys_mods:
            [bground_phys, movable_phys], [bground_init_pose, movable_init_pose] = TaskModel.create_lazy_phys_mods(
                self.scene_model, movable_obj, self.scene_phys_bounds,
                save_dir=os.path.join(self.data_dir, 'phys_mod/'), 
                embodied=self.embodied,
                vis=False, 
                use_cache=self.use_cache_phys, 
                use_phys_tsdf=self.use_phys_tsdf,
                use_vis_pcds=self.use_vis_pcds,
                single_view_idx=self.single_view_idx
            )

        # Create visual models
        movable_obj.vis_model = TaskModel.create_movable_vis_model(
            self.scene_model,
            movable_obj,
            self.out_scene_bound_masks,
            os.path.join(self.data_dir, 'movable_vis_mod/'),
            use_vis_pcds=self.use_vis_pcds,
            pcds_type=self.pcds_type,
            single_view_idx=self.single_view_idx,
            use_cache=self.use_cache_vis,
            data_dir=self.data_dir
        )

        task_bground_obj, task_bground_masks = TaskModel.create_task_relevant_obj(
            self.scene_model,
            movable_obj,
            relevant_objs,
            self.out_scene_bound_masks,
            os.path.join(self.data_dir, 'task_bground_vis_mod/'),
            use_vis_pcds=self.use_vis_pcds,
            pcds_type=self.pcds_type,
            single_view_idx=self.single_view_idx,
            render_distractors=self.render_distractors,
            use_cache=self.use_cache_vis,
            data_dir=self.data_dir
        )

        # Update models if using lazy loading
        if self.lazy_phys_mods:
            movable_obj.phys_model = movable_phys
            movable_obj.pose = movable_init_pose
            task_bground_obj.phys_model = bground_phys

        # Create and return task model
        return TaskModel(
            user_instr, goal_caption, norm_captions, 
            self.scene_model, movable_obj, task_bground_obj, 
            task_bground_masks, self.topdown
        )

    def dream_best_pose(self, task_model, vis_cost_vol=True):
        """Dream best pose for movable object based on task model.

        Args:
            task_model: TaskModel for the task
            vis_cost_vol: whether to visualise the cost volume

        Returns:
            best_pose: best pose for movable object
        """
        torch.cuda.empty_cache()

        # Initialize renderer only when needed
        if self.renderer is None and self.use_vis_pcds and not self.use_cache_goal_pose:
            self.renderer = PointCloudRenderer()

        # Setup physics checks and get initial pose
        movable_init_pose, phys_check = self._setup_physics_checks(task_model)

        # Find best pose and visualize results
        best_pose = self._find_best_pose(
            task_model, movable_init_pose, phys_check, vis_cost_vol
        )

        # Free renderer after use
        if self.renderer:
            del self.renderer
            self.renderer = None
            torch.cuda.empty_cache()

        return best_pose

    def _setup_physics_checks(self, task_model):
        """Setup physics checks for pose validation.
        
        Args:
            task_model: TaskModel for the task
            
        Returns:
            tuple: (movable_init_pose, phys_check)
                - movable_init_pose: initial pose of movable object
                - phys_check: function to check physics validity of poses
        """
        # Get the initial pose of the movable object from the task model
        movable_init_pose = task_model.movable_obj.pose

        # Define a function to compose multiple validity checks
        def compose_checks(checks):
            def composed_check(pose_batch, task_model, valid_so_far):
                # Clone the current validity state
                valid_so_far = valid_so_far.clone()
                # Iterate through each check and update the validity state
                for check in checks:
                    valid_so_far &= check(pose_batch, task_model, valid_so_far)
                return valid_so_far
            return composed_check

        # Check if physics validation is enabled and if cached renders are not being used
        if self.use_phys and not self.use_cache_renders:
            # Setting up the physics checking simulator
            pyb_planner = pp  # Reference to the planning module
            if not self.embodied:
                # Connect to the physics simulator without GUI if not in embodied mode
                pyb_planner.connect(use_gui=False)

            # Create a check for unsupervised collision detection and get handles for static and movable objects
            unsup_col_check, static_obj_handles, movable_handles = create_unsup_col_check(
                pyb_planner, task_model, self.sample_res,
                self.embodied, lazy_phys_mods=self.lazy_phys_mods
            )
            
            # Store the handles for later use
            self.static_phys_handles = static_obj_handles
            self.movable_phys_handle = movable_handles[0]
            
            # Define a shutdown function to disconnect the planner after checks
            shutdown_pyb = lambda pose_batch, task_model, valid_so_far: valid_so_far if pyb_planner.disconnect() else valid_so_far
            
            # Determine the appropriate physics check based on whether the system is embodied
            if self.embodied:
                phys_check = unsup_col_check  # Use unsupervised collision check
            else:
                # Compose checks for both unsupervised collision and shutdown
                phys_check = compose_checks([unsup_col_check, shutdown_pyb])
        else:
            # If physics checks are not needed, return a function that always returns valid
            all_valid = lambda pose_batch, task_model, valid_so_far: torch.ones(len(pose_batch), dtype=torch.bool)
            phys_check = all_valid

        # Return the initial pose and the validity check function
        return movable_init_pose, phys_check

    def _find_best_pose(self, task_model, movable_init_pose, phys_check, vis_cost_vol):
        """Find best pose for movable object and visualize results.
        
        Args:
            task_model: TaskModel for the task
            movable_init_pose: initial pose of movable object
            phys_check: function to check physics validity of poses
            vis_cost_vol: whether to visualize cost volume
            
        Returns:
            best_pose: best pose for movable object
        """
        self._choose_renderer(task_model)

        # Get best pose
        if self.use_cache_goal_pose:
            best_pose = torch.tensor(np.loadtxt(os.path.join(self.data_dir, 'goal_pose.txt'))).float().to(device)
            pose_batch = torch.tensor(np.loadtxt(os.path.join(self.data_dir, 'pose_batch.txt'))).float().to(device)
            pose_scores = torch.tensor(np.loadtxt(os.path.join(self.data_dir, 'pose_scores.txt'))).float().to(device)
            best_render = Image.open(os.path.join(self.data_dir, 'best_render.png'))
            best_render.show()
        else:
            best_pose, pose_batch, pose_scores = optimise_pose_grid(
                self.renderer, self.depths_gt, self.render_cam_pose_idx,
                task_model, self.data_dir, sample_res=self.sample_res,
                phys_check=phys_check, use_templates=False,
                scene_type=self.scene_type, use_vis_pcds=self.use_vis_pcds,
                use_cache_renders=self.use_cache_renders,
                smoothing=self.spatial_smoothing,
            )
            np.savetxt(os.path.join(self.data_dir, 'goal_pose.txt'), best_pose.cpu().numpy())
            np.savetxt(os.path.join(self.data_dir, 'pose_batch.txt'), pose_batch.cpu().numpy())
            np.savetxt(os.path.join(self.data_dir, 'pose_scores.txt'), pose_scores.cpu().numpy())

        # Visualize results if requested
        if vis_cost_vol and (self.use_cache_goal_pose or pose_scores is not None):
            self._visualize_results(
                task_model, pose_scores, pose_batch, 
                best_pose, movable_init_pose
            )

        return best_pose

    def _choose_renderer(self, task_model):
        # Choose renderer
        if self.use_vis_pcds and not self.use_cache_goal_pose:
            self.renderer = PointCloudRenderer()
        else:
            self.renderer = renderer(self.data_dir, task_model)

    def _visualize_results(self, task_model, pose_scores, pose_batch, best_pose, movable_init_pose):
        """Visualize pose optimization results.
        
        Args:
            task_model: TaskModel for the task
            pose_scores: scores for each pose
            pose_batch: batch of poses
            best_pose: selected best pose
            movable_init_pose: initial pose of movable object
        """
        tsdf_vis = True  # Else: VHACD
        
        if self.use_vis_pcds:
            bground_vis = task_model.task_bground_obj.vis_model
            bground_geoms = [bground_vis]
        elif self.lazy_phys_mods:
            if tsdf_vis:
                bground_geoms = [os.path.join(self.data_dir, 'phys_mod/mesh_concave_0.obj')]
            else:
                bground_geoms = [task_model.task_bground_obj.phys_model]
        else:
            bground_geoms = [obj.phys_model for obj in task_model.scene_model.objs 
                            if obj is not task_model.movable_obj]

        if tsdf_vis:
            movable_geom = os.path.join(self.data_dir, 'phys_mod/mesh_concave_1.obj')
        else:
            movable_geom = task_model.movable_obj.phys_model

        if not self.use_vis_pcds:
            vis_cost_volume(pose_scores, self.sample_res, pose_batch, bground_geoms)
            if not tsdf_vis:
                vis_multiverse(pose_scores, self.sample_res, pose_batch, 
                             bground_geoms, movable_geom, movable_init_pose)

        pose_transform = best_pose.cpu() @ movable_init_pose.inverse().cpu()
        
        if not self.use_vis_pcds:
            meshes = [o3d.io.read_triangle_mesh(phys_model) for phys_model in bground_geoms]
            meshes.append(o3d.io.read_triangle_mesh(movable_geom))
            for i, mesh in enumerate(meshes):
                mesh.compute_vertex_normals()
                col = pastel_colors[i % pastel_colors.shape[0]] / 255.0
                mesh.paint_uniform_color(col)
                if i == len(meshes) - 1:
                    mesh.transform(pose_transform.numpy())
            o3d.visualization.draw_geometries(meshes)

    def __del__(self):
        """Cleanup when object is destroyed"""
        # Free all models with proper error handling
        try:
            if self.segmentor:
                self.segmentor.free()
        except Exception as e:
            print(f"Warning: Error freeing segmentor during cleanup: {e}")
        finally:
            self.segmentor = None

        try:
            if self.captioner:
                self.captioner.free()
        except Exception as e:
            print(f"Warning: Error freeing captioner during cleanup: {e}")
        finally:
            self.captioner = None

        try:
            if self.inpainter:
                del self.inpainter
            if self.renderer:
                del self.renderer
        except Exception as e:
            print(f"Warning: Error during final cleanup: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str, help="Path to the .json config file")
    parser.add_argument("data_dir", type=str, help="Path to the data directory")
    parser.add_argument("user_instr", type=str, help="User instruction")
    parser.add_argument("--goal_caption", type=str, default=None, required=False, help="Goal caption (optional, by default inferred from user_instr)")
    parser.add_argument("--norm_captions", type=str, nargs='+', default=None, required=False, help="Normalising captions (optional, by default inferred from user_instr)")
    args = parser.parse_args()
    user_instr = args.user_instr
    goal_caption = args.goal_caption
    norm_captions = args.norm_captions

    cfg = Config(args.cfg_path, args.data_dir)
    if not os.path.exists(args.data_dir):
        raise ValueError("The specified data_dir does not exist.")

    assert not ((not cfg.use_cache_cam_poses) and cfg.use_cache_phys), "Cannot use new camera poses with old cached physics models. Disable use_cache_phys."
    assert not ((not cfg.use_cache_cam_poses) and cfg.use_cache_vis), "Cannot use new camera poses with old cached visual models. Disable use_cache_vis."
    assert not ((not cfg.use_cache_segs) and cfg.use_cache_captions), "Cannot use new segmentations with old cached captions. Disable use_cache_captions."
    if cfg.use_cache_renders:
        assert os.path.exists(os.path.join(args.data_dir, 'cb_render/')), "Cannot use cached renders since cb_render directory not yet created and renders not yet created. Disable use_cache_renders."

    if not egi_gpu:
        caption = False
        print(colored("Warning:", "red"), " setting caption to False automatically based on GPU availability")

    if not cfg.use_cache_segs:
        print(colored("Warning:", "red"), " about to delete and regenerate everything from segmentations onwards. Press Ctrl+C to cancel, or Enter to continue.")
        input()

    imagination = ImaginationEngine(cfg)

    imagination.build_scene_model()

    if goal_caption is not None:
        print('Using goal caption: ', goal_caption)
        print('Using normalising captions: ', norm_captions)
    task_model = imagination.create_task_model(user_instr, goal_caption=goal_caption, norm_captions=norm_captions)
    movable_best_pose = imagination.dream_best_pose(task_model)
    print(colored("Predicted pose for movable object:", "green"))
    print(movable_best_pose)
