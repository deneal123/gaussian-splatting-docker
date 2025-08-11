#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import math
import numpy as np
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from PIL import Image

def create_cubemap_cameras(scene_center, resolution=512):
    """
    Create 6 cameras for cubemap rendering positioned at scene center
    with 90-degree FOV facing the 6 cardinal directions.
    """
    fov = math.pi / 2  # 90 degrees in radians
    
    # Cubemap directions and their corresponding up vectors
    directions = {
        'posx': {'dir': np.array([1.0, 0.0, 0.0]), 'up': np.array([0.0, -1.0, 0.0])},   # +X right
        'negx': {'dir': np.array([-1.0, 0.0, 0.0]), 'up': np.array([0.0, -1.0, 0.0])},  # -X left
        'posy': {'dir': np.array([0.0, 1.0, 0.0]), 'up': np.array([0.0, 0.0, 1.0])},    # +Y up
        'negy': {'dir': np.array([0.0, -1.0, 0.0]), 'up': np.array([0.0, 0.0, -1.0])},  # -Y down
        'posz': {'dir': np.array([0.0, 0.0, 1.0]), 'up': np.array([0.0, -1.0, 0.0])},   # +Z forward
        'negz': {'dir': np.array([0.0, 0.0, -1.0]), 'up': np.array([0.0, -1.0, 0.0])}   # -Z back
    }
    
    cameras = {}
    
    for face_name, orientation in directions.items():
        direction = orientation['dir']
        up = orientation['up']
        
        # Calculate look_at point
        look_at = scene_center + direction
        
        # Calculate camera right vector
        right = np.cross(direction, up)
        right = right / np.linalg.norm(right)
        
        # Recalculate up to ensure orthogonality
        up = np.cross(right, direction)
        up = up / np.linalg.norm(up)
        
        # Create rotation matrix (world to camera)
        R = np.array([right, up, -direction])  # -direction because camera looks down -Z axis
        
        # Translation (camera position in world coordinates)
        T = -R @ scene_center
        
        # Create camera using the same interface as COLMAP cameras
        # Need to create a dummy image as PIL Image for the Camera constructor
        dummy_image = Image.new('RGB', (resolution, resolution), color='black')
        
        camera = Camera(
            (resolution, resolution),  # resolution tuple
            colmap_id=0,
            R=R,
            T=T,
            FoVx=fov,
            FoVy=fov,
            depth_params=None,
            image=dummy_image,  # PIL Image
            invdepthmap=None,
            image_name=f"cubemap_{face_name}",
            uid=hash(face_name),
            data_device="cuda"
        )
        
        cameras[face_name] = camera
    
    return cameras

def render_cubemap(model_path, iteration, scene, gaussians, pipeline, background, resolution=512):
    """
    Render cubemap views and save them as separate images.
    """
    # Calculate scene center (use camera extent center as approximation)
    scene_center = np.array([0.0, 0.0, 0.0])  # Default to origin
    
    # Try to get a better scene center from existing cameras
    if hasattr(scene, 'train_cameras') and 1.0 in scene.train_cameras:
        train_cams = scene.train_cameras[1.0]
        if len(train_cams) > 0:
            # Calculate average camera position as scene center approximation
            cam_positions = []
            for cam in train_cams:
                cam_pos = cam.camera_center.cpu().numpy()
                cam_positions.append(cam_pos)
            scene_center = np.mean(cam_positions, axis=0)
    
    print(f"Using scene center: {scene_center}")
    
    # Create output directory
    cubemap_path = os.path.join(model_path, "cubemap", "ours_{}".format(iteration))
    makedirs(cubemap_path, exist_ok=True)
    
    # Create cubemap cameras
    cubemap_cameras = create_cubemap_cameras(scene_center, resolution=resolution)
    
    # Store rendered images for potential atlas creation
    rendered_faces = {}
    
    # Render each face
    for face_name, camera in tqdm(cubemap_cameras.items(), desc="Rendering cubemap faces"):
        rendering = render(camera, gaussians, pipeline, background)["render"]
        
        # Save the rendered image
        output_path = os.path.join(cubemap_path, f"cubemap_{face_name}.png")
        torchvision.utils.save_image(rendering, output_path)
        print(f"Saved {face_name} face to {output_path}")
        
        # Store for atlas creation
        rendered_faces[face_name] = rendering
    
    # Create atlas layout (optional)
    create_cubemap_atlas(rendered_faces, cubemap_path)

def create_cubemap_atlas(rendered_faces, output_dir):
    """
    Create a single atlas image from the 6 cubemap faces in cross layout:
           [posy]
    [negx][posz][posx][negz]
           [negy]
    """
    try:
        # Get face size (assume all faces are same size)
        face_size = rendered_faces['posx'].shape[1]  # Height/width
        
        # Create atlas canvas (3x4 grid)
        atlas_height = face_size * 3
        atlas_width = face_size * 4
        
        # Create empty atlas
        atlas = torch.zeros(3, atlas_height, atlas_width, device=rendered_faces['posx'].device)
        
        # Layout faces in cross pattern
        # Row 0: [empty][posy][empty][empty]
        atlas[:, 0:face_size, face_size:face_size*2] = rendered_faces['posy']
        
        # Row 1: [negx][posz][posx][negz]
        atlas[:, face_size:face_size*2, 0:face_size] = rendered_faces['negx']
        atlas[:, face_size:face_size*2, face_size:face_size*2] = rendered_faces['posz']
        atlas[:, face_size:face_size*2, face_size*2:face_size*3] = rendered_faces['posx']
        atlas[:, face_size:face_size*2, face_size*3:face_size*4] = rendered_faces['negz']
        
        # Row 2: [empty][negy][empty][empty]
        atlas[:, face_size*2:face_size*3, face_size:face_size*2] = rendered_faces['negy']
        
        # Save atlas
        atlas_path = os.path.join(output_dir, "cubemap_atlas.png")
        torchvision.utils.save_image(atlas, atlas_path)
        print(f"Saved cubemap atlas to {atlas_path}")
        
    except Exception as e:
        print(f"Warning: Could not create atlas - {e}")
        print("Individual face images are still available")

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, cubemap : bool = False, cubemap_resolution : int = 512):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if cubemap:
            # Render cubemap instead of standard views
            render_cubemap(dataset.model_path, scene.loaded_iter, scene, gaussians, pipeline, background, cubemap_resolution)
        else:
            # Standard rendering mode
            if not skip_train:
                 render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp)

            if not skip_test:
                 render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--cubemap", action="store_true", help="Render cubemap views instead of COLMAP cameras")
    parser.add_argument("--cubemap_resolution", default=512, type=int, help="Resolution for each cubemap face (default: 512)")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.cubemap, args.cubemap_resolution)