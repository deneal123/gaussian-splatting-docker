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

def create_panorama_cameras(scene_center, width=2048, height=1024, camera_fov=70):
    """
    Create cameras for equidistant panorama rendering positioned at scene center.
    Cameras are distributed across the sphere to capture the full 360x180 degree view.
    """
    fov = math.radians(camera_fov)  # Convert FOV to radians
    
    # Calculate number of cameras needed based on FOV and desired overlap
    # We want some overlap between adjacent cameras for better blending
    overlap_factor = 0.8  # 20% overlap between cameras
    cameras_horizontal = max(4, int(2 * math.pi / (fov * overlap_factor)))
    cameras_vertical = max(2, int(math.pi / (fov * overlap_factor)))
    
    cameras = []
    camera_id = 0
    
    # Create cameras distributed across the sphere
    for v_idx in range(cameras_vertical):
        for h_idx in range(cameras_horizontal):
            # Calculate spherical coordinates
            theta = (h_idx / cameras_horizontal) * 2 * math.pi  # Azimuth: 0 to 2π
            phi = ((v_idx + 0.5) / cameras_vertical - 0.5) * math.pi  # Elevation: -π/2 to π/2
            
            # Convert spherical to Cartesian direction
            direction = np.array([
                math.cos(phi) * math.cos(theta),  # X
                math.sin(phi),                    # Y  
                math.cos(phi) * math.sin(theta)   # Z
            ])
            
            # Calculate up vector (generally point towards north pole, adjusted for camera orientation)
            if abs(phi) > math.pi/2 - 0.1:  # Near poles
                up = np.array([0.0, 0.0, 1.0]) if phi > 0 else np.array([0.0, 0.0, -1.0])
            else:
                # For most cameras, up points toward the north pole
                up = np.array([0.0, 1.0, 0.0])
            
            # Calculate camera right vector
            right = np.cross(direction, up)
            if np.linalg.norm(right) < 1e-6:  # Handle degenerate case
                right = np.array([1.0, 0.0, 0.0])
            right = right / np.linalg.norm(right)
            
            # Recalculate up to ensure orthogonality
            up = np.cross(right, direction)
            up = up / np.linalg.norm(up)
            
            # Create rotation matrix (world to camera)
            R = np.array([right, up, -direction])  # -direction because camera looks down -Z axis
            
            # Translation (camera position in world coordinates)
            T = -R @ scene_center
            
            # Create dummy image for Camera constructor
            dummy_image = Image.new('RGB', (512, 512), color='black')
            
            camera = Camera(
                (512, 512),  # Fixed resolution for each camera
                colmap_id=camera_id,
                R=R,
                T=T,
                FoVx=fov,
                FoVy=fov,
                depth_params=None,
                image=dummy_image,
                invdepthmap=None,
                image_name=f"panorama_cam_{camera_id:03d}",
                uid=camera_id,
                data_device="cuda"
            )
            
            # Store camera with its spherical coordinates for later projection
            camera.theta = theta
            camera.phi = phi
            cameras.append(camera)
            camera_id += 1
    
    print(f"Created {len(cameras)} cameras for panorama ({cameras_horizontal}x{cameras_vertical} grid)")
    return cameras

def project_to_equirectangular(rendered_views, cameras, output_width, output_height):
    """
    Project multiple rendered camera views into a single equirectangular panorama.
    Optimized vectorized version for better performance.
    
    Args:
        rendered_views: List of rendered images from cameras
        cameras: List of camera objects with theta, phi attributes
        output_width: Width of output panorama (e.g., 2048)
        output_height: Height of output panorama (e.g., 1024)
    
    Returns:
        Single panoramic image tensor
    """
    device = rendered_views[0].device
    panorama = torch.zeros(3, output_height, output_width, device=device)
    weight_map = torch.zeros(output_height, output_width, device=device)
    
    # Create coordinate meshgrids for the output panorama
    y_coords, x_coords = torch.meshgrid(
        torch.arange(output_height, device=device, dtype=torch.float32),
        torch.arange(output_width, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # Convert panorama coordinates to spherical coordinates
    u = x_coords / output_width   # 0 to 1
    v = y_coords / output_height  # 0 to 1
    
    theta = u * 2 * math.pi - math.pi  # -π to π (azimuth)
    phi = (0.5 - v) * math.pi          # -π/2 to π/2 (elevation)
    
    # Convert to 3D direction vectors
    directions = torch.stack([
        torch.cos(phi) * torch.cos(theta),  # X
        torch.sin(phi),                     # Y
        torch.cos(phi) * torch.sin(theta)   # Z
    ], dim=0)  # Shape: [3, H, W]
    
    for camera, rendered_view in zip(cameras, rendered_views):
        # Camera direction vector
        camera_direction = torch.tensor([
            math.cos(camera.phi) * math.cos(camera.theta),
            math.sin(camera.phi),
            math.cos(camera.phi) * math.sin(camera.theta)
        ], device=device, dtype=torch.float32)
        
        # Calculate angular distance for all pixels at once
        dot_products = torch.sum(directions * camera_direction.view(3, 1, 1), dim=0)
        dot_products = torch.clamp(dot_products, -1.0, 1.0)
        angular_distances = torch.acos(dot_products)
        
        # Create mask for pixels within camera FOV
        fov_mask = angular_distances <= (camera.FoVx / 2)
        
        if not fov_mask.any():
            continue  # No pixels from this camera contribute
        
        # Transform directions to camera space for pixels within FOV
        R = torch.tensor(camera.R, device=device, dtype=torch.float32)
        directions_cam = torch.einsum('ij,jhw->ihw', R, directions)  # Shape: [3, H, W]
        
        # Project to camera image coordinates using pinhole projection
        # Only process pixels that are within FOV and in front of camera
        front_mask = directions_cam[2] < 0  # Points in front of camera (negative Z)
        valid_mask = fov_mask & front_mask
        
        if not valid_mask.any():
            continue
        
        # Convert to normalized device coordinates
        focal_length = 0.5 / math.tan(camera.FoVx / 2)
        x_ndc = directions_cam[0] / (-directions_cam[2])
        y_ndc = directions_cam[1] / (-directions_cam[2])
        
        # Convert to pixel coordinates
        camera_height, camera_width = rendered_view.shape[1], rendered_view.shape[2]
        cam_x = (x_ndc / focal_length + 1) * camera_width / 2
        cam_y = (y_ndc / focal_length + 1) * camera_height / 2
        
        # Check bounds
        bounds_mask = (cam_x >= 0) & (cam_x < camera_width) & (cam_y >= 0) & (cam_y < camera_height)
        final_mask = valid_mask & bounds_mask
        
        if not final_mask.any():
            continue
        
        # Extract valid coordinates
        valid_y, valid_x = torch.where(final_mask)
        valid_cam_x = cam_x[final_mask]
        valid_cam_y = cam_y[final_mask]
        
        # Bilinear interpolation
        x0 = torch.floor(valid_cam_x).long()
        y0 = torch.floor(valid_cam_y).long()
        x1 = torch.clamp(x0 + 1, 0, camera_width - 1)
        y1 = torch.clamp(y0 + 1, 0, camera_height - 1)
        
        wx = valid_cam_x - x0.float()
        wy = valid_cam_y - y0.float()
        
        # Sample the rendered view with bilinear interpolation
        pixel_values = (
            rendered_view[:, y0, x0] * (1 - wx).unsqueeze(0) * (1 - wy).unsqueeze(0) +
            rendered_view[:, y0, x1] * wx.unsqueeze(0) * (1 - wy).unsqueeze(0) +
            rendered_view[:, y1, x0] * (1 - wx).unsqueeze(0) * wy.unsqueeze(0) +
            rendered_view[:, y1, x1] * wx.unsqueeze(0) * wy.unsqueeze(0)
        )
        
        # Calculate weights based on distance from camera center
        valid_angular_distances = angular_distances[final_mask]
        weights = torch.cos(valid_angular_distances / camera.FoVx * math.pi / 2)
        weights = torch.clamp(weights, min=0)  # Ensure non-negative
        
        # Accumulate contributions
        panorama[:, valid_y, valid_x] += pixel_values * weights.unsqueeze(0)
        weight_map[valid_y, valid_x] += weights
    
    # Normalize by total weights
    weight_map = torch.clamp(weight_map, min=1e-8)  # Prevent division by zero
    panorama = panorama / weight_map.unsqueeze(0)
    
    return panorama
def render_equidistant_panorama(model_path, iteration, scene, gaussians, pipeline, background, width=2048, height=1024, camera_fov=70):
    """
    Render an equidistant panorama and save it as a single image.
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
    print(f"Rendering equidistant panorama: {width}x{height}")
    
    # Create output directory
    panorama_path = os.path.join(model_path, "panorama", "ours_{}".format(iteration))
    makedirs(panorama_path, exist_ok=True)
    
    # Create panorama cameras
    panorama_cameras = create_panorama_cameras(scene_center, width=width, height=height, camera_fov=camera_fov)
    
    # Store rendered views
    rendered_views = []
    
    # Render each camera view
    print(f"Rendering {len(panorama_cameras)} camera views...")
    for i, camera in enumerate(tqdm(panorama_cameras, desc="Rendering panorama views")):
        rendering = render(camera, gaussians, pipeline, background)["render"]
        rendered_views.append(rendering)
        
        # Optionally save individual camera views for debugging
        if False:  # Set to True for debugging
            debug_path = os.path.join(panorama_path, f"debug_cam_{i:03d}.png")
            torchvision.utils.save_image(rendering, debug_path)
    
    # Project all views into equirectangular panorama
    print("Projecting to equirectangular coordinates...")
    panorama = project_to_equirectangular(rendered_views, panorama_cameras, width, height)
    
    # Save the panorama
    output_path = os.path.join(panorama_path, f"panorama_{width}x{height}.png")
    torchvision.utils.save_image(panorama, output_path)
    print(f"Saved equidistant panorama to {output_path}")
    
    return panorama


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

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, panorama : bool = False, panorama_width : int = 2048, panorama_height : int = 1024, camera_fov : int = 70):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if panorama:
            # Render equidistant panorama instead of standard views
            render_equidistant_panorama(dataset.model_path, scene.loaded_iter, scene, gaussians, pipeline, background, panorama_width, panorama_height, camera_fov)
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
    parser.add_argument("--panorama", action="store_true", help="Render equidistant panorama instead of COLMAP cameras")
    parser.add_argument("--panorama_width", default=2048, type=int, help="Width of panorama in pixels (default: 2048)")
    parser.add_argument("--panorama_height", default=1024, type=int, help="Height of panorama in pixels (default: 1024)")
    parser.add_argument("--camera_fov", default=70, type=int, help="Field of view for individual cameras in degrees (default: 70)")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.panorama, args.panorama_width, args.panorama_height, args.camera_fov)