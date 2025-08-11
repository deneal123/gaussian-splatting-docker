#!/usr/bin/env python3
"""
Integration test for panorama rendering with mock Gaussian Splatting components.
This simulates the full rendering pipeline without requiring the actual dependencies.
"""

import math
import numpy as np
import torch
import os
from PIL import Image

# Mock classes to simulate the Gaussian Splatting components
class MockCamera:
    """Mock Camera class that mimics the interface used in render.py"""
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda",
                 train_test_exp=False, is_test_dataset=False, is_test_view=False):
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_width = resolution[0]
        self.image_height = resolution[1]

def create_panorama_cameras_test(scene_center, width=2048, height=1024, camera_fov=70):
    """
    Test version of create_panorama_cameras function from render.py
    """
    fov = math.radians(camera_fov)
    
    # Calculate number of cameras needed
    overlap_factor = 0.8
    cameras_horizontal = max(4, int(2 * math.pi / (fov * overlap_factor)))
    cameras_vertical = max(2, int(math.pi / (fov * overlap_factor)))
    
    cameras = []
    camera_id = 0
    
    for v_idx in range(cameras_vertical):
        for h_idx in range(cameras_horizontal):
            # Calculate spherical coordinates
            theta = (h_idx / cameras_horizontal) * 2 * math.pi
            phi = ((v_idx + 0.5) / cameras_vertical - 0.5) * math.pi
            
            # Convert to Cartesian direction
            direction = np.array([
                math.cos(phi) * math.cos(theta),
                math.sin(phi),
                math.cos(phi) * math.sin(theta)
            ])
            
            # Calculate up vector
            if abs(phi) > math.pi/2 - 0.1:
                up = np.array([0.0, 0.0, 1.0]) if phi > 0 else np.array([0.0, 0.0, -1.0])
            else:
                up = np.array([0.0, 1.0, 0.0])
            
            # Calculate right vector
            right = np.cross(direction, up)
            if np.linalg.norm(right) < 1e-6:
                right = np.array([1.0, 0.0, 0.0])
            right = right / np.linalg.norm(right)
            
            # Recalculate up
            up = np.cross(right, direction)
            up = up / np.linalg.norm(up)
            
            # Create rotation matrix
            R = np.array([right, up, -direction])
            T = -R @ scene_center
            
            # Create dummy image
            dummy_image = Image.new('RGB', (512, 512), color='black')
            
            camera = MockCamera(
                (512, 512),
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
            
            # Store spherical coordinates
            camera.theta = theta
            camera.phi = phi
            cameras.append(camera)
            camera_id += 1
    
    return cameras

def mock_render(camera):
    """
    Mock render function that creates a simple test pattern
    """
    # Create a simple gradient pattern based on camera direction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate a simple test pattern
    height, width = 512, 512
    
    # Create gradient based on camera theta and phi
    theta_norm = (camera.theta + math.pi) / (2 * math.pi)  # 0 to 1
    phi_norm = (camera.phi + math.pi/2) / math.pi          # 0 to 1
    
    # Create RGB channels with different patterns
    r_channel = torch.full((height, width), theta_norm, device=device)
    g_channel = torch.full((height, width), phi_norm, device=device)
    b_channel = torch.full((height, width), 0.5, device=device)
    
    # Add some spatial variation
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(0, 1, height, device=device),
        torch.linspace(0, 1, width, device=device),
        indexing='ij'
    )
    
    # Combine into RGB image
    rendered_image = torch.stack([
        r_channel * (0.7 + 0.3 * x_coords),
        g_channel * (0.7 + 0.3 * y_coords),
        b_channel * (0.7 + 0.3 * (x_coords + y_coords) / 2)
    ], dim=0)
    
    return {"render": rendered_image}

def project_to_equirectangular_test(rendered_views, cameras, output_width, output_height):
    """
    Test version of the projection function with simplified logic for validation
    """
    device = rendered_views[0].device
    panorama = torch.zeros(3, output_height, output_width, device=device)
    weight_map = torch.zeros(output_height, output_width, device=device)
    
    # Use a simpler sampling approach for testing
    for y in range(0, output_height, 4):  # Sample every 4th pixel for speed
        for x in range(0, output_width, 4):
            # Convert to spherical coordinates
            u = x / output_width
            v = y / output_height
            
            theta = u * 2 * math.pi - math.pi
            phi = (0.5 - v) * math.pi
            
            # Find the best camera for this direction
            best_camera = None
            best_distance = float('inf')
            
            for camera in cameras:
                # Calculate angular distance
                camera_theta, camera_phi = camera.theta, camera.phi
                
                # Simple angular distance approximation
                d_theta = abs(theta - camera_theta)
                d_phi = abs(phi - camera_phi)
                
                # Handle wraparound for theta
                d_theta = min(d_theta, 2 * math.pi - d_theta)
                
                distance = math.sqrt(d_theta**2 + d_phi**2)
                
                if distance < best_distance and distance < camera.FoVx:
                    best_distance = distance
                    best_camera = camera
            
            if best_camera is not None:
                # Get the corresponding rendered view
                camera_idx = cameras.index(best_camera)
                rendered_view = rendered_views[camera_idx]
                
                # Simple sampling from center of camera view
                center_y, center_x = rendered_view.shape[1] // 2, rendered_view.shape[2] // 2
                pixel_value = rendered_view[:, center_y, center_x]
                
                # Weight based on distance
                weight = max(0, 1 - best_distance / best_camera.FoVx)
                
                # Fill in a small region around this pixel
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < output_height and 0 <= nx < output_width:
                            panorama[:, ny, nx] += pixel_value * weight
                            weight_map[ny, nx] += weight
    
    # Normalize
    weight_map = torch.clamp(weight_map, min=1e-8)
    panorama = panorama / weight_map.unsqueeze(0)
    
    return panorama

def test_full_pipeline():
    """Test the complete panorama rendering pipeline"""
    print("=== Full Pipeline Integration Test ===\n")
    
    # Test parameters
    scene_center = np.array([0.0, 0.0, 0.0])
    panorama_width = 256   # Small for testing
    panorama_height = 128
    camera_fov = 70
    
    print(f"Testing panorama: {panorama_width}x{panorama_height}")
    print(f"Camera FOV: {camera_fov}°")
    print(f"Scene center: {scene_center}")
    
    # Step 1: Create panorama cameras
    print("\n1. Creating panorama cameras...")
    cameras = create_panorama_cameras_test(scene_center, panorama_width, panorama_height, camera_fov)
    print(f"   Created {len(cameras)} cameras")
    
    # Step 2: Mock render each camera
    print("\n2. Rendering camera views...")
    rendered_views = []
    for i, camera in enumerate(cameras):
        rendering = mock_render(camera)["render"]
        rendered_views.append(rendering)
        if i < 3:  # Show details for first few cameras
            print(f"   Camera {i}: θ={math.degrees(camera.theta):.1f}°, φ={math.degrees(camera.phi):.1f}°")
    
    # Step 3: Project to panorama
    print("\n3. Projecting to equirectangular panorama...")
    panorama = project_to_equirectangular_test(rendered_views, cameras, panorama_width, panorama_height)
    
    print(f"   Panorama shape: {panorama.shape}")
    print(f"   Panorama value range: [{panorama.min():.3f}, {panorama.max():.3f}]")
    
    # Step 4: Save test result
    print("\n4. Saving test panorama...")
    
    # Create output directory
    output_dir = "/tmp/panorama_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to PIL and save
    panorama_np = panorama.cpu().numpy().transpose(1, 2, 0)  # HWC format
    panorama_np = np.clip(panorama_np * 255, 0, 255).astype(np.uint8)
    
    pil_image = Image.fromarray(panorama_np)
    output_path = os.path.join(output_dir, f"test_panorama_{panorama_width}x{panorama_height}.png")
    pil_image.save(output_path)
    
    print(f"   Saved test panorama to: {output_path}")
    
    # Step 5: Validate results
    print("\n5. Validating results...")
    
    # Check that we have a valid image
    assert panorama.shape == (3, panorama_height, panorama_width), f"Wrong shape: {panorama.shape}"
    assert not torch.isnan(panorama).any(), "Panorama contains NaN values"
    assert not torch.isinf(panorama).any(), "Panorama contains infinite values"
    
    # Check that we have some variation (not all zeros)
    assert panorama.std() > 0.001, "Panorama has no variation (all zeros?)"
    
    print("   ✓ Shape is correct")
    print("   ✓ No NaN or infinite values")
    print("   ✓ Has appropriate variation")
    
    print("\n🎉 Full pipeline test passed!")
    return output_path

def main():
    """Run integration test"""
    try:
        output_path = test_full_pipeline()
        print(f"\nTest completed successfully!")
        print(f"Test panorama saved to: {output_path}")
        print("You can view this image to verify the panorama generation works.")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()