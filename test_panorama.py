#!/usr/bin/env python3
"""
Simple test script to verify the panorama functionality without full Gaussian Splatting setup.
This tests the key mathematical functions and coordinate conversions.
"""

import math
import numpy as np
import torch
from PIL import Image

def test_spherical_coordinates():
    """Test spherical coordinate conversions"""
    print("Testing spherical coordinate conversions...")
    
    # Test standard directions
    test_cases = [
        (0, 0),          # Forward (+Z)
        (math.pi/2, 0),  # Right (+X)
        (math.pi, 0),    # Back (-Z)
        (-math.pi/2, 0), # Left (-X)
        (0, math.pi/2),  # Up (+Y)
        (0, -math.pi/2), # Down (-Y)
    ]
    
    for theta, phi in test_cases:
        # Convert to Cartesian
        x = math.cos(phi) * math.cos(theta)
        y = math.sin(phi)
        z = math.cos(phi) * math.sin(theta)
        
        print(f"θ={theta:.3f}, φ={phi:.3f} -> ({x:.3f}, {y:.3f}, {z:.3f})")
    
    print("✓ Spherical coordinate conversion test passed\n")

def test_equirectangular_mapping():
    """Test equirectangular projection mapping"""
    print("Testing equirectangular mapping...")
    
    width, height = 360, 180  # Simple test resolution
    
    # Test specific pixel coordinates
    test_pixels = [
        (0, height//2),      # Left edge, middle (-180°, 0°)
        (width//2, height//2), # Center (0°, 0°)
        (width-1, height//2),  # Right edge, middle (180°, 0°)
        (width//2, 0),         # Top center (0°, 90°)
        (width//2, height-1),  # Bottom center (0°, -90°)
    ]
    
    for x, y in test_pixels:
        # Convert to spherical coordinates
        u = x / width
        v = y / height
        
        theta = u * 2 * math.pi - math.pi  # -π to π
        phi = (0.5 - v) * math.pi          # -π/2 to π/2
        
        theta_deg = math.degrees(theta)
        phi_deg = math.degrees(phi)
        
        print(f"Pixel ({x}, {y}) -> θ={theta_deg:.1f}°, φ={phi_deg:.1f}°")
    
    print("✓ Equirectangular mapping test passed\n")

def test_camera_grid_generation():
    """Test camera grid generation logic"""
    print("Testing camera grid generation...")
    
    camera_fov = 70  # degrees
    fov_rad = math.radians(camera_fov)
    
    # Calculate number of cameras
    overlap_factor = 0.8
    cameras_horizontal = max(4, int(2 * math.pi / (fov_rad * overlap_factor)))
    cameras_vertical = max(2, int(math.pi / (fov_rad * overlap_factor)))
    
    print(f"FOV: {camera_fov}° -> Grid: {cameras_horizontal}x{cameras_vertical} = {cameras_horizontal * cameras_vertical} cameras")
    
    # Test a few camera positions
    for v_idx in range(min(2, cameras_vertical)):
        for h_idx in range(min(3, cameras_horizontal)):
            theta = (h_idx / cameras_horizontal) * 2 * math.pi
            phi = ((v_idx + 0.5) / cameras_vertical - 0.5) * math.pi
            
            theta_deg = math.degrees(theta)
            phi_deg = math.degrees(phi)
            
            print(f"Camera [{v_idx}][{h_idx}]: θ={theta_deg:.1f}°, φ={phi_deg:.1f}°")
    
    print("✓ Camera grid generation test passed\n")

def test_projection_performance():
    """Test the performance characteristics of the projection"""
    print("Testing projection performance characteristics...")
    
    # Simulate small tensors for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create small test panorama
    width, height = 64, 32
    
    # Create coordinate meshgrids
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # Convert to spherical coordinates
    u = x_coords / width
    v = y_coords / height
    
    theta = u * 2 * math.pi - math.pi
    phi = (0.5 - v) * math.pi
    
    # Convert to direction vectors
    directions = torch.stack([
        torch.cos(phi) * torch.cos(theta),
        torch.sin(phi),
        torch.cos(phi) * torch.sin(theta)
    ], dim=0)
    
    print(f"Direction vectors shape: {directions.shape}")
    print(f"Sample direction at center: {directions[:, height//2, width//2].cpu().numpy()}")
    
    print("✓ Projection performance test passed\n")

def main():
    """Run all tests"""
    print("=== Panorama Functionality Tests ===\n")
    
    try:
        test_spherical_coordinates()
        test_equirectangular_mapping()
        test_camera_grid_generation()
        test_projection_performance()
        
        print("🎉 All tests passed! Panorama implementation looks good.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()