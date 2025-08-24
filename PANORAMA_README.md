# Equidistant Panorama Rendering Feature

## Overview

The equidistant panorama rendering feature allows you to generate full 360° × 180° panoramic views from any Gaussian Splatting scene. Instead of using the original COLMAP cameras or cubemap faces, this feature creates multiple artificial cameras positioned at the scene center and distributed across the sphere to capture the complete surrounding view. The resulting views are then projected into a single equirectangular panorama image.

## Usage

### Basic Usage

```bash
python render.py --panorama -m <model_path> -s <source_path>
```

### With Custom Resolution and Camera Settings

```bash
python render.py --panorama --panorama_width 4096 --panorama_height 2048 --camera_fov 60 -m <model_path> -s <source_path>
```

## Command Line Arguments

- `--panorama`: Enable equidistant panorama rendering mode (ignores COLMAP cameras)
- `--panorama_width`: Width of the output panorama in pixels (default: 2048)
- `--panorama_height`: Height of the output panorama in pixels (default: 1024)
- `--camera_fov`: Field of view for individual cameras in degrees (default: 70)

## Output

The panorama renderer generates:

### Single Panoramic Image
- `panorama_<width>x<height>.png` - Complete 360° × 180° equirectangular panorama

### Optional Debug Images
- Individual camera views can be saved for debugging by setting the debug flag in the code

## Technical Details

### Camera Configuration
- **Position**: Scene center (calculated from average camera positions)
- **Distribution**: Cameras distributed across the sphere using spherical coordinates
- **Field of View**: Configurable (default: 70°) with overlap for seamless blending
- **Aspect Ratio**: Individual cameras use 1:1 aspect ratio; final panorama typically 2:1

### Scene Center Calculation
The scene center is determined by:
1. If training cameras exist: Average position of all training cameras
2. Fallback: Origin (0, 0, 0)

### Equirectangular Projection
- **Coordinate System**: Standard spherical coordinates
- **Horizontal Range**: 360° (full azimuth)
- **Vertical Range**: 180° (full elevation)
- **Projection**: Linear mapping from spherical coordinates to rectangular image coordinates

### Camera Sampling Strategy
- Cameras are distributed in a grid pattern across the sphere
- Number of cameras automatically calculated based on FOV and desired overlap
- Each camera renders a view that is then projected into the final panorama
- Overlapping regions are blended using distance-weighted averaging

## Implementation Details

The feature is implemented with minimal changes to the existing codebase:

1. **New CLI arguments** added to `render.py` for panorama configuration
2. **Camera generation function** `create_panorama_cameras()` creates distributed camera array
3. **Rendering function** `render_equidistant_panorama()` handles the panorama rendering pipeline
4. **Projection function** `project_to_equirectangular()` combines views into single panorama
5. **Modified main logic** to route to panorama rendering when enabled

### Key Functions

- `create_panorama_cameras(scene_center, width, height, camera_fov)`: Creates array of Camera objects distributed across sphere
- `render_equidistant_panorama(model_path, iteration, scene, gaussians, pipeline, background, width, height, camera_fov)`: Renders and saves panorama
- `project_to_equirectangular(rendered_views, cameras, output_width, output_height)`: Projects multiple camera views into single equirectangular image

## Compatibility

- ✅ **Preserves existing functionality**: All existing rendering modes continue to work unchanged
- ✅ **Uses existing rendering pipeline**: Leverages the same `render()` function and Camera class
- ✅ **No breaking changes**: Existing scripts and workflows are unaffected
- ✅ **Standard output format**: Uses PNG format compatible with most panoramic viewers
- ✅ **Replaces cubemap functionality**: Provides superior 360° coverage compared to cubemap

## Use Cases

- **360° Content Creation**: Generate panoramas for VR/AR applications and virtual tours
- **Environment Mapping**: Create seamless environment maps for 3D graphics pipelines  
- **Quality Assessment**: Evaluate scene reconstruction from all viewpoints
- **Panoramic Visualization**: Convert 3D scenes to standard panoramic formats
- **Virtual Tourism**: Create immersive panoramic experiences from 3D scenes

## Advantages over Cubemap

- **Seamless 360° Coverage**: Single image covers entire sphere without seams
- **Standard Format**: Equirectangular is widely supported by panoramic viewers
- **Configurable Resolution**: Flexible output resolution and aspect ratios
- **Better Quality**: Overlapping camera views provide better blending and quality
- **Industry Standard**: Compatible with VR/AR platforms and panoramic tools

## Limitations

- Requires a trained Gaussian Splatting model
- Scene center approximation may not be perfect for all scenes
- Higher computational cost due to multiple camera renders
- Memory usage scales with number of cameras and output resolution

## Example Workflow

1. Train your Gaussian Splatting model normally
2. Run panorama rendering:
   ```bash
   python render.py --panorama --panorama_width 4096 --panorama_height 2048 -m ./models/my_scene -s ./data/my_scene
   ```
3. Find output in `./models/my_scene/panorama/ours_<iteration>/`
4. Use the panorama in VR viewers, panoramic image viewers, or 3D applications

## Performance Optimization

- Camera FOV can be adjusted to balance quality vs. performance
- Lower resolution cameras with higher counts provide better quality
- GPU memory requirements scale with number of cameras and resolution
- Vectorized projection reduces computational overhead

## Future Enhancements

Potential improvements could include:
- Custom scene center specification
- Stereoscopic panorama generation
- HDR output formats (EXR, HDR)
- Temporal panorama sequences
- Integration with streaming panorama formats
- Adaptive camera density based on scene complexity