# Cubemap Rendering Feature

## Overview

The cubemap rendering feature allows you to generate 6-sided cubemap views from any Gaussian Splatting scene. Instead of using the original COLMAP cameras, this feature creates 6 artificial cameras positioned at the scene center, each facing one of the cardinal directions with 90° field of view.

## Usage

### Basic Usage

```bash
python render.py --cubemap -m <model_path> -s <source_path>
```

### With Custom Resolution

```bash
python render.py --cubemap --cubemap_resolution 1024 -m <model_path> -s <source_path>
```

## Command Line Arguments

- `--cubemap`: Enable cubemap rendering mode (ignores COLMAP cameras)
- `--cubemap_resolution`: Resolution for each cubemap face in pixels (default: 512)

## Output

The cubemap renderer generates:

### Individual Face Images
- `cubemap_posx.png` - Right face (+X direction)
- `cubemap_negx.png` - Left face (-X direction)  
- `cubemap_posy.png` - Up face (+Y direction)
- `cubemap_negy.png` - Down face (-Y direction)
- `cubemap_posz.png` - Forward face (+Z direction)
- `cubemap_negz.png` - Back face (-Z direction)

### Atlas Image
- `cubemap_atlas.png` - Combined atlas in cross layout:
```
       [posy]
[negx][posz][posx][negz]
       [negy]
```

## Technical Details

### Camera Configuration
- **Position**: Scene center (calculated from average camera positions)
- **Field of View**: 90° (π/2 radians) for both X and Y axes
- **Aspect Ratio**: 1:1 (square images)
- **Up Vectors**: Properly oriented for each direction according to cubemap convention

### Scene Center Calculation
The scene center is determined by:
1. If training cameras exist: Average position of all training cameras
2. Fallback: Origin (0, 0, 0)

### Coordinate System
- **+X**: Right
- **-X**: Left  
- **+Y**: Up
- **-Y**: Down
- **+Z**: Forward
- **-Z**: Back

## Implementation Details

The feature is implemented with minimal changes to the existing codebase:

1. **New CLI arguments** added to `render.py`
2. **Camera generation function** `create_cubemap_cameras()` creates proper camera matrices
3. **Rendering function** `render_cubemap()` handles the cubemap rendering pipeline
4. **Atlas creation** `create_cubemap_atlas()` combines faces into a single image
5. **Modified main logic** to route to cubemap rendering when enabled

### Key Functions

- `create_cubemap_cameras(scene_center, resolution)`: Creates 6 Camera objects
- `render_cubemap(model_path, iteration, scene, gaussians, pipeline, background, resolution)`: Renders and saves cubemap
- `create_cubemap_atlas(rendered_faces, output_dir)`: Creates combined atlas image

## Compatibility

- ✅ **Preserves existing functionality**: All existing rendering modes continue to work unchanged
- ✅ **Uses existing rendering pipeline**: Leverages the same `render()` function and Camera class
- ✅ **No breaking changes**: Existing scripts and workflows are unaffected
- ✅ **Standard output format**: Uses PNG format compatible with most applications

## Use Cases

- **360° Content Creation**: Generate cubemaps for VR/AR applications
- **Environment Mapping**: Create environment maps for 3D graphics pipelines  
- **Quality Assessment**: Evaluate scene reconstruction from multiple viewpoints
- **Panoramic Visualization**: Convert 3D scenes to panoramic formats

## Limitations

- Requires a trained Gaussian Splatting model
- Scene center approximation may not be perfect for all scenes
- Fixed 90° FOV (cubemap standard)
- Square aspect ratio only

## Example Workflow

1. Train your Gaussian Splatting model normally
2. Run cubemap rendering:
   ```bash
   python render.py --cubemap --cubemap_resolution 1024 -m ./models/my_scene -s ./data/my_scene
   ```
3. Find output in `./models/my_scene/cubemap/ours_<iteration>/`
4. Use individual faces or atlas as needed

## Future Enhancements

Potential improvements could include:
- Custom scene center specification
- Variable FOV support
- Different output formats (EXR, HDR)
- Batch processing multiple models
- Integration with 360° video formats