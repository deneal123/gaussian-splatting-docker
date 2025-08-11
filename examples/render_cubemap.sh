#!/bin/bash
# Example usage script for cubemap rendering

echo "Gaussian Splatting Cubemap Rendering Example"
echo "============================================"
echo

# Check if model path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <model_path> [source_path] [resolution]"
    echo
    echo "Example:"
    echo "  $0 ./models/my_scene ./data/my_scene 1024"
    echo "  $0 ./models/my_scene ./data/my_scene"
    echo "  $0 ./models/my_scene"
    echo
    exit 1
fi

MODEL_PATH="$1"
SOURCE_PATH="${2:-$MODEL_PATH}"  # Default to model path if not provided
RESOLUTION="${3:-512}"           # Default to 512 if not provided

echo "Configuration:"
echo "  Model Path: $MODEL_PATH"
echo "  Source Path: $SOURCE_PATH"
echo "  Resolution: ${RESOLUTION}x${RESOLUTION} per face"
echo

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path '$MODEL_PATH' does not exist"
    exit 1
fi

# Check if source path exists
if [ ! -d "$SOURCE_PATH" ]; then
    echo "Error: Source path '$SOURCE_PATH' does not exist"
    exit 1
fi

echo "Starting cubemap rendering..."
echo "Command: python render.py --cubemap --cubemap_resolution $RESOLUTION -m \"$MODEL_PATH\" -s \"$SOURCE_PATH\""
echo

# Run the cubemap rendering
python render.py --cubemap --cubemap_resolution "$RESOLUTION" -m "$MODEL_PATH" -s "$SOURCE_PATH"

if [ $? -eq 0 ]; then
    echo
    echo "✓ Cubemap rendering completed successfully!"
    echo
    echo "Output files should be in:"
    echo "  $MODEL_PATH/cubemap/ours_<iteration>/"
    echo
    echo "Generated files:"
    echo "  - cubemap_posx.png  (Right face)"
    echo "  - cubemap_negx.png  (Left face)"
    echo "  - cubemap_posy.png  (Up face)"
    echo "  - cubemap_negy.png  (Down face)"
    echo "  - cubemap_posz.png  (Forward face)"
    echo "  - cubemap_negz.png  (Back face)"
    echo "  - cubemap_atlas.png (Combined atlas)"
    echo
else
    echo
    echo "✗ Cubemap rendering failed!"
    echo "Check the error messages above for details."
    echo
    exit 1
fi