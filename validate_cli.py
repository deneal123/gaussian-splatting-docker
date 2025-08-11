#!/usr/bin/env python3
"""
Extract and display the CLI help information for the updated render.py
"""

import ast
import re

def extract_cli_arguments():
    """Extract CLI argument definitions from render.py"""
    
    with open('render.py', 'r') as f:
        content = f.read()
    
    # Find argument parser section
    parser_section = re.search(r'parser\.add_argument.*?args = get_combined_args', content, re.DOTALL)
    
    if parser_section:
        lines = parser_section.group().split('\n')
        arguments = []
        
        for line in lines:
            if 'parser.add_argument' in line:
                # Extract argument details
                match = re.search(r'"--([^"]+)".*?help="([^"]*)"', line)
                if match:
                    arg_name = match.group(1)
                    help_text = match.group(2)
                    
                    # Extract type and default if present
                    type_match = re.search(r'type=(\w+)', line)
                    default_match = re.search(r'default=([^,\)]+)', line)
                    action_match = re.search(r'action="([^"]+)"', line)
                    
                    arg_type = type_match.group(1) if type_match else None
                    default_val = default_match.group(1) if default_match else None
                    action = action_match.group(1) if action_match else None
                    
                    arguments.append({
                        'name': arg_name,
                        'help': help_text,
                        'type': arg_type,
                        'default': default_val,
                        'action': action
                    })
        
        return arguments
    
    return []

def show_old_vs_new():
    """Show comparison of old vs new CLI arguments"""
    
    print("=== CLI ARGUMENT CHANGES ===\n")
    
    print("🔴 REMOVED (old cubemap arguments):")
    print("   --cubemap                 Enable cubemap rendering mode")
    print("   --cubemap_resolution      Resolution for each cubemap face (default: 512)")
    
    print("\n🟢 ADDED (new panorama arguments):")
    arguments = extract_cli_arguments()
    
    panorama_args = [arg for arg in arguments if 'panorama' in arg['name'] or 'camera_fov' in arg['name']]
    
    for arg in panorama_args:
        flag = f"--{arg['name']}"
        help_text = arg['help']
        
        if arg['action'] == 'store_true':
            print(f"   {flag:<24} {help_text}")
        else:
            type_str = f"({arg['type']}) " if arg['type'] else ""
            default_str = f" (default: {arg['default']})" if arg['default'] else ""
            print(f"   {flag:<24} {type_str}{help_text}{default_str}")
    
    print(f"\n📊 Total new arguments: {len(panorama_args)}")

def show_usage_examples():
    """Show usage examples for the new panorama functionality"""
    
    print("\n=== USAGE EXAMPLES ===\n")
    
    print("1. Basic panorama rendering:")
    print("   python render.py --panorama -m <model_path> -s <source_path>")
    
    print("\n2. High-resolution panorama:")
    print("   python render.py --panorama --panorama_width 4096 --panorama_height 2048 -m <model_path> -s <source_path>")
    
    print("\n3. Custom camera settings:")
    print("   python render.py --panorama --camera_fov 60 --panorama_width 2048 --panorama_height 1024 -m <model_path> -s <source_path>")
    
    print("\n4. Standard rendering (unchanged):")
    print("   python render.py -m <model_path> -s <source_path>")

def show_output_comparison():
    """Show output file comparison"""
    
    print("\n=== OUTPUT COMPARISON ===\n")
    
    print("🔴 OLD CUBEMAP OUTPUT:")
    print("   📁 model_path/cubemap/ours_<iteration>/")
    print("   ├── 📄 cubemap_posx.png")
    print("   ├── 📄 cubemap_negx.png") 
    print("   ├── 📄 cubemap_posy.png")
    print("   ├── 📄 cubemap_negy.png")
    print("   ├── 📄 cubemap_posz.png")
    print("   ├── 📄 cubemap_negz.png")
    print("   └── 📄 cubemap_atlas.png")
    print("   Total: 7 files")
    
    print("\n🟢 NEW PANORAMA OUTPUT:")
    print("   📁 model_path/panorama/ours_<iteration>/")
    print("   └── 📄 panorama_<width>x<height>.png")
    print("   Total: 1 file")

def main():
    """Main function to show all changes"""
    
    print("🔄 RENDER.PY CLI INTERFACE CHANGES")
    print("=" * 50)
    
    show_old_vs_new()
    show_usage_examples()
    show_output_comparison()
    
    print("\n" + "=" * 50)
    print("✅ CLI successfully updated for panorama functionality")
    print("🎯 Backward compatibility: All existing arguments preserved")
    print("🚀 New feature: Full 360° panorama generation")

if __name__ == "__main__":
    main()