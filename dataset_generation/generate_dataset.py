import argparse
import os
import yaml
import numpy as np
import subprocess
from tqdm import tqdm
import shutil 

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate a dataset for reflection removal.")
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder for the dataset')
    parser.add_argument('--mode', type=str, choices=['hdri', 'img'], default='hdri', help='Rendering mode: "hdri" for HDRI panorama, "img" for standard image with HDRI background')
    parser.add_argument('--dataset_config', type=str, default="sample_config.yaml", help='Path to the dataset configuration file')
    parser.add_argument('--num_scenes', type=int, default=None, help='Number of scenes to render (optional)')
    parser.add_argument('--save_blend ', action='store_true', help='Save the Blender file')
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    shutil.copy(args.dataset_config, os.path.join(args.output_folder, 'dataset_config.yaml'))

    # Load dataset configuration
    with open(args.dataset_config, 'r') as f:
        config = yaml.safe_load(f)
    camera_config = config['camera']
    render_config = config['render']
    glass_config = config['glass']

    # Generate dataset based on configuration
    hdr_folder = config['hdr_source']
    print(list(os.listdir(hdr_folder)))
    for id, hdr_file in tqdm(enumerate(os.listdir(hdr_folder)), desc="Generating dataset"):
        if args.num_scenes is not None and id >= args.num_scenes:
            break
        if hdr_file.endswith('.hdr'):
            hdr_path = os.path.join(hdr_folder, hdr_file)
            for i in range(config['samples_per_scene']):
                output_name = f"sample_{id:04d}_{i:04d}"

                azimuth = np.random.uniform(camera_config['min_camera_azimuth'], camera_config['max_camera_azimuth'])
                zimuth = np.random.uniform(camera_config['min_camera_zimuth'], camera_config['max_camera_zimuth'])
                tilt = np.random.uniform(camera_config['min_camera_tilt'], camera_config['max_camera_tilt'])

                default_ior = np.random.uniform(glass_config['min_glass_default_ior'], glass_config['max_glass_default_ior'])
                distance = np.random.uniform(glass_config['min_glass_distance'], glass_config['max_glass_distance'])
                roughness = np.random.uniform(glass_config['min_glass_roughness'], glass_config['max_glass_roughness'])
                metallic = np.random.uniform(glass_config['min_glass_metallic'], glass_config['max_glass_metallic'])
                thickness = np.random.uniform(glass_config['min_glass_thickness'], glass_config['max_glass_thickness'])
                color_grey_scale = np.random.uniform(glass_config['min_glass_color_grey_scale'], glass_config['max_glass_color_grey_scale'])
                background_strength = np.random.uniform(config['min_background_strength'], config['max_background_strength'])
                command = [
                    'blender', '--background','--python', 'render_hdr.py', '--', 
                    '--input_hdr', str(hdr_path),
                    '--output_name', output_name,
                    '--output_folder', args.output_folder,
                    '--resolution_x', str(render_config['resolution_x']),
                    '--resolution_y', str(render_config['resolution_y']),
                    '--cycles_samples', str(render_config['cycles_samples']),
                    '--camera_focal_length', str(camera_config['camera_focal_length']),
                    '--camera_azimuth', str(azimuth),
                    '--camera_zimuth', str(zimuth),
                    '--camera_tilt', str(tilt),
                    '--background_strength', str(background_strength),
                    '--glass_default_ior', str(default_ior),
                    '--glass_distance', str(distance),
                    '--glass_transmissive_ior', str(glass_config['glass_transmissive_ior']),
                    '--glass_roughness', str(roughness),
                    '--glass_metallic', str(metallic),
                    '--glass_color_grey_scale', str(color_grey_scale),
                    '--glass_thickness', str(thickness), 
                    '--glass_alpha', str(glass_config['glass_alpha'])
                ]
                if hasattr(args, 'save_blend') and args.save_blend:
                    command.append('--save_blend')

                # Print the command for debugging
                print(f"Running command: {' '.join(command)}")

                # Run the Blender command to render the HDR image
                subprocess.run(command, check=True)

if __name__ == "__main__":
    main()