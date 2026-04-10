# Dataset Generation

This folder contains Blender-based scripts for generating synthetic reflection-removal training data.

The pipeline renders multiple variants per sample:
- `*_default.png`: base render with glass
- `*_transmissive.png`: transmissive glass variant
- `*_reflection.png`: reflective variant
- `*_no_glass.png`: only in HDR-only scene mode (`generate_dataset.py`)
- `*_config.json`: per-sample render parameters

## Scripts

- `generate_dataset.py`
  - Iterates over `.hdr` files from `hdr_source`
  - Uses `render_hdr.py` in `hdri` mode
  - Produces scene variants including `no_glass`

- `generate_img_dataset.py`
  - Iterates over images from `img_source`
  - Uses `render_hdr.py` in `img` mode
  - Background can be chosen from either HDR files (`--background_mode hdri`) or images (`--background_mode img`)

- `render_hdr.py`
  - Blender entry script used by both generators
  - Sets up camera, background, medium/glass material, and compositor outputs

## Requirements

- Blender available on `PATH` as `blender`
- Blender add-on: **Import Images as Planes** (used by image mode)
- Python packages for launcher scripts:
  - `pyyaml`
  - `numpy`
  - `tqdm`

Example:

```bash
pip install pyyaml numpy tqdm
```

## Dataset Config

Both launchers expect a YAML config passed with `--dataset_config` (default: `sample_config.yaml`).

Required top-level keys:

```yaml
hdr_source: /absolute/or/relative/path/to/hdr/files
img_source: /absolute/or/relative/path/to/images
samples_per_scene: 4
min_background_strength: 0.5
max_background_strength: 2.0

camera:
  camera_focal_length: 15.0
  min_camera_azimuth: 0
  max_camera_azimuth: 360
  min_camera_zimuth: 45
  max_camera_zimuth: 135
  min_camera_tilt: -10
  max_camera_tilt: 10

render:
  resolution_x: 1920
  resolution_y: 1080
  cycles_samples: 128

glass:
  min_glass_default_ior: 1.3
  max_glass_default_ior: 1.6
  min_glass_distance: 0.1
  max_glass_distance: 2.0
  glass_transmissive_ior: 1.0
  min_glass_roughness: 0.0
  max_glass_roughness: 0.2
  min_glass_metallic: 0.0
  max_glass_metallic: 1.0
  min_glass_color_grey_scale: 0.7
  max_glass_color_grey_scale: 1.0
  min_glass_thickness: 0.001
  max_glass_thickness: 0.05
  glass_alpha: 1.0
  glass_as_plane: false  # optional, used in generate_img_dataset.py
```

## Usage

Run commands from this directory (`dataset_generation`) so `render_hdr.py` resolves correctly.

### 1) HDR-only scene generation

```bash
python generate_dataset.py \
  --output_folder ../storage/generated_hdri \
  --dataset_config ./sample_config.yaml \
  --num_scenes 20
```

Optional:
- `--save_blend`: save `.blend` files per sample

### 2) Image-based generation

```bash
python generate_img_dataset.py \
  --output_folder ../storage/generated_img \
  --dataset_config ./sample_config.yaml \
  --background_mode img \
  --num_images 100
```

Notes:
- `--background_mode img`: picks random background images from `img_source`
- `--background_mode hdri`: picks random `.hdr` backgrounds from `hdr_source`

## Output Naming

Samples are named:
- `sample_<scene_or_img_id>_<sample_id>_default.png`
- `sample_<scene_or_img_id>_<sample_id>_transmissive.png`
- `sample_<scene_or_img_id>_<sample_id>_reflection.png`
- `sample_<scene_or_img_id>_<sample_id>_no_glass.png` (HDR-only generation)
- `sample_<scene_or_img_id>_<sample_id>_config.json`

## Troubleshooting

- `blender: command not found`
  - Install Blender and ensure `blender` is on `PATH`.
- Blender import operator error for image planes
  - Enable **Import Images as Planes** in Blender add-ons.
- Script cannot find config or assets
  - Verify `--dataset_config`, `hdr_source`, and `img_source` paths.
- CUDA device setup warnings in Blender
  - `render_hdr.py` defaults to CUDA; adjust device settings in script if needed for your environment.
