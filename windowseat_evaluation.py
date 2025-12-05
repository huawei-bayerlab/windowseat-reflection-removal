import gdown
import os
import zipfile
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm as tbar
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch
import lpips
import piq
import torch
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
import sys

if HERE not in sys.path:
    sys.path.insert(0, HERE)
from windowseat_inference import (
    load_network,
    run_inference,
    SUPPORTED_MODEL_URIS,
    LORA_MODEL_URI,
)


REGISTRY_EVAL = {"paths": {}, "evaluation_results": {}}
DATASET_LINKS = {
    "nature": "https://drive.google.com/uc?id=1YWkm80jWsjX6XwLTHOsa8zK3pSRalyCg",  # Link from https://github.com/ZhenboSong/RobustSIRR?tab=readme-ov-file#two-prepare-dataset
    "real": "https://github.com/mingcv/YTMT-Strategy/releases/download/data/real20_420.zip",  # Following RDNET, YTMT and DSRNet, link from https://github.com/mingcv/YTMT-Strategy/releases/tag/data
    "SIR2_500": "https://www.dropbox.com/scl/fi/qgg1whla1jb3a9cgis18l/SIR2.zip?rlkey=kmhrc2uk63be2s9hzr43gc3hm&e=2&st=cfsh8sol&dl=1",  # Link from https://sir2data.github.io/
}


def download_nature(skip_existing: bool = True):
    zip_path = os.path.join(
        HERE, "data", "evaluation", "test_datasets", "nature_dataset.zip"
    )
    REGISTRY_EVAL["paths"]["nature_zip"] = zip_path

    # If file already exists and we want to skip, just return
    if skip_existing and os.path.isfile(zip_path) and os.path.getsize(zip_path) > 0:
        return

    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    gdown.download(DATASET_LINKS["nature"], output=zip_path, quiet=False)


def unzip_nature(skip_existing: bool = True):
    zip_path = REGISTRY_EVAL["paths"]["nature_zip"]
    out_dir = os.path.splitext(zip_path)[0]
    REGISTRY_EVAL["paths"]["nature_dir"] = out_dir
    REGISTRY_EVAL["paths"]["nature_input"] = os.path.join(out_dir, "testB")
    REGISTRY_EVAL["paths"]["nature_gt"] = os.path.join(out_dir, "testA1")
    REGISTRY_EVAL["paths"]["nature_output"] = os.path.join(
        HERE, "data", "evaluation", "predictions", "nature"
    )
    # If dir already exists and is non-empty, and we want to skip, just set paths & return
    if skip_existing and os.path.isdir(out_dir) and os.listdir(out_dir):
        return

    os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def download_real(skip_existing: bool = True):
    zip_path = os.path.join(
        HERE, "data", "evaluation", "test_datasets", "real20_420.zip"
    )
    REGISTRY_EVAL["paths"]["real_zip"] = zip_path

    # If file already exists and we want to skip, just return
    if skip_existing and os.path.isfile(zip_path) and os.path.getsize(zip_path) > 0:
        return

    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    # gdown also works with normal HTTP(S) URLs
    gdown.download(DATASET_LINKS["real"], output=zip_path, quiet=False)


def unzip_real(skip_existing: bool = True):
    zip_path = REGISTRY_EVAL["paths"]["real_zip"]
    out_dir = os.path.splitext(zip_path)[0]

    REGISTRY_EVAL["paths"]["real_dir"] = out_dir
    # adjust these two to match the actual folder structure inside real20_420.zip
    REGISTRY_EVAL["paths"]["real_input"] = os.path.join(
        out_dir, "real20_420", "blended"
    )
    REGISTRY_EVAL["paths"]["real_gt"] = os.path.join(
        out_dir, "real20_420", "transmission_layer"
    )
    REGISTRY_EVAL["paths"]["real_output"] = os.path.join(
        HERE, "data", "evaluation", "predictions", "real"
    )

    # If dir already exists and is non-empty, and we want to skip, just return
    if skip_existing and os.path.isdir(out_dir) and os.listdir(out_dir):
        return

    os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def download_sir2_500(skip_existing: bool = True):
    zip_path = os.path.join(HERE, "data", "evaluation", "test_datasets", "SIR2_500.zip")
    REGISTRY_EVAL["paths"]["sir2_500_zip"] = zip_path

    # If file already exists and we want to skip, just return
    if skip_existing and os.path.isfile(zip_path) and os.path.getsize(zip_path) > 0:
        return

    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    # gdown also works with normal HTTPS URLs like Dropbox links
    gdown.download(DATASET_LINKS["SIR2_500"], output=zip_path, quiet=False)


def _unzip_nested_zips(root_dir: str, skip_existing: bool = True):
    """
    Recursively unzip all .zip files found under root_dir.

    Each zip is extracted into a folder with the same name (without .zip)
    in the same directory as the zip file.
    """
    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)

        if os.path.isdir(path):
            # Recurse into subdirectories
            _unzip_nested_zips(path, skip_existing=skip_existing)
        elif name.lower().endswith(".zip"):
            target_dir = os.path.splitext(path)[0]

            # Skip if already extracted and non-empty
            if skip_existing and os.path.isdir(target_dir) and os.listdir(target_dir):
                continue

            os.makedirs(target_dir, exist_ok=True)
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(target_dir)

            # Recursively handle any zips inside this newly extracted folder
            _unzip_nested_zips(target_dir, skip_existing=skip_existing)


def unzip_sir2_500(skip_existing: bool = True):
    zip_path = REGISTRY_EVAL["paths"]["sir2_500_zip"]
    out_dir = os.path.splitext(zip_path)[0]

    REGISTRY_EVAL["paths"]["sir2_500_dir"] = out_dir
    REGISTRY_EVAL["paths"]["sir2_500_output"] = os.path.join(
        HERE, "data", "evaluation", "predictions", "sir2_500"
    )

    # If directory already exists and non-empty, assume everything is extracted
    if skip_existing and os.path.isdir(out_dir) and os.listdir(out_dir):
        return

    os.makedirs(out_dir, exist_ok=True)

    # First unzip the main SIR2_500.zip
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    # Then recursively unzip any nested .zip files inside out_dir
    _unzip_nested_zips(out_dir, skip_existing=skip_existing)


def format_sir2_500(skip_existing=True):
    sir_root = REGISTRY_EVAL["paths"]["sir2_500_dir"]  # e.g. .../SIR2_500

    # Base dir for new structured dataset
    base_out = os.path.join(os.path.dirname(sir_root), "SIR2_500_structured")
    os.makedirs(base_out, exist_ok=True)

    # Prepare target dirs
    postcard_input_dir = os.path.join(base_out, "Postcard", "input")
    postcard_gt_dir = os.path.join(base_out, "Postcard", "gt")
    solid_input_dir = os.path.join(base_out, "SolidObject", "input")
    solid_gt_dir = os.path.join(base_out, "SolidObject", "gt")
    wild_input_dir = os.path.join(base_out, "Wildscene", "input")
    wild_gt_dir = os.path.join(base_out, "Wildscene", "gt")

    for d in [
        postcard_input_dir,
        postcard_gt_dir,
        solid_input_dir,
        solid_gt_dir,
        wild_input_dir,
        wild_gt_dir,
    ]:
        os.makedirs(d, exist_ok=True)

    # Optionally store in REGISTRY_EVAL for later use
    REGISTRY_EVAL["paths"]["sir2_500_structured"] = base_out
    REGISTRY_EVAL["paths"]["sir2_500_Postcard_input"] = postcard_input_dir
    REGISTRY_EVAL["paths"]["sir2_500_Postcard_gt"] = postcard_gt_dir
    REGISTRY_EVAL["paths"]["sir2_500_SolidObject_input"] = solid_input_dir
    REGISTRY_EVAL["paths"]["sir2_500_SolidObject_gt"] = solid_gt_dir
    REGISTRY_EVAL["paths"]["sir2_500_Wildscene_input"] = wild_input_dir
    REGISTRY_EVAL["paths"]["sir2_500_Wildscene_gt"] = wild_gt_dir

    REGISTRY_EVAL["paths"]["sir2_500_Postcard_output"] = os.path.join(
        REGISTRY_EVAL["paths"]["sir2_500_output"], "Postcard"
    )
    REGISTRY_EVAL["paths"]["sir2_500_SolidObject_output"] = os.path.join(
        REGISTRY_EVAL["paths"]["sir2_500_output"], "SolidObject"
    )
    REGISTRY_EVAL["paths"]["sir2_500_Wildscene_output"] = os.path.join(
        REGISTRY_EVAL["paths"]["sir2_500_output"], "Wildscene"
    )
    # -------------------------
    # 1) Postcard Dataset
    # -------------------------
    postcard_root = os.path.join(sir_root, "Postcard Dataset", "Postcard Dataset")

    if os.path.isdir(postcard_root):
        for root, _, files in os.walk(postcard_root):
            for fname in files:
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                # Input images contain "-m-" in the name, gt have "-g-"
                if "m" not in fname:
                    continue

                input_path = os.path.join(root, fname)
                gt_fname = fname.replace("m", "g")
                gt_path = os.path.join(root, gt_fname)

                if not os.path.isfile(gt_path):
                    # No matching GT -> skip
                    continue

                # Merge relative path into filename
                rel_input = os.path.relpath(
                    input_path, postcard_root
                )  # e.g. Focus/ab/11/ab-5-m-11.png
                new_input_name = rel_input.replace(
                    os.sep, "_"
                )  # Focus_ab_11_ab-5-m-11.png
                new_gt_name = new_input_name.replace(
                    "m", "g"
                )  # Focus_ab_11_ab-5-g-11.png

                dest_path = os.path.join(postcard_input_dir, new_input_name)
                if not (os.path.exists(dest_path) and skip_existing):
                    shutil.copy2(input_path, dest_path)
                dest_path = os.path.join(postcard_gt_dir, new_gt_name)
                if not (os.path.exists(dest_path) and skip_existing):
                    shutil.copy2(gt_path, dest_path)

    # -------------------------
    # 2) SolidObjectDataset
    # -------------------------
    solid_root = os.path.join(sir_root, "SolidObjectDataset", "SolidObjectDataset")

    if os.path.isdir(solid_root):
        for root, _, files in os.walk(solid_root):
            # we pair m.jpg and g.jpg inside each directory
            for fname in files:
                if not fname.lower().endswith(".jpg"):
                    continue
                if fname.lower() != "m.jpg":
                    continue

                input_path = os.path.join(root, fname)
                gt_fname = "g.jpg"
                gt_path = os.path.join(root, gt_fname)

                if not os.path.isfile(gt_path):
                    continue

                rel_input = os.path.relpath(
                    input_path, solid_root
                )  # e.g. 8/Focus/13/m.jpg
                new_input_name = rel_input.replace(os.sep, "_")  # 8_Focus_13_m.jpg
                new_gt_name = new_input_name.replace("m.jpg", "g.jpg")

                dest_path = os.path.join(solid_input_dir, new_input_name)
                if not (os.path.exists(dest_path) and skip_existing):
                    shutil.copy2(input_path, dest_path)
                dest_path = os.path.join(solid_gt_dir, new_gt_name)
                if not (os.path.exists(dest_path) and skip_existing):
                    shutil.copy2(gt_path, dest_path)

    # -------------------------
    # 3) Wildscene
    # -------------------------
    wild_root = os.path.join(sir_root, "Wildscene")

    if os.path.isdir(wild_root):
        for root, _, files in os.walk(wild_root):
            for fname in files:
                if not fname.lower().endswith(".jpg"):
                    continue
                if fname.lower() != "m.jpg":
                    continue

                input_path = os.path.join(root, fname)
                gt_fname = "g.jpg"
                gt_path = os.path.join(root, gt_fname)

                if not os.path.isfile(gt_path):
                    continue

                rel_input = os.path.relpath(
                    input_path, wild_root
                )  # e.g. 7s/m.jpg or 3/m.jpg
                new_input_name = rel_input.replace(os.sep, "_")  # 7s_m.jpg
                new_gt_name = new_input_name.replace("m.jpg", "g.jpg")

                dest_path = os.path.join(wild_input_dir, new_input_name)
                if not (os.path.exists(dest_path) and skip_existing):
                    shutil.copy2(input_path, dest_path)
                dest_path = os.path.join(wild_gt_dir, new_gt_name)
                if not (os.path.exists(dest_path) and skip_existing):
                    shutil.copy2(gt_path, dest_path)


SIR2_500_COUNTS = {
    "SIR2-500 SolidObject (200)": 200,
    "SIR2-500 Postcard (199)": 199,
    "SIR2-500 Wildscene (101)": 101,
}


def compute_sir2_500_weighted(eval_results):
    """
    eval_results: e.g. REGISTRY_EVAL["evaluation_results"]

    Returns a dict with weighted PSNR/SSIM/MS-SSIM/LPIPS over the
    3 SIR2_500 subsets, weighted by #images (200/199/101).
    """
    subset_names = list(SIR2_500_COUNTS.keys())
    total_imgs = sum(SIR2_500_COUNTS.values())

    # assume all subsets have the same metric keys
    metric_names = eval_results[subset_names[0]].keys()

    weighted = {}
    for m in metric_names:
        num = 0.0
        for subset in subset_names:
            w = SIR2_500_COUNTS[subset]
            num += eval_results[subset][m] * w
        weighted[m] = num / total_imgs
    return weighted


def compute_metrics(prediction_dir, gt_dir, dataset_name=None):
    """
    Compute PSNR, SSIM, MS-SSIM, and LPIPS between prediction_dir and gt_dir.

    - Images are paired by identical filenames.
    - Results are printed.
    - If dataset_name is not None, results are stored in
      REGISTRY_EVAL["evaluation_results"][dataset_name][metric_name].
    - Uses a tbar-style progress bar.
    """
    # Collect all prediction filenames
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    pred_names = sorted(
        [
            f
            for f in os.listdir(prediction_dir)
            if f.lower().endswith(valid_exts)
            and os.path.isfile(os.path.join(prediction_dir, f))
            and not any(s in f.lower() for s in ("side_by_side", "alternating"))
        ],
        key=lambda f: os.path.splitext(f)[0].replace("_windowseat_output", ""),
    )
    gt_names = sorted(
        [
            f
            for f in os.listdir(gt_dir)
            if f.lower().endswith(valid_exts)
            and os.path.isfile(os.path.join(gt_dir, f))
        ]
    )
    if not pred_names:
        print(f"[compute_metrics] No prediction images found in: {prediction_dir}")
        return

    # Setup device and LPIPS network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_net = lpips.LPIPS(net="alex").to(device)
    lpips_net.eval()

    psnr_vals = []
    ssim_vals = []
    msssim_vals = []
    lpips_vals = []

    desc = (
        f"Computing metrics ({dataset_name})" if dataset_name else "Computing metrics"
    )
    for i in tbar(range(len(pred_names)), desc=desc):

        pred_path = os.path.join(prediction_dir, pred_names[i])
        gt_path = os.path.join(gt_dir, gt_names[i])
        if not os.path.isfile(gt_path):
            # If no matching GT exists, skip this file
            print(f"{gt_path} doesn't exist, skipping...")
            continue

        # ---- Load images as RGB, [0,1] float32 ----
        pred_img = Image.open(pred_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        pred_np = np.asarray(pred_img, dtype=np.float32) / 255.0
        gt_np = np.asarray(gt_img, dtype=np.float32) / 255.0

        # Ensure shapes match
        if pred_np.shape != gt_np.shape:
            raise ValueError(
                f"Shape mismatch for {pred_names[i]} and {gt_names[i]}: pred {pred_np.shape}, gt {gt_np.shape}"
            )

        # ---- PSNR ----
        psnr = compare_psnr(gt_np, pred_np, data_range=1.0)
        psnr_vals.append(psnr)

        # ---- SSIM ----
        # Handle both old (multichannel) and new (channel_axis) APIs.
        try:
            ssim = compare_ssim(gt_np, pred_np, data_range=1.0, channel_axis=-1)
        except TypeError:
            ssim = compare_ssim(gt_np, pred_np, data_range=1.0, multichannel=True)
        ssim_vals.append(ssim)

        # ---- MS-SSIM (expects NCHW, [0,1]) ----
        gt_t = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0)  # 1 x 3 x H x W
        pred_t = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0)

        gt_t = gt_t.to(device=device, dtype=torch.float32)
        pred_t = pred_t.to(device=device, dtype=torch.float32)

        msssim = piq.multi_scale_ssim(pred_t, gt_t, data_range=1.0).item()
        msssim_vals.append(msssim)

        # ---- LPIPS (expects [-1,1], NCHW) ----
        gt_lp = gt_t * 2.0 - 1.0
        pred_lp = pred_t * 2.0 - 1.0

        with torch.no_grad():
            lp = lpips_net(pred_lp, gt_lp).item()
        lpips_vals.append(lp)

    if not psnr_vals:
        print(
            f"[compute_metrics] No valid image pairs found between {prediction_dir} and {gt_dir}"
        )
        return

    # ---- Aggregate ----
    psnr_mean = float(np.mean(psnr_vals))
    ssim_mean = float(np.mean(ssim_vals))
    msssim_mean = float(np.mean(msssim_vals))
    lpips_mean = float(np.mean(lpips_vals))

    # ---- Store in REGISTRY_EVAL if requested ----
    if dataset_name is not None:
        if "evaluation_results" not in REGISTRY_EVAL:
            REGISTRY_EVAL["evaluation_results"] = {}
        if dataset_name not in REGISTRY_EVAL["evaluation_results"]:
            REGISTRY_EVAL["evaluation_results"][dataset_name] = {}

        REGISTRY_EVAL["evaluation_results"][dataset_name]["psnr"] = psnr_mean
        REGISTRY_EVAL["evaluation_results"][dataset_name]["ssim"] = ssim_mean
        REGISTRY_EVAL["evaluation_results"][dataset_name]["ms_ssim"] = msssim_mean
        REGISTRY_EVAL["evaluation_results"][dataset_name]["lpips"] = lpips_mean

    # ---- Print nicely ----
    tag = f" [{dataset_name}]" if dataset_name else ""
    print(
        f"Metrics{tag}: "
        f"PSNR = {psnr_mean:.2f} dB, "
        f"SSIM = {ssim_mean:.4f}, "
        f"MS-SSIM = {msssim_mean:.4f}, "
        f"LPIPS = {lpips_mean:.4f}"
    )

    return {
        "psnr": psnr_mean,
        "ssim": ssim_mean,
        "ms_ssim": msssim_mean,
        "lpips": lpips_mean,
    }


def print_evaluation_summary(eval_results: dict) -> None:
    """
    Pretty-print an evaluation summary table from a dict like:
    {
        "real": {...},
        "nature": {...},
        ...
    }
    """

    title = "Evaluation Summary"
    print()
    print(title)
    print("=" * len(title))

    # ---- Define columns ----
    headers = ["Dataset", "PSNR", "SSIM", "MS-SSIM", "LPIPS"]

    # Build rows (as strings, rounded)
    rows = []
    for name, metrics in eval_results.items():
        row = [
            name,
            f"{metrics['psnr']:.2f}",
            f"{metrics['ssim']:.4f}",
            f"{metrics['ms_ssim']:.4f}",
            f"{metrics['lpips']:.4f}",
        ]
        rows.append(row)

    # ---- Column widths ----
    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            if len(cell) > col_widths[i]:
                col_widths[i] = len(cell)

    # Build format string (no f-strings, to avoid brace issues)
    parts = []
    for i, w in enumerate(col_widths):
        align = "<" if i == 0 else ">"
        parts.append("{:" + align + str(w) + "}")
    fmt = " " + " | ".join(parts)

    # Separator line
    sep = "-+-".join("-" * w for w in col_widths)

    # ---- Print header + rows ----
    print(fmt.format(*headers))
    print(sep)
    for r in rows:
        print(fmt.format(*r))
    print()  # final blank line


def evaluate_all(device: str = "cuda", batch_size=2, num_workers=1):
    """
    Full benchmark evaluation on nature, real, SIR2_500 subsets.
    """
    # Set device in REGISTRY (default: cuda)
    dev_str = device or "cuda"
    if dev_str == "cuda" and not torch.cuda.is_available():
        print(
            "[warning] Requested device 'cuda' but CUDA is not available; falling back to 'cpu'."
        )
        dev_str = "cpu"

    # --- Download / unzip / format datasets ---
    download_nature()
    unzip_nature()
    download_real()
    unzip_real()
    download_sir2_500()
    unzip_sir2_500()
    format_sir2_500()

    # --- Run inference ---
    try:
        vae, transformer, embeds_dict, processing_resolution = load_network(
            SUPPORTED_MODEL_URIS[0], LORA_MODEL_URI, torch.device(dev_str)
        )
    except Exception as e:
        print(f"Failed to load network: {e}")
        raise

    run_inference(
        vae,
        transformer,
        embeds_dict,
        processing_resolution,
        REGISTRY_EVAL["paths"]["nature_input"],
        REGISTRY_EVAL["paths"]["nature_output"],
        save_comparison=False,
        save_alternating=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    run_inference(
        vae,
        transformer,
        embeds_dict,
        processing_resolution,
        REGISTRY_EVAL["paths"]["real_input"],
        REGISTRY_EVAL["paths"]["real_output"],
        save_comparison=False,
        save_alternating=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    run_inference(
        vae,
        transformer,
        embeds_dict,
        processing_resolution,
        REGISTRY_EVAL["paths"]["sir2_500_SolidObject_input"],
        REGISTRY_EVAL["paths"]["sir2_500_SolidObject_output"],
        save_comparison=False,
        save_alternating=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    run_inference(
        vae,
        transformer,
        embeds_dict,
        processing_resolution,
        REGISTRY_EVAL["paths"]["sir2_500_Postcard_input"],
        REGISTRY_EVAL["paths"]["sir2_500_Postcard_output"],
        save_comparison=False,
        save_alternating=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    run_inference(
        vae,
        transformer,
        embeds_dict,
        processing_resolution,
        REGISTRY_EVAL["paths"]["sir2_500_Wildscene_input"],
        REGISTRY_EVAL["paths"]["sir2_500_Wildscene_output"],
        save_comparison=False,
        save_alternating=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # --- Compute metrics ---
    compute_metrics(
        REGISTRY_EVAL["paths"]["nature_output"],
        REGISTRY_EVAL["paths"]["nature_gt"],
        dataset_name="Nature (20)",
    )
    compute_metrics(
        REGISTRY_EVAL["paths"]["real_output"],
        REGISTRY_EVAL["paths"]["real_gt"],
        dataset_name="Real (20)",
    )
    compute_metrics(
        REGISTRY_EVAL["paths"]["sir2_500_SolidObject_output"],
        REGISTRY_EVAL["paths"]["sir2_500_SolidObject_gt"],
        dataset_name="SIR2-500 SolidObject (200)",
    )
    compute_metrics(
        REGISTRY_EVAL["paths"]["sir2_500_Postcard_output"],
        REGISTRY_EVAL["paths"]["sir2_500_Postcard_gt"],
        dataset_name="SIR2-500 Postcard (199)",
    )
    compute_metrics(
        REGISTRY_EVAL["paths"]["sir2_500_Wildscene_output"],
        REGISTRY_EVAL["paths"]["sir2_500_Wildscene_gt"],
        dataset_name="SIR2-500 Wildscene (101)",
    )

    # --- Weighted SIR2_500 and summary ---
    REGISTRY_EVAL["evaluation_results"]["sir2_500_weighted"] = (
        compute_sir2_500_weighted(REGISTRY_EVAL["evaluation_results"])
    )
    print_evaluation_summary(REGISTRY_EVAL["evaluation_results"])


def evaluate_custom(
    input_folder: str,
    gt_folder: str,
    output_folder: str,
    device: str = "cuda",
    batch_size=2,
    num_workers=1,
):
    """
    Evaluate on a custom set of folders:
      - input_folder: blended / input images
      - gt_folder: corresponding ground-truth images
      - output_folder: where predictions will be written

    Only this dataset will be evaluated.
    """
    # Normalize device (default: cuda)
    dev_str = device or "cuda"
    if dev_str == "cuda" and not torch.cuda.is_available():
        print(
            "[warning] Requested device 'cuda' but CUDA is not available; falling back to 'cpu'."
        )
        dev_str = "cpu"

    os.makedirs(output_folder, exist_ok=True)

    # Run inference on the custom folder
    try:
        vae, transformer, embeds_dict, processing_resolution = load_network(
            SUPPORTED_MODEL_URIS[0], LORA_MODEL_URI, torch.device(dev_str)
        )
    except Exception as e:
        print(f"Failed to load network: {e}")
        raise

    run_inference(
        vae,
        transformer,
        embeds_dict,
        processing_resolution,
        input_folder,
        output_folder,
        save_comparison=False,
        save_alternating=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Compute metrics, store as "custom"
    compute_metrics(output_folder, gt_folder, dataset_name="custom")

    # Print summary table (only 'custom' in there)
    print_evaluation_summary(REGISTRY_EVAL["evaluation_results"])


def main():
    parser = argparse.ArgumentParser(
        description="WindowSeat reflection removal evaluation."
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        default=None,
        help="Folder with input/blended images for custom evaluation.",
    )
    parser.add_argument(
        "--gt-folder",
        type=str,
        default=None,
        help="Folder with ground-truth images for custom evaluation.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Folder where predictions for custom evaluation will be written.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device string for inference, e.g. 'cuda', 'cpu'. Default: cuda.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size. ",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Num workers. ",
    )
    args = parser.parse_args()

    # If *all three* custom folders are provided -> custom eval only
    if args.input_folder and args.gt_folder and args.output_folder:
        evaluate_custom(
            input_folder=args.input_folder,
            gt_folder=args.gt_folder,
            output_folder=args.output_folder,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    # If some but not all are provided -> error
    elif any([args.input_folder, args.gt_folder, args.output_folder]):
        parser.error(
            "When using custom evaluation, you must provide "
            "--input-folder, --gt-folder and --output-folder together."
        )
    else:
        # No custom folders -> full benchmark evaluation
        evaluate_all(
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
