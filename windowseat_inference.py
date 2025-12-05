import argparse
import functools
import json
import math
import os
import sys
import warnings

import imageio.v2 as imageio
import numpy as np
import safetensors
import torch
import torchvision
from diffusers import (
    AutoencoderKLQwenImage,
    BitsAndBytesConfig,
    QwenImageEditPipeline,
    QwenImageTransformer2DModel,
)
from huggingface_hub import hf_hub_download
from peft import LoraConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SUPPORTED_MODEL_URIS = [
    "Qwen/Qwen-Image-Edit-2509",
]
LORA_MODEL_URI = "huawei-bayerlab/windowseat-reflection-removal-v1-0"


def fetch_state_dict(
    pretrained_model_name_or_path_or_dict: str,
    weight_name: str,
    use_safetensors: bool = True,
    subfolder: str | None = None,
):
    file_path = hf_hub_download(pretrained_model_name_or_path_or_dict, weight_name, subfolder=subfolder)
    if use_safetensors:
        state_dict = safetensors.torch.load_file(file_path)
    else:
        state_dict = torch.load(file_path, weights_only=True)
    return state_dict


def load_qwen_vae(uri: str, device: torch.device):
    vae = AutoencoderKLQwenImage.from_pretrained(
        uri,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    vae.to(device, dtype=torch.bfloat16)
    return vae


def load_qwen_transformer(uri: str, device: torch.device):
    nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
    )

    transformer = QwenImageTransformer2DModel.from_pretrained(
        uri,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        quantization_config=nf4,
        device_map=device,
    )

    return transformer


def load_lora_into_transformer(uri: str, transformer: QwenImageTransformer2DModel):
    lora_config = LoraConfig.from_pretrained(uri, subfolder="transformer_lora")
    transformer.add_adapter(lora_config)
    state_dict = fetch_state_dict(uri, "pytorch_lora_weights.safetensors", subfolder="transformer_lora")
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    if len(unexpected) > 0:
        raise ValueError(f"Unexpected keys in transformer state dict: {unexpected}")
    return transformer


def load_embeds_dict(uri: str):
    embeds_dict = fetch_state_dict(uri, "state_dict.safetensors", subfolder="text_embeddings")
    return embeds_dict


def load_network(uri_base: str, uri_lora: str, device: torch.device):
    config_file = hf_hub_download(uri_lora, "model_index.json")
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    base_model_uri = config_dict["base_model"]
    processing_resolution = config_dict["processing_resolution"]
    if base_model_uri not in SUPPORTED_MODEL_URIS:
        raise ValueError(f"Unsupported base model URI: {base_model_uri}")

    vae = load_qwen_vae(uri_base, device)
    transformer = load_qwen_transformer(uri_base, device)
    load_lora_into_transformer(uri_lora, transformer)
    embeds_dict = load_embeds_dict(uri_lora)
    return vae, transformer, embeds_dict, processing_resolution


def encode(image: torch.Tensor, vae: AutoencoderKLQwenImage) -> torch.Tensor:
    image = image.to(device=vae.device, dtype=vae.dtype)
    out = vae.encode(image.unsqueeze(2)).latent_dist.sample()
    latents_mean = torch.tensor(vae.config.latents_mean, device=out.device, dtype=out.dtype)
    latents_mean = latents_mean.view(1, vae.config.z_dim, 1, 1, 1)
    latents_std_inv = 1.0 / torch.tensor(vae.config.latents_std, device=out.device, dtype=out.dtype)
    latents_std_inv = latents_std_inv.view(1, vae.config.z_dim, 1, 1, 1)
    out = (out - latents_mean) * latents_std_inv
    return out


def decode(latents: torch.Tensor, vae: AutoencoderKLQwenImage) -> torch.Tensor:
    latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
    latents_mean = latents_mean.view(1, vae.config.z_dim, 1, 1, 1)
    latents_std_inv = (1.0 / torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype))
    latents_std_inv = latents_std_inv.view(1, vae.config.z_dim, 1, 1, 1)
    latents = latents / latents_std_inv + latents_mean
    out = vae.decode(latents)
    out = out.sample[:, :, 0]
    return out


def _match_batch(t: torch.Tensor, B: int) -> torch.Tensor:
    if t.size(0) == B:
        return t
    if t.size(0) == 1 and B > 1:
        return t.expand(B, *t.shape[1:])
    if t.size(0) > B:
        return t[:B]
    reps = (B + t.size(0) - 1) // t.size(0)
    return t.repeat((reps,) + (1,) * (t.ndim - 1))[:B]


def flow_step(
    model_input: torch.Tensor, 
    transformer: QwenImageTransformer2DModel, 
    vae: AutoencoderKLQwenImage,
    embeds_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    prompt_embeds = embeds_dict["prompt_embeds"]  # [N_ctx, L, D]
    prompt_mask = embeds_dict["prompt_mask"]  # [N_ctx, L]
    
    if prompt_mask.dtype != torch.bool:
        prompt_mask = prompt_mask > 0

    # Accept [B, C, 1, H, W] or [B, C, H, W]
    if model_input.ndim == 5 and model_input.shape[2] == 1:
        model_input_4d = model_input[:, :, 0]  # [B, C, H, W]
    elif model_input.ndim == 4:
        model_input_4d = model_input
    else:
        raise ValueError(f"Unexpected lat_encoding shape: {model_input.shape}")

    B, C, H, W = model_input_4d.shape
    device = next(transformer.parameters()).device

    prompt_embeds = _match_batch(prompt_embeds, B).to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )  # [B, L, D]

    prompt_mask = _match_batch(prompt_mask, B).to(
        device=device, dtype=torch.bool, non_blocking=True
    )  # [B, L]

    num_channels_latents = C
    packed_model_input = QwenImageEditPipeline._pack_latents(
        model_input_4d,
        batch_size=B,
        num_channels_latents=num_channels_latents,
        height=H,
        width=W,
    )  # [B, N_patches, C * 4], where N_patches = (H // 2) * (W // 2)
    packed_model_input = packed_model_input.to(torch.bfloat16)

    t_const = 499
    timestep = torch.full(
        (B,),
        float(t_const),
        device=device,
        dtype=torch.bfloat16,
    )
    timestep = timestep / 1000.0

    h_img = H // 2
    w_img = W // 2

    img_shapes = [[(1, h_img, w_img)]] * B
    txt_seq_lens = prompt_mask.sum(dim=1).tolist() if prompt_mask is not None else None

    if getattr(transformer, "attention_kwargs", None) is None:
        attention_kwargs = {}
    else:
        attention_kwargs = transformer.attention_kwargs

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model_pred = transformer(
            hidden_states=packed_model_input,  # [B, N_patches, C*4]
            timestep=timestep,  # [B], float / 1000
            encoder_hidden_states=prompt_embeds,  # [B, L, D]
            encoder_hidden_states_mask=prompt_mask,  # [B, L]
            img_shapes=img_shapes,  # single stream per batch
            txt_seq_lens=txt_seq_lens,
            guidance=None,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]  # [B, N_patches, C*4]

    temperal_downsample = vae.config.get("temperal_downsample", None)
    if temperal_downsample is not None:
        vae_scale_factor = 2 ** len(temperal_downsample)
    else:
        vae_scale_factor = 8

    model_pred = QwenImageEditPipeline._unpack_latents(
        model_pred,
        height=H * vae_scale_factor,  # H, W here are latent H,W from encode
        width=W * vae_scale_factor,
        vae_scale_factor=vae_scale_factor,
    )  # [B, C, 1, H_lat, W_lat]

    latent_output = model_input.to(vae.dtype) - model_pred.to(vae.dtype)

    return latent_output


def _supports_color() -> bool:
    return sys.stdout.isatty()


def _style(text: str, *, color: str | None = None, bold: bool = False) -> str:
    if not _supports_color():
        return text

    codes = []
    if bold:
        codes.append("1")
    if color == "red":
        codes.append("31")
    elif color == "green":
        codes.append("32")
    elif color == "yellow":
        codes.append("33")
    elif color == "blue":
        codes.append("34")
    elif color == "magenta":
        codes.append("35")
    elif color == "cyan":
        codes.append("36")

    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def print_banner(title: str):
    title = f" {title} "
    bar = "═" * len(title)
    print(_style(f"╔{bar}╗", color="cyan", bold=True))
    print(_style(f"║{title}║", color="cyan", bold=True))
    print(_style(f"╚{bar}╝", color="cyan", bold=True))


def print_step(step: str, msg: str):
    prefix = _style(f"[{step}] ", color="yellow", bold=True)
    print(prefix + msg)


def print_ok(msg: str):
    print(_style("✔ ", color="green", bold=True) + msg)


def print_info(msg: str):
    print(_style("ℹ ", color="blue", bold=True) + msg)


def print_error(msg: str):
    print(_style("✖ ", color="red", bold=True) + msg)


def print_final_success(output_dir: str):
    print_ok("Inference finished successfully!")
    print_info("Predictions have been written to:")
    print("   " + _style(output_dir, color="cyan", bold=True))
    print(_style("Thank you for trying out WindowSeat! 🪟", color="green"))


def _required_side_for_axis(size: int, nmax: int, min_overlap: int) -> int:
    """Smallest tile side T (1D) so that #tiles <= nmax with overlap >= min_overlap."""
    nmax = max(1, int(nmax))
    if nmax == 1:
        return size
    return math.ceil((size + (nmax - 1) * min_overlap) / nmax)


def _starts(size: int, T: int, min_overlap: int):
    """Uniform stepping with stride = T - min_overlap; last tile flush with edge."""
    if size <= T:
        return [0]
    stride = max(1, T - min_overlap)
    xs = list(range(0, size - T + 1, stride))
    last = size - T
    if xs[-1] != last:
        xs.append(last)
    # monotonic dedupe
    out = []
    for v in xs:
        if not out or v > out[-1]:
            out.append(v)
    return out


class TilingDataset(Dataset):
    def __init__(
        self,
        transform_graph,
        input_folder,
        tiling_w=768,
        tiling_h=768,
        processing_resolution=768,
        max_num_tiles_w=4,
        max_num_tiles_h=4,
        min_overlap_w=64,
        min_overlap_h=64,
        use_short_edge_tile=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.transform_graph = transform_graph
        self.kwargs = kwargs
        self.disp_name = kwargs.get("disp_name", "tiling_dataset")

        img_paths = sorted(
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if os.path.isfile(os.path.join(input_folder, f))
        )

        self.filenames = []

        Nw, Nh = int(max_num_tiles_w), int(max_num_tiles_h)
        ow, oh = int(min_overlap_w), int(min_overlap_h)

        for i, p in enumerate(img_paths):
            with Image.open(p) as im:
                W, H = im.size

                # Choose preferred tile size for this image
                if use_short_edge_tile:
                    short_edge = min(W, H)
                    short_edge = max(short_edge, processing_resolution)
                    tiling_w_i = short_edge
                    tiling_h_i = short_edge
                else:
                    tiling_w_i = tiling_w
                    tiling_h_i = tiling_h

                # Optional upscaling if image is smaller than desired tile
                if W < tiling_w_i or H < tiling_h_i:
                    min_side = min(W, H)
                    scale_ratio = tiling_w_i / min_side
                    W = round(scale_ratio * W)
                    H = round(scale_ratio * H)

            pref_side = max(int(tiling_w_i), int(tiling_h_i))

            # Feasible square-side interval [T_low, T_high]
            T_low = max(
                _required_side_for_axis(W, Nw, ow),
                _required_side_for_axis(H, Nh, oh),
                ow + 1,
                oh + 1,
            )
            T_high = min(W, H)

            if T_low > T_high:
                msg = (
                    f"Infeasible square constraints for {os.path.basename(p)}: "
                    f"need T >= {T_low}, but max square inside is {T_high}. "
                    f"Relax max_num_tiles_w/h or overlaps, allow non-square tiles, or pad."
                )
                raise ValueError(msg)
            else:
                T = max(T_low, min(pref_side, T_high))
                Tw = Th = T

            # Build starts with axis-specific tile sizes
            xs = _starts(W, Tw, ow)
            ys = _starts(H, Th, oh)

            for y0 in ys:
                for x0 in xs:
                    x1, y1 = x0 + Tw, y0 + Th
                    self.filenames.append([str(p), (x0, y0, x1, y1), False])

            if self.filenames:
                self.filenames[-1][-1] = True

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        sample = {}
        sample["line"] = self.filenames[index]
        sample["idx"] = index
        self.transform_graph(sample)
        return sample


def read_scalars(sample):
    scalar_dict = {"tile_info": 1, "is_last_tile": 2}
    for name, col in scalar_dict.items():
        sample[name] = sample["line"][col]


def load_rgb_data(rgb_path, key_prefix="input"):
    rgb = read_rgb_file(rgb_path)
    rgb_norm = rgb / 255.0 * 2.0 - 1.0
    outputs = {
        f"{key_prefix}_int": torch.from_numpy(rgb).int(),
        f"{key_prefix}_norm": torch.from_numpy(rgb_norm),
    }
    return outputs


def read_rgb_file(rgb_path) -> np.ndarray:
    img = Image.open(rgb_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)  # [H, W, 3]
    return arr.transpose(2, 0, 1)  # [3, H, W]


def read_rgb_image(sample):
    column = 0
    name = "input"

    img_path = sample["line"][column]
    img = load_rgb_data(img_path, name)
    sample.update(img)
    sample.setdefault("meta", {})
    sample["meta"]["orig_res"] = [
        sample[name + "_norm"].shape[-2],
        sample[name + "_norm"].shape[-1],
    ]


def _lanczos_resize_chw(x, out_hw):
    H_out, W_out = map(int, out_hw)

    is_torch = isinstance(x, torch.Tensor)
    if is_torch:
        dev = x.device
        arr = x.detach().cpu().numpy()
    else:
        arr = x

    assert isinstance(arr, np.ndarray) and arr.ndim == 3, "expect CHW"
    chw = arr.astype(np.float32, copy=False)
    C, _, _ = chw.shape

    out_chw = np.empty((C, H_out, W_out), dtype=np.float32)
    for c in range(C):
        ch = chw[c]
        img = Image.fromarray(ch).convert("F")
        img = img.resize((W_out, H_out), resample=Image.LANCZOS)
        out_chw[c] = np.asarray(img, dtype=np.float32)

    if is_torch:
        return torch.from_numpy(out_chw).to(dev)
    return out_chw


def reshape(sample, height, width):
    Ht, Wt = height, width
    for k, v in list(sample.items()):
        if not (torch.is_tensor(v) and v.ndim >= 2) or "orig" in k:
            continue
        x = v.to(torch.float32)
        x = _lanczos_resize_chw(x, (Ht, Wt))
        if v.dtype == torch.bool:
            x = x > 0.5
        elif not torch.is_floating_point(v):
            x = x.round().to(v.dtype)
        sample[k] = x

    return sample


def tile(sample, processing_resolution: int):
    x0, y0, x1, y1 = map(int, sample["tile_info"])
    processing_width = x1 - x0
    processing_height = y1 - y0

    # Reshape input while keeping aspect ratio
    H, W = sample["input_norm"].shape[-2:]
    if W < processing_width or H < processing_height:
        min_side = min(W, H)
        scale_ratio = processing_width / min_side
        W = round(scale_ratio * W)
        H = round(scale_ratio * H)

    reshape(sample, height=H, width=W)
    sample["input_int"] = sample["input_int"][:, y0:y1, x0:x1]
    sample["input_norm"] = sample["input_norm"][:, y0:y1, x0:x1]
    reshape(sample, height=processing_resolution, width=processing_resolution)


@torch.no_grad()
def validate_single_dataset(
    vae: AutoencoderKLQwenImage,
    transformer: QwenImageTransformer2DModel,
    embeds_dict: dict[str, torch.Tensor],
    data_loader: DataLoader,
    save_to_dir: str = None,
    save_comparison: bool = True,
    save_alternating: bool = True,
):
    preds = []

    for i, batch in enumerate(
        tqdm(data_loader, desc=f"Reflection Removal Progress"),
        start=1,
    ):
        batch["out"] = {}
        with torch.no_grad():
            latents = encode(batch["input_norm"], vae)
            latents = flow_step(latents, transformer, vae, embeds_dict)
            batch["out"]["pixel_pred"] = decode(latents, vae)

        for b in range(len(batch["idx"])):
            preds.append(
                {
                    "file": batch["line"][0][b],

                    # [x0, y0, x1, y1] tuple for the tile
                    "tile_info": [batch["tile_info"][i][b] for i in range(4)],
                    
                    # Shape 1, 3, H, W, torch tensor in range -1 to 1
                    "pred": batch["out"]["pixel_pred"][b].to("cpu"),
                }
            )

            if batch["is_last_tile"][b]:
                # Stitch predictions together
                W = max(int(t["tile_info"][2]) for t in preds)
                H = max(int(t["tile_info"][3]) for t in preds)

                acc = torch.zeros(3, H, W, dtype=torch.float32)
                wsum = torch.zeros(H, W, dtype=torch.float32)

                for t in preds:
                    tile_info = [t["tile_info"][i] for i in range(4)]
                    x0, y0, x1, y1 = map(int, tile_info)
                    tile = t["pred"].squeeze(0).float()  # [3, h, w], [-1,1]

                    h, w = tile.shape[-2:]
                    tH, tW = (y1 - y0), (x1 - x0)
                    if (h != tH) or (w != tW):
                        tile = _lanczos_resize_chw(tile, (tH, tW))
                        h, w = tH, tW

                    # triangular window for the tile
                    # fmt: off
                    wx = 1 - (2 * torch.arange(w, dtype=torch.float32) / (max(w - 1, 1)) - 1).abs()
                    wy = 1 - (2 * torch.arange(h, dtype=torch.float32) / (max(h - 1, 1)) - 1).abs()
                    # fmt: on
                    w2 = (wy[:, None] * wx[None, :]).clamp_min(1e-3)
                    acc[:, y0:y1, x0:x1] += tile * w2
                    wsum[y0:y1, x0:x1] += w2
                stitched = (acc / wsum.clamp_min(1e-6)).unsqueeze(0)  # [1,3,H,W], [-1,1]

                # Lanczos resize to gt_orig shape
                orig_H, orig_W = (
                    batch["meta"]["orig_res"][0][b].item(),
                    batch["meta"]["orig_res"][1][b].item(),
                )

                x = stitched.squeeze(0)
                x01 = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
                device = x01.device

                pil = torchvision.transforms.functional.to_pil_image(x01.cpu())
                pil_resized = pil.resize((orig_W, orig_H), resample=Image.LANCZOS)
                pred_ts = torchvision.transforms.functional.to_tensor(pil_resized).to(device)  # [3,H,W], [0,1]
                pred = pred_ts.cpu().numpy()
                preds = []
            else:
                continue

            pred_ts = torch.from_numpy(pred).to(device)  # [3,H,W]
            scene_path = batch["line"][0][b]
            scene_name = scene_path.split("/")[-1][:-4]

            # Load original input image (CHW, uint8 in [0,255])
            input_chw = read_rgb_file(scene_path)
            input_hwc = (
                np.transpose(input_chw, (1, 2, 0)).astype(np.float32) / 255.0
            )  # [H,W,3], [0,1]

            pred_hwc = np.transpose(pred, (1, 2, 0))
            if input_hwc.shape[:2] != pred_hwc.shape[:2]:
                pil_pred = Image.fromarray(
                    (pred_hwc.clip(0, 1) * 255).round().astype(np.uint8)
                )
                H_in, W_in = input_hwc.shape[:2]
                pil_pred = pil_pred.resize((W_in, H_in), resample=Image.LANCZOS)
                pred_hwc = (np.array(pil_pred, dtype=np.uint8) / 255.0).clip(0, 1)

            visualize(
                file_prefix=scene_name,
                input_hwc=input_hwc,
                pred_hwc=pred_hwc,
                output_dir=save_to_dir,
                save_comparison=save_comparison,
                save_alternating=save_alternating,
            )

    return


def save_prediction_only(
    file_prefix: str,
    pred_uint8: np.ndarray,
    output_dir: str,
) -> None:
    imageio.imwrite(
        os.path.join(output_dir, f"{file_prefix}_windowseat_output.png"),
        pred_uint8,
        plugin="pillow",
    )


def save_comparison_image(
    file_prefix: str,
    pred_uint8: np.ndarray,
    input_uint8: np.ndarray,
    output_dir: str,
    margin_width: int = 10,
) -> None:
    H_in, W_in, _ = input_uint8.shape
    if pred_uint8.shape[:2] != (H_in, W_in):
        pil_pred = Image.fromarray(pred_uint8)
        pil_pred = pil_pred.resize((W_in, H_in), resample=Image.LANCZOS)
        pred_uint8 = np.asarray(pil_pred, dtype=np.uint8)

    margin = np.ones((H_in, margin_width, 3), dtype=np.uint8) * 255
    comparison = np.concatenate([input_uint8, margin, pred_uint8], axis=1)

    imageio.imwrite(
        os.path.join(output_dir, f"{file_prefix}_windowseat_side_by_side.png"),
        comparison,
        plugin="pillow",
    )


def save_alternating_video(
    file_prefix: str,
    input_uint8: np.ndarray,
    pred_uint8: np.ndarray,
    output_dir: str,
    fps: float = 1.0,
    total_frames: int = 20,
) -> None:
    video_path = os.path.join(output_dir, f"{file_prefix}_windowseat_alternating.mp4")

    H, W = input_uint8.shape[:2]
    pad_h = (0, H % 2)
    pad_w = (0, W % 2)
    if pad_h[1] or pad_w[1]:
        input_uint8 = np.pad(input_uint8, (pad_h, pad_w, (0, 0)), mode="edge")
        pred_uint8 = np.pad(pred_uint8, (pad_h, pad_w, (0, 0)), mode="edge")

    with imageio.get_writer(
        video_path, fps=fps, macro_block_size=1, ffmpeg_params=["-loglevel", "quiet"]
    ) as writer:
        for i in range(total_frames):
            frame = input_uint8 if i % 2 == 0 else pred_uint8
            writer.append_data(frame)


def visualize(
    file_prefix: str,
    input_hwc: np.ndarray,
    pred_hwc: np.ndarray,
    output_dir: str,
    save_comparison: bool = True,
    save_alternating: bool = True,
) -> None:
    pred_hwc = pred_hwc.clip(0, 1)
    pred_uint8 = (pred_hwc * 255).round().astype(np.uint8)
    input_hwc = np.asarray(input_hwc, dtype=np.float32)
    if input_hwc.max() > 1.0:
        input_hwc = input_hwc / 255.0
    input_uint8 = (input_hwc.clip(0, 1) * 255).round().astype(np.uint8)

    save_prediction_only(
        file_prefix=file_prefix,
        pred_uint8=pred_uint8,
        output_dir=output_dir,
    )

    if save_comparison:
        save_comparison_image(
            file_prefix=file_prefix,
            pred_uint8=pred_uint8,
            input_uint8=input_uint8,
            output_dir=output_dir,
        )

    if save_alternating:
        save_alternating_video(
            file_prefix=file_prefix,
            input_uint8=input_uint8,
            pred_uint8=pred_uint8,
            output_dir=output_dir,
        )


def data_transform(sample, processing_resolution=None):
    read_scalars(sample)
    read_rgb_image(sample)
    tile(sample, processing_resolution)


def run_inference(
    vae: AutoencoderKLQwenImage,
    transformer: QwenImageTransformer2DModel,
    embeds_dict: dict[str, torch.Tensor],
    processing_resolution: int,
    image_dir: str,
    output_dir: str,
    use_short_edge_tile=True,
    save_comparison=True,
    save_alternating=True,
    batch_size=2,
    num_workers=0,
):
    dataset = TilingDataset(
        transform_graph=functools.partial(data_transform, processing_resolution=processing_resolution),
        input_folder=image_dir,
        gt_folder=image_dir,
        use_short_edge_tile=use_short_edge_tile,
        tiling_w=processing_resolution,
        tiling_h=processing_resolution,
        processing_resolution=processing_resolution,
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    os.makedirs(output_dir, exist_ok=True)

    validate_single_dataset(
        vae,
        transformer,
        embeds_dict,
        data_loader=data_loader,
        save_to_dir=output_dir,
        save_comparison=save_comparison,
        save_alternating=save_alternating,
    )


def parse_args():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_DIR = os.path.join(SCRIPT_DIR, "example_images")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

    parser = argparse.ArgumentParser(
        description="WindowSeat: reflection removal inference"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=IMAGE_DIR,
        help="Directory with input images (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to write predictions (default: %(default)s)",
    )
    parser.add_argument(
        "--uri-base",
        type=str,
        default=SUPPORTED_MODEL_URIS[0],
        help="URI of the base model (default: %(default)s)",
    )
    parser.add_argument(
        "--uri-lora",
        type=str,
        default=LORA_MODEL_URI,
        help="URI of the LoRA model (default: %(default)s)",
    )
    parser.add_argument(
        "--more-tiles",
        action="store_true",
        help="Use more tiles for processing.",
    )
    parser.add_argument(
        "--no-save-comparison",
        dest="save_comparison",
        action="store_false",
        help="Do NOT save comparison image between input and prediction.",
    )
    parser.add_argument(
        "--no-save-alternating",
        dest="save_alternating",
        action="store_false",
        help="Do NOT save alternating video.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device used for inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = args.input_dir
    output_dir = args.output_dir
    uri_base = args.uri_base
    uri_lora = args.uri_lora
    use_short_edge_tile = not args.more_tiles
    save_comparison = args.save_comparison
    save_alternating = args.save_alternating
    device = torch.device(args.device)
    if device != torch.device("cuda"):
        warnings.warn(
            f"WindowSeat inference was only tested with 'cuda'. "
            f"Device {device} is not officially supported and may be slow or fail."
        )

    if not os.path.isdir(image_dir):
        print_error(f"Input image directory does not exist: {image_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print_banner("WindowSeat: Reflection Removal")
    print_step("1/2", "Loading network components:")
    print_info(f"Base:        {uri_base}")
    print_info(f"WindowSeat:  {uri_lora}")

    try:
        vae, transformer, embeds_dict, processing_resolution = load_network(uri_base, uri_lora, device)
    except Exception as e:
        print_error(f"Failed to load network: {e}")
        raise

    print_step("2/2", f"Running reflection removal inference on: {image_dir}")
    run_inference(
        vae, transformer, embeds_dict, processing_resolution, image_dir, output_dir, use_short_edge_tile, save_comparison, save_alternating
    )
    print_final_success(output_dir)


if __name__ == "__main__":
    main()
