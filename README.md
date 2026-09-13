<h1 style="text-align:center;">🪟 WindowSeat: Reflection Removal through Efficient Adaptation of Diffusion Transformers</h1>

<p align="center">
  <a href="https://hf.co/spaces/huawei-bayerlab/windowseat-reflection-removal-web"><img src="assets/shields/project-website.svg"></a>
  <a href="https://arxiv.org/abs/2512.05000"><img src="assets/shields/arxiv-pdf.svg"></a>
  <a href="https://huggingface.co/huawei-bayerlab/windowseat-reflection-removal-v1-0"><img src="assets/shields/huggingface-model-yellow.svg"></a>
  <a href="LICENSE.txt"><img src="assets/shields/license-apache-2.0.svg"></a>
</p>

<p align="center">
  <b>Daniyar Zakarin</b><sup>*1,2</sup>,
  <b>Thiemo Wandel</b><sup>*2</sup>,
  <b>Anton Obukhov</b><sup>†1,2</sup>,
  <b>Dengxin Dai</b><sup>2</sup>
  <br>
  <sub>
    <sup>1</sup>ETH Zurich &nbsp;&nbsp;|&nbsp;&nbsp;
    <sup>2</sup>Huawei Bayer Lab &nbsp;&nbsp;|&nbsp;&nbsp;
    <sup>*</sup>Equal contributors &nbsp;&nbsp;|&nbsp;&nbsp;
    <sup>†</sup>Project lead
  </sub>
</p>

![Teaser](./doc/images/windowseat_teaser.jpg)

## News

2026-09-13: Mirrored on ModelScope:
<sub><a href="https://www.modelscope.cn/studios/huawei-bayerlab/windowseat-reflection-removal-web" target="_blank" rel="noopener noreferrer"><img src="assets/shields/modelscope-website.svg" height="17" alt="ModelScope website"></a>
<a href="https://www.modelscope.cn/models/huawei-bayerlab/windowseat-reflection-removal-v1-0" target="_blank" rel="noopener noreferrer"><img src="assets/shields/modelscope-model.svg" height="17" alt="ModelScope model"></a>
<a href="https://www.modelscope.cn/studios/huawei-bayerlab/windowseat-reflection-removal" target="_blank" rel="noopener noreferrer"><img src="assets/shields/modelscope-demo.svg" height="17" alt="ModelScope demo"></a></sub><br>
2026-06-04: Presented at the NTIRE workshop at CVPR 2026.<br>
2025-12-05: Initial release: inference code, the released checkpoint, and the demo.<br>

## Visualizations

![Visualizations](./doc/images/visualizations.png)

## Quick Start
### Environment & Requirements

Code tested on a **CUDA GPU with 24 GB VRAM**.

Create and activate the environment:
```bash
git clone https://github.com/huawei-bayerlab/windowseat-reflection-removal.git
cd windowseat-reflection-removal
conda env create -f environment.yaml
conda activate windowseat
```

### Inference: Removing Reflections with WindowSeat

```bash
python windowseat_inference.py
```

This will:

- ⬇️ Download the Qwen-Image-Edit 2509 backbone and WindowSeat LoRA from Hugging Face. If you are prompted to log in, please provide a read access token from Hugging Face → Settings → Access Tokens. When asked 'Add token as git credential? (Y/n)', select 'n'. 
- 🪞 Remove reflections from input images (example_images)
- 💾 Save predictions to an output folder (outputs)  

You can also pass your own directories:

```bash
python windowseat_inference.py \
  --input-dir /path/to/your/input_images \
  --output-dir /path/to/save_predictions
```

By default, the script uses the short edge of the image as the tile size. To increase the number of tiles for high resolution images, you can set:
```bash
python windowseat_inference.py --more-tiles
```

## Reproducibility

![Results Table](./doc/images/both_result_tables.png)

To reproduce the WindowSeat (ours, Apache 2.0) numbers, run 
```bash 
python windowseat_reproducibility.py
```
This will: 
- ⬇️ Download Nature, Real, and SIR2 500 datasets to data/evaluation/test_datasets
- 🪞 Remove reflections from all datasets
- ⚖️ Compute metrics between prediction images and ground truth

Metrics may vary slightly due to nondeterminism from the ones reported in the tables. 
To validate on an already downloaded dataset or a custom dataset, you can pass additional arguments:
```bash
python windowseat_reproducibility.py --input-folder /path/to/input-images --output-folder /path/to/output --gt-folder /path/to/ground_truth 
```

By default, the evaluation will use batch size of 2 and number of workers of 1. You can increase them with cli arguments:

```bash
python windowseat_reproducibility.py --batch-size=4 --num-workers=4
```

## Troubleshooting

| Problem                                                                                                                                      | Solution                                                       |
|----------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| (pip) Errors installing requirements via `pip install -r requirements.txt` | `python -m pip install --upgrade pip` |
| (pip) Other package errors | Use conda environment with pinned package versions: `conda env create -f environment.yaml` |

## Citation
Please cite our paper:

```bibtex
@InProceedings{Zakarin_2026_CVPR,
  author    = {Zakarin, Daniyar and Wandel, Thiemo and Obukhov, Anton and Dai, Dengxin},
  title     = {Reflection Removal through Efficient Adaptation of Diffusion Transformers},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year      = {2026},
  pages     = {2776--2785}
}
```

## License

The code and models of this work are licensed under the Apache License, Version 2.0.
By downloading and using the code and model you agree to the terms in [LICENSE](LICENSE.txt).
