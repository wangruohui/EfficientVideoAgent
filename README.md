# [Paper Title Placeholder]

[![Paper](https://img.shields.io/badge/Paper-Link-b31b1b.svg)](Paper_Link_Placeholder)
[![Model](https://img.shields.io/badge/Model-Link-blue.svg)](Model_Link_Placeholder)

This repository contains the official evaluation code for the model proposed in our paper.

## 1. Paper and Model

- Paper Title: `Paper Title Placeholder`
- Paper Link: `Paper_Link_Placeholder`
- Model Link: `Model_Link_Placeholder`
- Repository Name: `Repo_Name_Placeholder`

## 2. Reference Results (Placeholder)

Reference result files are provided in this repository (`results placeholder`).
You can compute accuracy with `accuracy.py`:

```bash
python accuracy.py <result_jsonl_path>
```

Reference results table: `Placeholder`

## 3. Run Your Own Evaluation

### Step 1. Clone the Repository

```bash
git clone Repo_URL_Placeholder
cd Repo_Name_Placeholder
```

### Step 2. Download Model and Install Dependencies

1. Download model weights: `Model_Download_Placeholder`
2. Install dependencies from `requirements.txt` (recommended: `uv`)
3. Install FFmpeg following `https://www.ffmpeg.org/download.html`, ensure `ffprobe` is in `PATH`, and ensure FFmpeg shared libraries are in `LD_LIBRARY_PATH`.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Step 3. Serve the Model with vLLM (Multi-GPU Data Parallel)

You can refer to `deploy` for the full job wrapper.  
The core serving command is:

```bash
vllm serve <MODEL_PATH_OR_HF_ID> \
  --data-parallel-size <NUM_GPUS> \
  --limit-mm-per-prompt '{"image": 9999, "video":0}' \
  --mm_processor_cache_gb 20 \
  --allowed-local-media-path <LOCAL_MEDIA_ROOT>
```

Example (4 GPUs):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve <MODEL_PATH_OR_HF_ID> \
  --data-parallel-size 4 \
  --limit-mm-per-prompt '{"image": 9999, "video":0}' \
  --mm_processor_cache_gb 20 \
  --allowed-local-media-path <LOCAL_MEDIA_ROOT>
```

### Step 4. Configure `eval-eva.py` and Run Evaluation

Before running, edit the config section at the top of `eval-eva.py`:

- `BASE_URL`: OpenAI-compatible endpoint for your vLLM server (for example, `http://localhost:8000/v1`).
- `API_KEY`: API key used by the client (can be a dummy value for local vLLM setups if authentication is disabled).
- `MODEL_TOKENIZER_PATH`: tokenizer path used to format/chat-template prompts consistently with your model.
- `FRAME_TOOL_PATH`: path to the frame selection tool script (default is `select_frame_fallback.py`).
- `FRAME_SAVE_ROOT`: directory where extracted frames are saved during tool calls.
- `DATASET_CONFIG`: per-dataset I/O configuration.
- `DATASET_CONFIG[*].jsonl`: input dataset annotation file.
- `DATASET_CONFIG[*].video_root`: root directory containing raw video files.
- `DATASET_CONFIG[*].cache`: incremental cache file used during running.
- `DATASET_CONFIG[*].result`: final merged output file written at the end.

Run one dataset:

```bash
python eval-eva.py --dataset videomme
python eval-eva.py --dataset lsdbench
python eval-eva.py --dataset lvbench
python eval-eva.py --dataset videoholmes
python eval-eva.py --dataset longvideobench
python eval-eva.py --dataset mlvu
```

Run all supported datasets with `batch.sh`:

```bash
bash batch.sh
```

## 4. Output Files and Cache/Resume Mechanism

- Output naming is controlled by `DATASET_CONFIG` in `eval-eva.py`.
- By default, each dataset writes:
- `cache_*.jsonl`: online cache (appended sample-by-sample)
- `result_*.jsonl`: final merged output
- If the process is interrupted, rerunning the same command resumes from cache and skips finished samples.
- Useful options:
- `--retry-error`: retry only failed/error cached samples
- `--new-cache`: recreate cache from scratch

## 5. Reproducibility Note

Across different runs, final accuracy may fluctuate by a few tenths of a percentage point (typically around `0.x%`) due to inference/runtime variability.

## Citation

```bibtex
@article{placeholder2026,
  title={Paper Title Placeholder},
  author={Author Placeholder},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```
