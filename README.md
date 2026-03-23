# EVA: Efficient Reinforcement Learning for End-to-End Video Agent

[![Paper](https://img.shields.io/badge/Paper-Link-b31b1b.svg)](Paper_Link_Placeholder)
[![Model](https://img.shields.io/badge/Model-Link-blue.svg)](https://huggingface.co/WRHC/EfficientVideoAgent/)

This repository contains the official evaluation code for the model proposed in our paper.

## 1. Paper and Model

- Paper Title: `EVA: Efficient Reinforcement Learning for End-to-End Video Agent`
- Paper Link: `Paper_Link_Placeholder`
- Model Link: `https://huggingface.co/WRHC/EfficientVideoAgent/`

## 2. Reference Results

Reference result files are provided in this repository, under `results-12k`.
You can compute accuracy with `accuracy.py`:

```bash
python accuracy.py <result_jsonl_path>
```

You can control per-tool-call visual token budget via `-v/--max-visual-tokens`.
When a tool call exceeds this budget, `eval-eva.py` automatically reduces resolution and frame count before extraction.

Main results:

| Dataset | Acc | Round | Token |
| --- | ---: | ---: | ---: |
| VideoMME | 60.15 | 2.42 | 16911 |
| LongVideoBench | 54.97 | 2.57 | 19042 |
| MLVU | 68.26 | 2.42 | 16570 |
| LSDBench | 49.31 | 2.48 | 13914 |
| VideoHolmes | 37.18 | 2.75 | 9085 |
| LVBench | 43.32 | 2.62 | 20412 |

`Token` includes both text tokens and image tokens.

## 3. Run Your Own Evaluation

### Step 1. Clone the Repository

```bash
git clone https://github.com/wangruohui/EfficientVideoAgent.git
cd EfficientVideoAgent
```

### Step 2. Download Model and Install Dependencies

1. Download model weights: `https://huggingface.co/WRHC/EfficientVideoAgent/`
2. Install FFmpeg following `https://www.ffmpeg.org/download.html`, ensure `ffprobe` is in `PATH`, and ensure FFmpeg shared libraries are in `LD_LIBRARY_PATH`.
3. Install dependencies from `requirements.txt` (recommended: `uv`)

   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

### Step 3. Download Evaluation Datasets and Update Dataset Paths

`eval-eva.py` reads dataset meta from `DATASET_CONFIG`. Before running evaluation, make sure each dataset is available locally and paths are correct.

1. Download and extract video datasets (VideoMME / LSDBench / LVBench / VideoHolmes / LongVideoBench / MLVU).
2. Annotation jsonl files are already provided in `data/*.jsonl` and have been normalized to a unified format.
3. Edit `eval-eva.py` -> `DATASET_CONFIG`: only `video_root` needs to be changed to your local video directory.

Example:

```python
DATASET_CONFIG = {
    "videomme": {
        "jsonl": "data/videomme_test_wosubtitles_raw_list_full.jsonl",
        "video_root": "/path/to/VideoMME/video",
        "cache": "cache_videomme.jsonl",
        "result": "result_videomme.jsonl",
    },
}
```

### Step 4. Serve the Model with vLLM (Multi-GPU Data Parallel)


```bash
vllm serve <MODEL_PATH_OR_HF_ID> \
  --data-parallel-size <NUM_GPUS> \
  --limit-mm-per-prompt '{"image": 9999, "video":0}' \
  --mm_processor_cache_gb 20 \
  --attention-backend FLASH_ATTN \
  --allowed-local-media-path <LOCAL_MEDIA_ROOT>
```

**Reproducibility Note**

With vLLM, even when `temperature=0`, final accuracy can still fluctuate by around `0.x%` across runs.

### Step 5. Configure `eval-eva.py` Runtime Settings and Run Evaluation

Before running, edit the config section at the top of `eval-eva.py`:

- `BASE_URL`: OpenAI-compatible endpoint for your vLLM server (for example, `http://localhost:8000/v1`).
- `API_KEY`: API key used by the client (can be a dummy value for local vLLM setups if authentication is disabled).
- `MODEL_TOKENIZER_PATH`: Tokenizer path, should pointing to downloaded hf model weights, i.e. `https://huggingface.co/WRHC/EfficientVideoAgent/` in step 2.
- `FRAME_TOOL_PATH`: path to the frame selection tool script (default is `select_frame_fallback.py`).
- `FRAME_SAVE_ROOT`: directory where extracted frames are saved during tool calls.
Also make sure:
   - `FRAME_SAVE_ROOT` directory exists and is writable (or set it to a writable path).
   - vLLM `--allowed-local-media-path` covers your dataset `video_root` directories.
- `DATASET_CONFIG`: per-dataset I/O configuration.
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

Set visual token budget per tool call:

```bash
python eval-eva.py --dataset videomme -v 12000
python eval-eva.py --dataset videomme -v 32000
```

If one tool call exceeds `-v`, the script will automatically downscale `resize` and `nframes` to fit the budget.

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
   - `--output-dir`: redirect cache/result outputs to another directory


## Citation

```bibtex
@article{placeholder2026,
  title={EVA: Efficient Reinforcement Learning for End-to-End Video Agent},
  author={Author Placeholder},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```
