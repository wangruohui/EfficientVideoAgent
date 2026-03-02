import argparse
import asyncio
import json
import os
import re
import shlex
import sys
from collections import Counter
from copy import copy
from typing import Any, Dict, List, Optional, Tuple

import openai
from openai import AsyncOpenAI, OpenAI
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

BASE_URL = "http://localhost:8000/v1"
API_KEY = "no"

MODEL_TOKENIZER_PATH = "/mnt/afs/wangruohui/hf_models/Qwen2.5-VL-7B-Instruct/"
FRAME_TOOL_PATH = "select_frame_fallback.py"
FRAME_SAVE_ROOT = "/mnt/afs2/wangruohui/extracted_frames/"

DATASET_CONFIG: Dict[str, Dict[str, str]] = {
    "videomme": {
        "jsonl": "data/videomme_test_wosubtitles_raw_list_full.jsonl",
        "video_root": "/mnt/afs/share_data/opencompass/.cache/VideoMME/video/",
        "cache": "cache_videomme.jsonl",
        "result": "result_videomme.jsonl",
    },
    "lsdbench": {
        "jsonl": "data/LSDBench_raw_full.jsonl",
        "video_root": "/mnt/afs/share_data/LMUData/LSDBench",
        "cache": "cache_lsdbench.jsonl",
        "result": "result_lsdbench.jsonl",
    },
    "lvbench": {
        "jsonl": "data/LVBench_raw_full_root.jsonl",
        "video_root": "/mnt/afs/share_data/LVBench/video/all_videos/",
        "cache": "cache_lvbench.jsonl",
        "result": "result_lvbench.jsonl",
    },
    "videoholmes": {
        "jsonl": "data/videoholmes_raw_full.jsonl",
        "video_root": "/mnt/afs/share_data/LMUData/holmes/videos/",
        "cache": "cache_videoholmes.jsonl",
        "result": "result_videoholmes.jsonl",
    },
    "longvideobench": {
        "jsonl": "data/LongVideoBench_nosub_raw_full.jsonl",
        "video_root": "/mnt/afs/share_data/opencompass/.cache/longvideobench/videos",
        "cache": "cache_longvideobench.jsonl",
        "result": "result_longvideobench.jsonl",
    },
    "mlvu": {
        "jsonl": "data/MLVU_MCQ_raw_full.jsonl",
        "video_root": "/mnt/afs/share_data/opencompass/.cache/MVLU/MLVU/",
        "cache": "cache_mlvu_mcq.jsonl",
        "result": "result_mlvu_mcq.jsonl",
    },
}

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
aclient = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)


def get_model() -> str:
    models = client.models.list()
    return models.data[0].id


def extract_answer(text: str) -> str:
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_answer_v2(text: str) -> str:
    pattern = r"[Aa]nswer[\.:\s]\s*(\w)(\W.{,100}|)$"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_answer_anyway(text: str) -> str:
    ans = extract_answer(text)
    if ans == "":
        ans = extract_answer_v2(text)
    return ans


async def call_frame_select(
    video_path: str,
    arguments: Dict[str, Any],
    fallback: bool = True,
    tool_version: str = "v3",
    tool_path: str = FRAME_TOOL_PATH,
) -> Tuple[Optional[List[str]], Optional[List[int]]]:
    python = sys.executable

    if fallback:
        arguments = {
            k: v
            for k, v in arguments.items()
            if k in {"start_time", "end_time", "resize", "nframes"}
        }

    arguments_str = ["--video-path", str(video_path)]
    arguments_str.extend(sum([[f"--{k}", str(v)] for k, v in arguments.items()], []))
    arguments_str.extend(["--save-root", FRAME_SAVE_ROOT])
    arguments_str.extend(["--factor", "28"])

    if fallback:
        arguments_str.extend(["--clamp"])

    cmd = " ".join([shlex.quote(python), shlex.quote(tool_path)] + [shlex.quote(x) for x in arguments_str])

    proc = await asyncio.create_subprocess_exec(
        python,
        tool_path,
        *arguments_str,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await proc.communicate()
    stdout_text = stdout.decode()

    m = re.search(
        r"from (.*?)/frame_(\d+)\.([pngje]*) to (.*?)/frame_(\d+)\.([pngje]*)",
        stdout_text,
    )

    if m is None:
        print("In tool call error: regex not matched")
        print(f"tool cmd: {cmd}")
        print(f"{stderr.decode()}, using {arguments_str=}")
        return None, None

    if stderr.strip():
        print(f"tool cmd: {cmd}")
        print(stderr.decode())

    path = m.group(1)
    start = int(m.group(2))
    ext = m.group(3)
    end = int(m.group(5))
    img_paths = [f"{path}/frame_{i:04d}.{ext}" for i in range(start, end + 1)]
    time_step = (arguments["end_time"] - arguments["start_time"]) / max(1, end - start)
    time_stamps = [
        int(arguments["start_time"] + i * time_step) for i in range(end - start + 1)
    ]

    return img_paths, time_stamps


async def ffprobe_video_stream_meta(video_path: str) -> Tuple[float, str]:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration,width,height",
        "-of",
        "json",
        video_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {stderr.decode().strip()}")

    data = json.loads(stdout.decode() or "{}")
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("ffprobe returned no video streams")

    stream = streams[0]
    duration = float(stream.get("duration"))
    width = int(stream.get("width"))
    height = int(stream.get("height"))

    resolution = f"{width}x{height}"
    return duration, resolution


def build_message_from_rl(prompt: List[Dict[str, Any]], video_length: float, resolution: str):
    chat: List[Dict[str, Any]] = []

    if prompt[0]["role"] != "system":
        chat.append(
            {
                "role": "system",
                "content": "Use Frame Select Tool to Anlysis the video and generate an answer to the question.",
            }
        )

    for p in prompt:
        chat.append(p)

    user_index = -1
    for i, turn in enumerate(chat):
        if turn["role"] == "user":
            user_index = i
            break
    if user_index < 0:
        raise ValueError("No user turn found in prompt")

    raw_question = chat[user_index]["content"].strip()
    width, height = resolution.split("x")
    pnumber = min(int(width), int(height))

    chat[user_index]["content"] = (
        f"Video Length: {int(max(video_length - 0.5, 1))} seconds. "
        f"Original video resolution: {pnumber}p. {raw_question}"
    )

    return chat


async def single(
    index: int,
    item: Dict[str, Any],
    model: str,
    dataset_cfg: Dict[str, str],
    tokenizer,
    max_turns: int = 8,
    timestamp_fmt: str = "mmss",
    max_visual_tokens: int = 60000,
    maxp: int = 720,
    fallback: bool = True,
) -> Dict[str, Any]:
    prompt = item["prompt"]
    video = os.path.join(dataset_cfg["video_root"], item["videos"][0])
    raw_query = ""
    for turn in prompt:
        if turn.get("role") == "user":
            raw_query = str(turn.get("content", "")).strip()
            break

    video_length, resolution = await ffprobe_video_stream_meta(video)
    width, height = (int(v) for v in resolution.split("x"))
    video_p = min(width, height)

    def estimated_tokens(nframes: int, resize: float) -> float:
        return nframes * height * width * resize * resize / 28 / 28

    messages = build_message_from_rl(prompt, video_length, resolution)
    messages = copy(messages)

    stop_reason = "max_turns_exceeded"
    answer: Optional[str] = None

    for _ in range(max_turns):
        try:
            response = await aclient.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=2048,
                timeout=9999,
            )
        except openai.BadRequestError as e:
            print(f"OpenAI BadRequestError: {str(e)}, retrying...")
            stop_reason = "openai_bad_request"
            return {
                "index": index,
                "videos": item.get("videos", []),
                "messages": tokenizer.apply_chat_template(messages, tokenize=False).replace("<|image_pad|>", ""),
                "gt": item.get("reward_model", {}).get("ground_truth"),
                "answer": "",
                "stop_reason": stop_reason,
                "error": f"OpenAI BadRequestError after retries: {str(e)}",
            }

        resp = response.choices[0].message.content
        messages.append({"role": "assistant", "content": resp})

        answer = extract_answer_anyway(resp)
        if answer != "":
            stop_reason = "answer_found"
            break

        tool_calls_match = re.search(r"<tool_call>(.*?)</tool_call>", resp, re.DOTALL)
        if not tool_calls_match:
            stop_reason = "no_answer_no_tool_call"
            break
        tool_calls = tool_calls_match.group(1).strip()

        arguments = re.findall(
            r'{\s*"tool":\s*"frame_select",\s*"(?:arguments|parameters)":\s*({.*?})\s*}',
            tool_calls,
            re.DOTALL,
        )
        if not arguments:
            stop_reason = "no_valid_tool_call"
            break

        img_paths: List[str] = []
        time_stamps: List[int] = []
        errors: List[str] = []

        for arg in arguments:
            try:
                parsed = json.loads(arg.strip())
            except json.JSONDecodeError:
                errors.append("arguments_not_json")
                continue

            if ("start_time" not in parsed) or ("end_time" not in parsed):
                errors.append("no_start_or_end_time")
                continue

            if "nframes" not in parsed:
                errors.append("no_nframes")
                continue

            nframes = parsed["nframes"]
            resize = parsed.get("resize", 1.0)
            if resize is None:
                resize = 1.0

            current_estimated_tokens = estimated_tokens(nframes, resize)
            if current_estimated_tokens > max_visual_tokens:
                # First-stage fallback: clamp video resolution to maxp.
                resize_cap_from_maxp = min(1.0, maxp / video_p)
                resize = min(resize, resize_cap_from_maxp)
                current_estimated_tokens = estimated_tokens(nframes, resize)

                if current_estimated_tokens > max_visual_tokens:
                    r = (max_visual_tokens / current_estimated_tokens) ** (1/3)
                    nframes = max(1, int(nframes * r))
                    resize = resize * r

                # # Second-stage fallback: iterative joint downscale.
                # nframes = max(1, int(nframes / 1.2))
                # resize = resize / 1.2
                current_estimated_tokens = estimated_tokens(nframes, resize)
                print(f"turn {_}, reduce to {current_estimated_tokens=} at nframes={nframes}, resize={resize:.4f}")

            parsed["nframes"] = nframes
            parsed["resize"] = resize

            img_paths_single, time_stamps_single = await call_frame_select(
                video,
                parsed,
                fallback=fallback,
            )

            if not img_paths_single:
                errors.append("call_frame_select_failed")
                continue

            img_paths.extend(img_paths_single)
            time_stamps.extend(time_stamps_single)

        if errors and len(errors) == len(arguments):
            stop_reason = "; ".join(errors)
            break

        content: List[Dict[str, Any]] = [{"type": "text", "text": "<tool_response>"}]
        for i, (frame_path, sec) in enumerate(zip(img_paths, time_stamps)):
            if timestamp_fmt == "mmss":
                minutes = sec // 60
                seconds = sec % 60
                ts = f"{minutes:02d}:{seconds:02d}"
                first_nextline = "\n" if i > 0 else ""
                content.append({"type": "text", "text": f"{first_nextline}Frame {i} at [{ts}]: "})
            elif timestamp_fmt == "seconds":
                content.append({"type": "text", "text": f"\nFrame {i} at {sec} seconds:"})
            elif timestamp_fmt in ["", None]:
                pass
            else:
                raise ValueError(f"Unknown timestamp_fmt: {timestamp_fmt}")

            img = Image.open(frame_path)
            img.load()

            content.append({"type": "image_url", "image_url": {"url": f"file://{frame_path}"}})

        content.append(
            {
                "type": "text",
                "text": (
                    "\nIf more information is needed, call the frame selection tool again.\n"
                    f"Question: {raw_query}</tool_response>"
                ),
            }
        )
        messages.append({"role": "tool", "content": content})

    return {
        "index": index,
        "videos": item.get("videos", []),
        "messages": tokenizer.apply_chat_template(messages, tokenize=False).replace("<|image_pad|>", ""),
        "gt": item.get("reward_model", {}).get("ground_truth"),
        "answer": answer if answer is not None else "",
        "stop_reason": stop_reason,
    }


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f):
            item = json.loads(line)
            if not isinstance(item.get("index"), int):
                item["index"] = lineno
            items.append(item)
    return items


def load_cache(cache_path: str) -> Dict[int, Dict[str, Any]]:
    records: Dict[int, Dict[str, Any]] = {}
    if not os.path.exists(cache_path):
        return records

    with open(cache_path, "r") as f:
        lines = f.readlines()

    for line in reversed(lines):
        data = json.loads(line)
        index = data["index"]
        if index in records:
            continue
        records[index] = data

    return records


def append_cache(cache_path: str, record: Dict[str, Any]) -> None:
    with open(cache_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def rewrite_cache(cache_path: str, records: Dict[int, Dict[str, Any]]) -> None:
    with open(cache_path, "w") as f:
        for index in sorted(records.keys()):
            f.write(json.dumps(records[index], ensure_ascii=False) + "\n")


def is_normal_stop_reason(stop_reason: Optional[str]) -> bool:
    return stop_reason in {"answer_found", "max_turns_exceeded", "no_answer_no_tool_call", "no_valid_tool_call"}


def should_retry_cached_record(record: Dict[str, Any]) -> bool:
    if record.get("error"):
        return True
    return not is_normal_stop_reason(record.get("stop_reason"))


def write_result_from_cache(cache_path: str, result_path: str) -> List[Dict[str, Any]]:
    all_records = load_cache(cache_path)
    sorted_items = [all_records[idx] for idx in sorted(all_records.keys())]
    with open(result_path, "w") as f:
        for item in sorted_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return sorted_items


async def process_single_item(
    index: int,
    item: Dict[str, Any],
    model_name: str,
    dataset_cfg: Dict[str, str],
    tokenizer,
    semaphore: asyncio.Semaphore,
    max_visual_tokens: int,
    maxp: int,
    fallback: bool,
) -> Tuple[int, Dict[str, Any]]:
    async with semaphore:
        try:
            record = await single(
                index,
                item,
                model_name,
                dataset_cfg,
                tokenizer,
                max_visual_tokens=max_visual_tokens,
                maxp=maxp,
                fallback=fallback,
            )
            return index, record
        except Exception as e:
            return index, {
                "index": index,
                "videos": item.get("videos", []),
                "messages": [],
                "gt": item.get("reward_model", {}).get("ground_truth"),
                "answer": "",
                "stop_reason": "exception",
                "error": f"{type(e).__name__}: {e}",
            }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIG.keys()))
    parser.add_argument("--max-concurrent", type=int, default=20)
    parser.add_argument("--new-cache", action="store_true", help="Recreate cache file")
    parser.add_argument(
        "--retry-error",
        action="store_true",
        help="Retry cached failed records; default is to skip all cached records",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Use first N samples only (default: all samples)",
    )
    parser.add_argument(
        "-v",
        "--max-visual-tokens",
        type=int,
        default=60000,
        help="Max visual token budget per single tool call before frame extraction",
    )
    parser.add_argument(
        "--maxp",
        type=int,
        default=720,
        help="Max effective video p used for first-stage resize fallback when tokens exceed budget",
    )
    parser.add_argument(
        "--fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fallback logic (arg filtering + clamp + robust handling)",
    )
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    cache_path = cfg["cache"]
    result_path = cfg["result"]

    if args.new_cache and os.path.exists(cache_path):
        print(f"WARNING: --new-cache will delete: {cache_path}")
        confirm = input("Confirm delete cache? (y/N): ").strip().lower()
        if confirm != "y":
            print("cancelled, cache not deleted")
            return
        os.remove(cache_path)
        print(f"deleted cache: {cache_path}")

    model_name = get_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKENIZER_PATH)

    items = load_jsonl(cfg["jsonl"])
    if args.sample is not None:
        if args.sample <= 0:
            raise ValueError(f"--sample must be positive, got {args.sample}")
        items = items[: args.sample]
    cache_records = load_cache(cache_path)
    cached_indices = set(cache_records.keys())
    done_indices = {
        index
        for index, record in cache_records.items()
        if not should_retry_cached_record(record)
    }
    cached_error_indices = cached_indices - done_indices
    if args.retry_error:
        retry_cached_indices = set(cached_error_indices)
        if cached_error_indices:
            print(f"found {len(cached_error_indices)} cached error records")
            confirm = input("Delete cached error records before retry? (y/N): ").strip().lower()
            if confirm != "y":
                print("cancelled, cache unchanged")
                return
            kept_records = {
                index: record
                for index, record in cache_records.items()
                if index not in cached_error_indices
            }
            rewrite_cache(cache_path, kept_records)
            cache_records = kept_records
            cached_indices = set(cache_records.keys())
            done_indices = set(cache_records.keys())
            cached_error_indices = cached_indices - done_indices
        processed_indices = done_indices
    else:
        processed_indices = cached_indices
        retry_cached_indices = set()

    print(f"dataset: {args.dataset}")
    print(f"total items: {len(items)}")
    if args.sample is not None:
        print(f"sample mode: using first {args.sample} samples")
    print(f"cache file: {cache_path}")
    print(f"already cached: {len(cache_records)}")
    print(f"skip as done: {len(processed_indices)}")
    print(f"cached errors: {len(cached_error_indices)}")
    print(f"retry from cache errors: {len(retry_cached_indices)}")

    semaphore = asyncio.Semaphore(args.max_concurrent)

    tasks = []
    for item in items:
        index = item["index"]
        if index in processed_indices:
            continue
        tasks.append(
            process_single_item(
                index,
                item,
                model_name,
                cfg,
                tokenizer,
                semaphore,
                args.max_visual_tokens,
                args.maxp,
                args.fallback,
            )
        )

    error_indices: List[int] = []
    pbar = tqdm(total=len(tasks), desc="Processing", unit="item", ncols=100)
    try:
        for coro in asyncio.as_completed(tasks):
            index, record = await coro
            append_cache(cache_path, record)

            if should_retry_cached_record(record):
                error_indices.append(index)

            pbar.update(1)
    finally:
        pbar.close()

    final_records = write_result_from_cache(cache_path, result_path)
    print(f"done, result saved to {result_path}, total={len(final_records)}")

    records_by_index: Dict[int, Dict[str, Any]] = {}
    for rec in final_records:
        idx = rec.get("index")
        if isinstance(idx, int):
            records_by_index[idx] = rec

    eval_total = len(items)
    eval_correct = 0
    fail_not_answer_found = 0
    missing_result = 0
    for item in items:
        idx = item["index"]
        rec = records_by_index.get(idx)
        if rec is None:
            missing_result += 1
            continue

        if rec.get("stop_reason") != "answer_found":
            fail_not_answer_found += 1
            continue

        gt_s = str(item.get("reward_model", {}).get("ground_truth", "")).strip()
        ans_s = str(rec.get("answer", "")).strip()
        if gt_s == ans_s:
            eval_correct += 1

    if eval_total > 0:
        acc = eval_correct / eval_total
        print(f"accuracy: {eval_correct}/{eval_total} = {acc:.4%}")
        print(f"failed (stop_reason != answer_found): {fail_not_answer_found}")
        print(f"missing result in cache: {missing_result}")
    else:
        print("accuracy: N/A (dataset is empty)")

    reason_counter = Counter()
    for rec in final_records:
        reason = rec.get("stop_reason", "missing_stop_reason")
        reason_counter[str(reason)] += 1
    print("stop_reason stats:")
    for reason, cnt in reason_counter.most_common():
        print(f"  {reason}: {cnt}")

    if error_indices:
        error_indices = sorted(set(error_indices))
        print(f"error indices ({len(error_indices)}): {error_indices}")
    else:
        print("no exception indices")


if __name__ == "__main__":
    asyncio.run(main())
