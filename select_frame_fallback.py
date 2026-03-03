import argparse
import datetime
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _parse_time_to_seconds(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    if ":" not in value:
        return float(value)
    parts = value.split(":")
    total = 0.0
    for i, part in enumerate(reversed(parts)):
        total += float(part) * (60**i)
    return total


class DecordBackend:
    name = "decord"

    def __init__(self, video_path: str):
        from decord import VideoReader, cpu

        self._vr = VideoReader(video_path, ctx=cpu(0))
        self._fps = float(self._vr.get_avg_fps())
        if self._fps <= 0:
            self._fps = 1.0

    def get_stream_bounds(self) -> Tuple[float, float]:
        total = len(self._vr)
        if total <= 0:
            raise ValueError("Empty video stream")
        first_pts = float(self._vr.get_frame_timestamp(0)[0])
        last_pts = float(self._vr.get_frame_timestamp(total - 1)[0])
        end_pts = max(last_pts + 1.0 / self._fps, last_pts + 1e-3)
        return first_pts, end_pts

    def get_frames_by_timestamps(self, timestamps: List[float]) -> np.ndarray:
        if not timestamps:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)
        stream_start, _ = self.get_stream_bounds()
        total = len(self._vr)
        indices = [
            max(0, min(int(round((ts - stream_start) * self._fps)), total - 1))
            for ts in timestamps
        ]
        return self._vr.get_batch(indices).asnumpy().astype(np.uint8)


class TorchcodecBackend:
    name = "torchcodec"

    def __init__(self, video_path: str):
        from torchcodec.decoders import VideoDecoder

        self._decoder = VideoDecoder(video_path)
        self._stream = self._pick_video_stream(self._get_metadata())

    def _get_metadata(self):
        metadata = getattr(self._decoder, "metadata", None)
        if callable(metadata):
            return metadata()
        return metadata

    @staticmethod
    def _get_first_attr(obj, names, default=None):
        for name in names:
            if hasattr(obj, name):
                value = getattr(obj, name)
                if value is not None:
                    return value
        return default

    @staticmethod
    def _pick_video_stream(container_metadata):
        if container_metadata is None:
            return None
        streams = getattr(container_metadata, "streams", None)
        if streams:
            for stream in streams:
                cls_name = stream.__class__.__name__.lower()
                if "video" in cls_name:
                    return stream
            return streams[0]
        return container_metadata

    def get_stream_bounds(self) -> Tuple[float, float]:
        stream = self._stream
        if stream is None:
            return 0.0, 1e10

        begin = float(
            self._get_first_attr(
                stream,
                [
                    "begin_stream_seconds",
                    "begin_stream_seconds_from_header",
                    "begin_seconds",
                ],
                0.0,
            )
        )
        end = self._get_first_attr(
            stream,
            ["end_stream_seconds", "end_stream_seconds_from_header"],
            None,
        )
        if end is None:
            duration = self._get_first_attr(
                stream,
                ["duration_seconds_from_header", "duration_seconds", "duration"],
                None,
            )
            if duration is not None:
                end = float(duration)
        if end is None:
            end = begin + 1e10
        else:
            end = float(end)

        if end <= begin:
            end = begin + 1e-3
        return begin, end

    def get_frames_by_timestamps(self, timestamps: List[float]) -> np.ndarray:
        if not timestamps:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)
        frame_batch = self._decoder.get_frames_played_at(timestamps)
        frames_tensor = frame_batch.data
        if frames_tensor.numel() == 0:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)
        frames_tensor = frames_tensor.permute(0, 2, 3, 1)
        return frames_tensor.numpy().astype(np.uint8)


def _build_timestamps(start_time: float, end_time: float, nframes: int) -> List[float]:
    if nframes <= 0:
        raise ValueError("nframes must be a positive integer")
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")
    if nframes == 1:
        return [(start_time + end_time) / 2.0]
    epsilon = min(1e-3, (end_time - start_time) / (nframes * 4.0))
    end_exclusive = max(start_time, end_time - epsilon)
    return np.linspace(start_time, end_exclusive, nframes).tolist()


def _resize_with_opencv(
    frames: np.ndarray, resize: Optional[float], factor: int = 1
) -> np.ndarray:
    if resize is None:
        return frames
    if resize <= 0:
        raise ValueError("resize must be > 0")
    if factor is not None and factor <= 0:
        raise ValueError("factor must be > 0 when provided")
    if frames.size == 0:
        return frames
    h, w = frames.shape[1:3]
    out_w = max(1, int(round(w * resize / factor)) * factor)
    out_h = max(1, int(round(h * resize / factor)) * factor)
    return np.array(
        [cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_CUBIC) for frame in frames],
        dtype=np.uint8,
    )


def _apply_time_bounds(
    start_time: float,
    end_time: Optional[float],
    stream_begin: float,
    stream_end: float,
    clamp_to_stream: bool,
) -> Tuple[float, float]:
    real_end = stream_end if end_time is None else end_time
    if clamp_to_stream:
        start_time = max(start_time, stream_begin)
        real_end = min(real_end, stream_end)
    if real_end <= start_time:
        if clamp_to_stream:
            start_time = stream_begin
            real_end = stream_end
        else:
            raise ValueError(
                f"Invalid time range: [{start_time}, {real_end}] vs stream [{stream_begin}, {stream_end}]"
            )
    return start_time, real_end


BACKENDS = {
    "decord": DecordBackend,
    "torchcodec": TorchcodecBackend,
}


def extract_frames(
    video_path: str,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    nframes: int = 32,
    resize: Optional[float] = None,
    factor: int = 1,
    backend: str = "auto",
    clamp_to_stream: bool = False,
) -> Tuple[np.ndarray, List[float], str]:
    if backend == "auto":
        backend_order = ["decord", "torchcodec"]
    elif backend in BACKENDS:
        backend_order = [backend]
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    errors: Dict[str, str] = {}
    for backend_name in backend_order:
        try:
            decoder = BACKENDS[backend_name](video_path)
            stream_begin, stream_end = decoder.get_stream_bounds()
            effective_start, effective_end = _apply_time_bounds(
                start_time=start_time,
                end_time=end_time,
                stream_begin=stream_begin,
                stream_end=stream_end,
                clamp_to_stream=clamp_to_stream,
            )
            timestamps = _build_timestamps(effective_start, effective_end, nframes)
            frames = decoder.get_frames_by_timestamps(timestamps)
            frames = _resize_with_opencv(frames, resize, factor)
            return frames, timestamps, backend_name
        except Exception as exc:
            errors[backend_name] = str(exc)

    formatted_errors = "\n".join(
        [f"  - backend={name}\n    reason={reason}" for name, reason in errors.items()]
    )
    raise RuntimeError(f"All backends failed:\n{formatted_errors}")


def _save_frames(
    frames: np.ndarray, video_path: str, save_root: str, save_dir: Optional[str]
) -> Tuple[Path, int, List[str]]:
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    time_str = datetime.datetime.now().strftime("%H%M%S")
    video_name = Path(video_path).stem
    rand = str(uuid.uuid4())[:8]
    if save_dir:
        out_dir = Path(save_dir)
    else:
        out_dir = Path(save_root) / date_str / video_name / f"{time_str}_{rand}"
    out_dir.mkdir(parents=True, exist_ok=True)

    names: List[str] = []
    saved_count = 0
    for i, frame in enumerate(frames):
        frame_path = out_dir / f"frame_{i:04d}.png"
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(frame_path), frame_bgr)
        names.append(str(frame_path))
        saved_count += 1
    return out_dir, saved_count, names


def main():
    parser = argparse.ArgumentParser(
        description="Single-file frame extraction tool with decord -> torchcodec fallback."
    )
    parser.add_argument("-i", "--video-path", "--video_path", required=True, help="Path to video file")
    parser.add_argument(
        "--start",
        "--start-time",
        "--start_time",
        type=str,
        default="0",
        help="Start time in seconds or [[HH:]MM:]SS[.mmm]",
    )
    parser.add_argument(
        "--end",
        "--end-time",
        "--end_time",
        type=str,
        default=None,
        help="End time in seconds or [[HH:]MM:]SS[.mmm], default: stream end",
    )
    parser.add_argument("--nframes", type=int, default=32, help="Number of frames to sample")
    parser.add_argument("--resize", type=float, default=None, help="Resize ratio, e.g. 0.5")
    parser.add_argument(
        "--factor",
        type=int,
        default=1,
        help="Round resized width/height to multiples of this value. Default: 1.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "decord", "torchcodec"],
        default="auto",
        help="Backend choice. auto means decord first, then torchcodec on failure.",
    )
    parser.add_argument(
        "--clamp",
        action="store_true",
        default=False,
        help="Clamp start/end to the video stream time range when requested times exceed stream bounds.",
    )
    parser.add_argument("--save-root", default="/tmp", help="Root directory for saving frames")
    parser.add_argument("--save-dir", default=None, help="Explicit output directory")
    args = parser.parse_args()

    start_time = _parse_time_to_seconds(args.start)
    end_time = _parse_time_to_seconds(args.end)
    assert start_time is not None

    frames, timestamps, used_backend = extract_frames(
        video_path=args.video_path,
        start_time=start_time,
        end_time=end_time,
        nframes=args.nframes,
        resize=args.resize,
        factor=args.factor,
        backend=args.backend,
        clamp_to_stream=args.clamp,
    )

    frame_dir, saved_count, names = _save_frames(
        frames=frames,
        video_path=args.video_path,
        save_root=args.save_root,
        save_dir=args.save_dir,
    )

    print(
        f"Frames saved to {frame_dir}, total frames saved: {saved_count}, from {names[0]} to {names[-1]}"
    )


if __name__ == "__main__":
    main()
