#!/usr/bin/env python3
"""Generate and score the MiniMax English target-speaker baseline.

This reproduces the smallest paper-comparable Qwen3-TTS evaluation:
Qwen3-TTS-12Hz-0.6B-CustomVoice, Aiden, 100 English MiniMax prompts,
and Whisper-large-v3 WER.
"""

import argparse
import gc
import json
import os
import string
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Qwen3-TTS only needs the PyTorch Transformers backend. Disabling TensorFlow
# also avoids the oneDNN shutdown crash seen in mixed server environments.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import soundfile as sf
import torch
from jiwer import process_words
from scipy.signal import resample_poly
from zhon.hanzi import punctuation as chinese_punctuation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEXT_FILE = REPO_ROOT / "eval_data" / "minimax" / "text" / "english.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval_outputs" / "minimax_english_aiden_12hz_0.6b"
TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
ASR_MODEL_ID = "openai/whisper-large-v3"
PAPER_WER_PERCENT = 1.188


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "generate", "score"),
        default="all",
        help="Run generation, scoring, or both sequentially (default: all).",
    )
    parser.add_argument("--text-file", type=Path, default=DEFAULT_TEXT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speaker", default="Aiden")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N prompts for a smoke test. Omit for all 100.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate WAVs and recompute existing ASR rows.",
    )
    return parser.parse_args()


def load_prompts(text_file: Path, limit: Optional[int]) -> List[Dict[str, object]]:
    if not text_file.is_file():
        raise FileNotFoundError(
            f"MiniMax text file not found: {text_file}\n"
            "Download it to eval_data/minimax/text/english.txt first."
        )

    prompts: List[Dict[str, object]] = []
    for raw_line in text_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "|" in line:
            source_voice, text = line.split("|", 1)
        else:
            source_voice, text = "", line
        prompts.append(
            {
                "index": len(prompts),
                "source_voice": source_voice.strip(),
                "text": text.strip(),
            }
        )

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        prompts = prompts[:limit]

    if not prompts:
        raise ValueError(f"No prompts found in {text_file}")
    return prompts


def audio_path(output_dir: Path, index: int) -> Path:
    return output_dir / "audio" / f"{index:04d}.wav"


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def generate_audio(
    prompts: List[Dict[str, object]],
    output_dir: Path,
    device: str,
    speaker: str,
    overwrite: bool,
) -> None:
    from qwen_tts import Qwen3TTSModel

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        prompt
        for prompt in prompts
        if overwrite or not audio_path(output_dir, int(prompt["index"])).is_file()
    ]

    if pending:
        print(f"Loading TTS model: {TTS_MODEL_ID}", flush=True)
        tts = Qwen3TTSModel.from_pretrained(
            TTS_MODEL_ID,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

        total = len(prompts)
        with torch.inference_mode():
            for prompt in prompts:
                index = int(prompt["index"])
                destination = audio_path(output_dir, index)
                if destination.is_file() and not overwrite:
                    print(f"[{index + 1:03d}/{total:03d}] exists: {destination}", flush=True)
                    continue

                started = time.perf_counter()
                wavs, sample_rate = tts.generate_custom_voice(
                    text=str(prompt["text"]),
                    language="English",
                    speaker=speaker,
                    instruct="",
                    max_new_tokens=2048,
                )
                elapsed = time.perf_counter() - started
                waveform = np.asarray(wavs[0], dtype=np.float32)
                sf.write(destination, waveform, sample_rate, subtype="PCM_16")
                print(
                    f"[{index + 1:03d}/{total:03d}] {elapsed:.2f}s -> {destination}",
                    flush=True,
                )

        del tts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("All requested WAV files already exist; generation skipped.", flush=True)

    manifest = []
    for prompt in prompts:
        index = int(prompt["index"])
        destination = audio_path(output_dir, index)
        if not destination.is_file():
            raise RuntimeError(f"Expected generated file is missing: {destination}")
        info = sf.info(destination)
        manifest.append(
            {
                **prompt,
                "speaker": speaker,
                "audio": str(destination.resolve()),
                "sample_rate": info.samplerate,
                "duration_seconds": info.duration,
            }
        )
    write_jsonl(output_dir / "manifest.jsonl", manifest)


def load_existing_asr(path: Path) -> Dict[int, Dict[str, object]]:
    rows: Dict[int, Dict[str, object]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["index"])] = row
    return rows


def load_audio_16khz(path: Path) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sample_rate != 16000:
        divisor = int(np.gcd(sample_rate, 16000))
        waveform = resample_poly(waveform, 16000 // divisor, sample_rate // divisor)
    return np.asarray(waveform, dtype=np.float32)


def normalize_english(text: str) -> str:
    # Match the public Seed-TTS English WER script: remove punctuation except
    # apostrophes, lowercase, and split on whitespace.
    for character in chinese_punctuation + string.punctuation:
        if character != "'":
            text = text.replace(character, "")
    return " ".join(text.lower().split())


def score_audio(
    prompts: List[Dict[str, object]],
    output_dir: Path,
    device: str,
    overwrite: bool,
) -> None:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    for prompt in prompts:
        destination = audio_path(output_dir, int(prompt["index"]))
        if not destination.is_file():
            raise FileNotFoundError(
                f"Missing {destination}. Run with --stage generate or --stage all first."
            )

    results_path = output_dir / "asr_results.jsonl"
    existing = {} if overwrite else load_existing_asr(results_path)
    pending = [prompt for prompt in prompts if int(prompt["index"]) not in existing]

    if pending:
        print(f"Loading ASR model: {ASR_MODEL_ID}", flush=True)
        processor = WhisperProcessor.from_pretrained(ASR_MODEL_ID)
        asr_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = WhisperForConditionalGeneration.from_pretrained(
            ASR_MODEL_ID,
            dtype=asr_dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )

        total = len(prompts)
        for prompt in prompts:
            index = int(prompt["index"])
            if index in existing:
                print(f"[{index + 1:03d}/{total:03d}] ASR result exists", flush=True)
                continue

            waveform = load_audio_16khz(audio_path(output_dir, index))
            input_features = processor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
            ).input_features.to(device=device, dtype=asr_dtype)
            with torch.inference_mode():
                predicted_ids = model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                )
            hypothesis = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()
            reference_normalized = normalize_english(str(prompt["text"]))
            hypothesis_normalized = normalize_english(hypothesis)
            measures = process_words(reference_normalized, hypothesis_normalized)
            row: Dict[str, object] = {
                **prompt,
                "audio": str(audio_path(output_dir, index).resolve()),
                "reference_normalized": reference_normalized,
                "hypothesis": hypothesis,
                "hypothesis_normalized": hypothesis_normalized,
                "wer": measures.wer,
                "substitutions": measures.substitutions,
                "deletions": measures.deletions,
                "insertions": measures.insertions,
                "hits": measures.hits,
            }
            existing[index] = row
            write_jsonl(results_path, [existing[key] for key in sorted(existing)])
            print(
                f"[{index + 1:03d}/{total:03d}] WER={measures.wer * 100:.3f}% "
                f"ASR={hypothesis!r}",
                flush=True,
            )

        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("All requested ASR rows already exist; transcription skipped.", flush=True)

    rows = [existing[int(prompt["index"])] for prompt in prompts]
    mean_utterance_wer = float(np.mean([float(row["wer"]) for row in rows]))
    errors = sum(
        int(row["substitutions"]) + int(row["deletions"]) + int(row["insertions"])
        for row in rows
    )
    reference_words = sum(
        int(row["hits"]) + int(row["substitutions"]) + int(row["deletions"])
        for row in rows
    )
    corpus_wer = errors / reference_words
    summary = {
        "samples": len(rows),
        "model": TTS_MODEL_ID,
        "speaker": rows[0].get("speaker", "Aiden"),
        "asr_model": ASR_MODEL_ID,
        "mean_utterance_wer_percent": round(mean_utterance_wer * 100, 6),
        "corpus_wer_percent": round(corpus_wer * 100, 6),
        "paper_reference_wer_percent": PAPER_WER_PERCENT,
        "delta_from_paper_percentage_points": round(
            mean_utterance_wer * 100 - PAPER_WER_PERCENT, 6
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("\nBaseline result", flush=True)
    print(f"Samples: {summary['samples']}", flush=True)
    print(
        f"Mean utterance WER: {summary['mean_utterance_wer_percent']:.3f}%",
        flush=True,
    )
    print(f"Corpus WER: {summary['corpus_wer_percent']:.3f}%", flush=True)
    print(f"Paper reference: {PAPER_WER_PERCENT:.3f}%", flush=True)
    print(f"Saved: {(output_dir / 'summary.json').resolve()}", flush=True)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({args.device}) but is unavailable")

    prompts = load_prompts(args.text_file.resolve(), args.limit)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(prompts)} prompts from {args.text_file.resolve()}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)

    if args.stage in ("all", "generate"):
        generate_audio(prompts, output_dir, args.device, args.speaker, args.overwrite)
    if args.stage in ("all", "score"):
        score_audio(prompts, output_dir, args.device, args.overwrite)


if __name__ == "__main__":
    main()
