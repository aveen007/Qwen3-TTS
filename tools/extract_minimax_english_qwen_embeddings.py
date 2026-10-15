#!/usr/bin/env python3
"""Export native Qwen3-TTS raw text embeddings for the 100 English prompts.

The artifact is intended for the deliberate same-data projection sanity check:
fit a source-to-Qwen projector on these exact benchmark prompts, project their
source features on Kaggle, and later inject the projected vectors immediately
before Qwen3-TTS's ``talker.text_projection`` layer.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Qwen3-TTS only needs the PyTorch Transformers backend. This also avoids the
# TensorFlow/oneDNN shutdown crash observed in the server base environment.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import torch

from qwen_tts import Qwen3TTSModel


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEXT_FILE = REPO_ROOT / "eval_data" / "minimax" / "text" / "english.txt"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "eval_outputs"
    / "projection"
    / "qwen3_tts_minimax_english_100_raw_text_embeddings.pt"
)
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
ASSISTANT_PREFIX = "<|im_start|>assistant\n"
ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
CONTENT_PREFIX_TOKENS = 3
CONTENT_SUFFIX_TOKENS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", type=Path, default=DEFAULT_TEXT_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Storage dtype only; embedding lookup runs in the model dtype.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Export only the first N prompts for debugging. Omit for all 100.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def load_prompts(text_file: Path, limit: Optional[int]) -> List[Dict[str, str]]:
    if not text_file.is_file():
        raise FileNotFoundError(f"MiniMax English text file not found: {text_file}")

    records: List[Dict[str, str]] = []
    for raw_line in text_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "|" in line:
            source_voice, text = line.split("|", 1)
        else:
            source_voice, text = "", line
        text = text.strip()
        if not text:
            raise ValueError(f"Empty prompt at non-empty source line {len(records) + 1}")
        records.append({"source_voice": source_voice.strip(), "text": text})

    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError(f"No prompts found in {text_file}")
    return records


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_layout(
    tokenizer,
    wrapped_text: str,
    raw_text: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[int]]:
    encoded = tokenizer(
        wrapped_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        padding=False,
        truncation=False,
    )
    full_ids = encoded["input_ids"][0].long().cpu()
    full_offsets = encoded["offset_mapping"][0].long().cpu()
    if len(full_ids) <= CONTENT_PREFIX_TOKENS + CONTENT_SUFFIX_TOKENS:
        raise ValueError(f"Unexpected wrapped token count for {raw_text!r}")

    content_ids = full_ids[CONTENT_PREFIX_TOKENS:-CONTENT_SUFFIX_TOKENS]
    wrapper_offsets = full_offsets[CONTENT_PREFIX_TOKENS:-CONTENT_SUFFIX_TOKENS]
    decoded = tokenizer.decode(
        content_ids.tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != raw_text:
        raise ValueError(
            "Qwen content-token slice did not decode to the original prompt. "
            f"Expected {raw_text!r}, decoded {decoded!r}."
        )

    raw_start = len(ASSISTANT_PREFIX)
    raw_end = raw_start + len(raw_text)
    relative_offsets = wrapper_offsets.clone()
    relative_offsets[:, 0] -= raw_start
    relative_offsets[:, 1] -= raw_start
    if bool(
        (relative_offsets[:, 0] < 0).any()
        or (relative_offsets[:, 1] > len(raw_text)).any()
        or (relative_offsets[:, 1] <= relative_offsets[:, 0]).any()
    ):
        raise ValueError(
            f"Invalid content offsets for {raw_text!r}: "
            f"wrapper text span is [{raw_start}, {raw_end})"
        )

    token_strings = tokenizer.convert_ids_to_tokens(content_ids.tolist())
    return content_ids, relative_offsets, token_strings, full_ids.tolist()


def write_manifest(
    path: Path,
    prompts: List[Dict[str, str]],
    prompt_hashes: List[str],
    row_offsets: torch.Tensor,
    token_ids: torch.Tensor,
    character_offsets: torch.Tensor,
    token_strings: List[str],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for index, prompt in enumerate(prompts):
            start = int(row_offsets[index])
            end = int(row_offsets[index + 1])
            record = {
                "index": index,
                "source_voice": prompt["source_voice"],
                "text": prompt["text"],
                "text_sha256": prompt_hashes[index],
                "flat_token_start": start,
                "flat_token_end": end,
                "token_count": end - start,
                "token_ids": token_ids[start:end].tolist(),
                "token_strings": token_strings[start:end],
                "character_offsets": character_offsets[start:end].tolist(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({args.device}) but is unavailable")

    text_file = args.text_file.resolve()
    output_path = args.output.resolve()
    prompts = load_prompts(text_file, args.limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(prompts)} prompts from {text_file}", flush=True)
    print(f"Loading {args.model_id} on {args.device}", flush=True)

    tts = Qwen3TTSModel.from_pretrained(
        args.model_id,
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    tokenizer = tts.processor.tokenizer
    if not tokenizer.is_fast:
        raise RuntimeError("A fast Qwen tokenizer is required to export character offsets")

    embedding_layer = tts.model.talker.get_text_embeddings()
    embedding_device = embedding_layer.weight.device
    output_dtype = torch.float16 if args.output_dtype == "float16" else torch.float32

    embedding_parts: List[torch.Tensor] = []
    token_id_parts: List[torch.Tensor] = []
    character_offset_parts: List[torch.Tensor] = []
    token_strings: List[str] = []
    full_input_ids: List[List[int]] = []
    wrapped_texts: List[str] = []
    row_offsets = [0]

    with torch.inference_mode():
        for index, prompt in enumerate(prompts):
            text = prompt["text"]
            wrapped_text = tts._build_assistant_text(text)
            expected_wrapped = ASSISTANT_PREFIX + text + ASSISTANT_SUFFIX
            if wrapped_text != expected_wrapped:
                raise RuntimeError("Qwen assistant wrapper changed; update the extractor before continuing")

            content_ids, offsets, strings, all_ids = content_layout(
                tokenizer, wrapped_text, text
            )

            # Validate against the exact processor path used by generate_custom_voice.
            inference_ids = tts._tokenize_texts([wrapped_text])[0][0].long().cpu()
            if not torch.equal(inference_ids, torch.tensor(all_ids, dtype=torch.long)):
                raise RuntimeError(f"Tokenizer mismatch for prompt {index}")

            embeddings = embedding_layer(content_ids.to(embedding_device))
            embedding_parts.append(embeddings.to(device="cpu", dtype=output_dtype))
            token_id_parts.append(content_ids.to(dtype=torch.int32))
            character_offset_parts.append(offsets.to(dtype=torch.int32))
            token_strings.extend(strings)
            full_input_ids.append(all_ids)
            wrapped_texts.append(wrapped_text)
            row_offsets.append(row_offsets[-1] + len(content_ids))
            print(
                f"[{index + 1:03d}/{len(prompts):03d}] {len(content_ids):3d} tokens: {text}",
                flush=True,
            )

    flat_embeddings = torch.cat(embedding_parts).contiguous()
    flat_token_ids = torch.cat(token_id_parts).contiguous()
    flat_character_offsets = torch.cat(character_offset_parts).contiguous()
    row_offsets_tensor = torch.tensor(row_offsets, dtype=torch.int64)
    prompt_texts = [prompt["text"] for prompt in prompts]
    prompt_hashes = [sha256_text(text) for text in prompt_texts]

    if flat_embeddings.shape[0] != int(row_offsets_tensor[-1]):
        raise RuntimeError("Flattened embedding count does not match row offsets")
    if flat_embeddings.shape[0] != len(flat_token_ids):
        raise RuntimeError("Embedding and token-ID counts differ")
    if flat_embeddings.shape[0] != len(flat_character_offsets):
        raise RuntimeError("Embedding and character-offset counts differ")

    payload = {
        "format_version": 1,
        "kind": "qwen3_tts_minimax_english_raw_text_embeddings",
        "purpose": "same_100_prompt_projection_sanity_check",
        "benchmark_prompts_used_for_fitting": True,
        "benchmark": "MiniMaxAI/TTS-Multilingual-Test-Set:text/english.txt",
        "model_id": args.model_id,
        "target_representation": "talker_raw_text_embedding_before_text_projection",
        "target_layer": "talker.model.text_embedding",
        "downstream_layer": "talker.text_projection",
        "target_hidden_dim": int(flat_embeddings.shape[1]),
        "storage_dtype": args.output_dtype,
        "content_prefix_token_count": CONTENT_PREFIX_TOKENS,
        "content_suffix_token_count": CONTENT_SUFFIX_TOKENS,
        "assistant_prefix": ASSISTANT_PREFIX,
        "assistant_suffix": ASSISTANT_SUFFIX,
        "alignment": "character_span_overlap_relative_to_raw_prompt",
        "prompt_texts": prompt_texts,
        "prompt_sha256": prompt_hashes,
        "source_voices": [prompt["source_voice"] for prompt in prompts],
        "wrapped_texts": wrapped_texts,
        "full_input_ids": full_input_ids,
        "row_offsets": row_offsets_tensor,
        "embeddings": flat_embeddings,
        "token_ids": flat_token_ids,
        "character_offsets": flat_character_offsets,
        "token_strings": token_strings,
        "embedding_norm_min": float(flat_embeddings.float().norm(dim=1).min()),
        "embedding_norm_median": float(flat_embeddings.float().norm(dim=1).median()),
        "embedding_norm_max": float(flat_embeddings.float().norm(dim=1).max()),
    }

    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary_output)
    os.replace(temporary_output, output_path)

    manifest_path = output_path.with_suffix(".jsonl")
    write_manifest(
        manifest_path,
        prompts,
        prompt_hashes,
        row_offsets_tensor,
        flat_token_ids,
        flat_character_offsets,
        token_strings,
    )
    artifact_hash = sha256_file(output_path)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{artifact_hash}  {output_path.name}\n", encoding="utf-8"
    )

    print("\nExport complete", flush=True)
    print(f"Prompts: {len(prompts)}", flush=True)
    print(f"Active text tokens: {len(flat_embeddings)}", flush=True)
    print(f"Embeddings: {tuple(flat_embeddings.shape)} {flat_embeddings.dtype}", flush=True)
    print(
        "Raw embedding norms: "
        f"min={payload['embedding_norm_min']:.6f}, "
        f"median={payload['embedding_norm_median']:.6f}, "
        f"max={payload['embedding_norm_max']:.6f}",
        flush=True,
    )
    print(f"Artifact: {output_path}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"SHA-256: {artifact_hash}", flush=True)


if __name__ == "__main__":
    main()
