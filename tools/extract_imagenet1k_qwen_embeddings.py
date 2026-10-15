#!/usr/bin/env python3
"""Export Qwen3-TTS raw text embeddings for 1,000 ImageNet class names.

Only the checkpoint shard containing ``talker.model.text_embedding.weight`` is
opened. The TTS model is not instantiated and no GPU is required.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
LABEL_SOURCE_ID = "google/vit-base-patch16-224"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "eval_outputs"
    / "projection"
    / "qwen3_tts_imagenet1k_1000_raw_text_embeddings.pt"
)
DEFAULT_BASELINE_TEXT = REPO_ROOT / "eval_data" / "minimax" / "text" / "english.txt"
ASSISTANT_PREFIX = "<|im_start|>assistant\n"
ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
CONTENT_PREFIX_TOKENS = 3
CONTENT_SUFFIX_TOKENS = 5
EMBEDDING_KEY_SUFFIX = "talker.model.text_embedding.weight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--label-source-id", default=LABEL_SOURCE_ID)
    parser.add_argument("--baseline-text", type=Path, default=DEFAULT_BASELINE_TEXT)
    parser.add_argument(
        "--output-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N labels for a quick test.",
    )
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 1000:
        parser.error("--limit must be between 1 and 1000")
    return args


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_imagenet_labels(repo_id: str) -> List[str]:
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", token=False)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    id2label = config.get("id2label", {})
    labels = [id2label.get(str(index)) for index in range(1000)]
    if any(label is None for label in labels):
        raise ValueError(f"{repo_id}/config.json does not contain 1,000 ordered labels")
    labels = [label.strip() for label in labels]

    # ImageNet uses the exact label "crane" for both a bird and a machine.
    # Keep 1,000 distinct training texts while preserving those meanings.
    if labels[134] == "crane" and labels[517] == "crane":
        labels[134] = "crane bird"
        labels[517] = "construction crane"
    if len(set(labels)) != 1000:
        raise ValueError("ImageNet label source contains duplicate class names")
    return labels


def load_baseline_texts(path: Path) -> List[str]:
    if not path.is_file():
        return []
    texts = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        texts.append(line.split("|", 1)[-1].strip())
    return texts


def find_embedding_weight(model_id: str) -> Tuple[torch.Tensor, str, str]:
    index_error = None
    try:
        index_path = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors.index.json",
            token=False,
        )
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        keys = [
            key
            for key in index["weight_map"]
            if key.endswith(EMBEDDING_KEY_SUFFIX)
        ]
        if len(keys) != 1:
            raise ValueError(f"Expected one text-embedding key, found: {keys}")
        key = keys[0]
        shard_name = index["weight_map"][key]
        shard_path = hf_hub_download(repo_id=model_id, filename=shard_name, token=False)
    except Exception as exc:
        index_error = exc
        shard_name = "model.safetensors"
        try:
            shard_path = hf_hub_download(repo_id=model_id, filename=shard_name, token=False)
        except Exception as single_error:
            raise RuntimeError(
                "Could not locate either a sharded or single safetensors checkpoint. "
                f"Index error: {index_error}; single-file error: {single_error}"
            ) from single_error
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            keys = [key for key in handle.keys() if key.endswith(EMBEDDING_KEY_SUFFIX)]
        if len(keys) != 1:
            raise ValueError(f"Expected one text-embedding key, found: {keys}")
        key = keys[0]

    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(key).contiguous()
    return weight, key, shard_name


def load_tokenizer(model_id: str):
    tokenizer_files = [
        hf_hub_download(repo_id=model_id, filename=name, token=False)
        for name in ("tokenizer_config.json", "vocab.json", "merges.txt")
    ]
    tokenizer_dir = Path(tokenizer_files[0]).parent
    if any(Path(path).parent != tokenizer_dir for path in tokenizer_files[1:]):
        raise RuntimeError("Tokenizer files resolved to different checkpoint revisions")
    return AutoTokenizer.from_pretrained(
        tokenizer_dir,
        use_fast=True,
        local_files_only=True,
        fix_mistral_regex=True,
    )


def content_layout(tokenizer, text: str) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[int], str]:
    wrapped = ASSISTANT_PREFIX + text + ASSISTANT_SUFFIX
    encoded = tokenizer(
        wrapped,
        return_tensors="pt",
        return_offsets_mapping=True,
        padding=False,
        truncation=False,
    )
    full_ids = encoded["input_ids"][0].long().cpu()
    full_offsets = encoded["offset_mapping"][0].long().cpu()
    content_ids = full_ids[CONTENT_PREFIX_TOKENS:-CONTENT_SUFFIX_TOKENS]
    content_offsets = full_offsets[CONTENT_PREFIX_TOKENS:-CONTENT_SUFFIX_TOKENS].clone()

    decoded = tokenizer.decode(
        content_ids.tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != text:
        raise ValueError(f"Content slice mismatch: expected {text!r}, decoded {decoded!r}")

    content_offsets[:, 0] -= len(ASSISTANT_PREFIX)
    content_offsets[:, 1] -= len(ASSISTANT_PREFIX)
    if bool(
        (content_offsets[:, 0] < 0).any()
        or (content_offsets[:, 1] > len(text)).any()
        or (content_offsets[:, 1] <= content_offsets[:, 0]).any()
    ):
        raise ValueError(f"Invalid tokenizer offsets for {text!r}")

    token_strings = tokenizer.convert_ids_to_tokens(content_ids.tolist())
    return content_ids, content_offsets, token_strings, full_ids.tolist(), wrapped


def write_manifest(path: Path, payload: Dict[str, object]) -> None:
    row_offsets = payload["row_offsets"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for index, text in enumerate(payload["prompt_texts"]):
            start = int(row_offsets[index])
            end = int(row_offsets[index + 1])
            record = {
                "index": index,
                "text": text,
                "text_sha256": payload["prompt_sha256"][index],
                "flat_token_start": start,
                "flat_token_end": end,
                "token_count": end - start,
                "token_ids": payload["token_ids"][start:end].tolist(),
                "token_strings": payload["token_strings"][start:end],
                "character_offsets": payload["character_offsets"][start:end].tolist(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = load_imagenet_labels(args.label_source_id)
    if args.limit is not None:
        labels = labels[: args.limit]

    baseline_texts = set(load_baseline_texts(args.baseline_text.resolve()))
    overlap = sorted(set(labels) & baseline_texts)
    if overlap:
        raise ValueError(f"Training labels overlap exact MiniMax evaluation prompts: {overlap}")

    print(f"Loaded {len(labels)} ImageNet class names", flush=True)
    print("Loading tokenizer and only the raw text-embedding tensor", flush=True)
    tokenizer = load_tokenizer(args.model_id)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for character offsets")
    embedding_weight, embedding_key, embedding_shard = find_embedding_weight(args.model_id)
    output_dtype = torch.float16 if args.output_dtype == "float16" else torch.float32

    embedding_parts = []
    token_id_parts = []
    character_offset_parts = []
    token_strings = []
    full_input_ids = []
    wrapped_texts = []
    row_offsets = [0]

    for index, text in enumerate(labels):
        content_ids, offsets, strings, all_ids, wrapped = content_layout(tokenizer, text)
        embedding_parts.append(embedding_weight[content_ids].to(dtype=output_dtype))
        token_id_parts.append(content_ids.to(dtype=torch.int32))
        character_offset_parts.append(offsets.to(dtype=torch.int32))
        token_strings.extend(strings)
        full_input_ids.append(all_ids)
        wrapped_texts.append(wrapped)
        row_offsets.append(row_offsets[-1] + len(content_ids))
        if (index + 1) % 100 == 0 or index + 1 == len(labels):
            print(f"[{index + 1:04d}/{len(labels):04d}]", flush=True)

    embeddings = torch.cat(embedding_parts).contiguous()
    token_ids = torch.cat(token_id_parts).contiguous()
    character_offsets = torch.cat(character_offset_parts).contiguous()
    row_offsets_tensor = torch.tensor(row_offsets, dtype=torch.int64)
    norms = embeddings.float().norm(dim=1)

    payload = {
        "format_version": 1,
        "kind": "qwen3_tts_imagenet1k_raw_text_embeddings",
        "purpose": "out_of_sample_projection_training",
        "benchmark_prompts_used_for_fitting": False,
        "evaluation_prompts_excluded": True,
        "text_dataset": "ImageNet-1K class names",
        "label_source_id": args.label_source_id,
        "model_id": args.model_id,
        "target_representation": "talker_raw_text_embedding_before_text_projection",
        "target_layer": "talker.model.text_embedding",
        "checkpoint_embedding_key": embedding_key,
        "checkpoint_embedding_shard": embedding_shard,
        "target_hidden_dim": int(embeddings.shape[1]),
        "storage_dtype": args.output_dtype,
        "content_prefix_token_count": CONTENT_PREFIX_TOKENS,
        "content_suffix_token_count": CONTENT_SUFFIX_TOKENS,
        "assistant_prefix": ASSISTANT_PREFIX,
        "assistant_suffix": ASSISTANT_SUFFIX,
        "alignment": "character_span_overlap_relative_to_raw_prompt",
        "prompt_texts": labels,
        "prompt_sha256": [sha256_text(text) for text in labels],
        "source_voices": [""] * len(labels),
        "wrapped_texts": wrapped_texts,
        "full_input_ids": full_input_ids,
        "row_offsets": row_offsets_tensor,
        "embeddings": embeddings,
        "token_ids": token_ids,
        "character_offsets": character_offsets,
        "token_strings": token_strings,
        "embedding_norm_min": float(norms.min()),
        "embedding_norm_median": float(norms.median()),
        "embedding_norm_max": float(norms.max()),
    }

    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary_output)
    os.replace(temporary_output, output_path)

    manifest_path = output_path.with_suffix(".jsonl")
    labels_path = output_path.with_suffix(".txt")
    write_manifest(manifest_path, payload)
    labels_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    artifact_hash = sha256_file(output_path)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{artifact_hash}  {output_path.name}\n",
        encoding="utf-8",
    )

    print("\nExport complete", flush=True)
    print(f"Classes: {len(labels)}", flush=True)
    print(f"Active text tokens: {len(embeddings)}", flush=True)
    print(f"Embeddings: {tuple(embeddings.shape)} {embeddings.dtype}", flush=True)
    print(f"Artifact: {output_path}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Labels: {labels_path}", flush=True)
    print(f"SHA-256: {artifact_hash}", flush=True)


if __name__ == "__main__":
    main()
