#!/usr/bin/env python3
"""Export Qwen3-TTS raw embeddings for a broad, benchmark-independent corpus.

The default corpus contains:
  * 1,000 ImageNet class names;
  * 4,000 WordNet concepts balanced across lexical categories;
  * 500 natural English sentences balanced across Brown corpus genres.

Only Qwen's raw text-embedding tensor is read. The TTS model is not
instantiated and no GPU is required.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import nltk
import torch
from nltk.corpus import brown, wordnet as wn
from nltk.tokenize.treebank import TreebankWordDetokenizer

try:
    from .extract_imagenet1k_qwen_embeddings import (
        DEFAULT_BASELINE_TEXT,
        LABEL_SOURCE_ID,
        MODEL_ID,
        content_layout,
        find_embedding_weight,
        load_baseline_texts,
        load_imagenet_labels,
        load_tokenizer,
        sha256_file,
        sha256_text,
    )
except ImportError:
    from extract_imagenet1k_qwen_embeddings import (
        DEFAULT_BASELINE_TEXT,
        LABEL_SOURCE_ID,
        MODEL_ID,
        content_layout,
        find_embedding_weight,
        load_baseline_texts,
        load_imagenet_labels,
        load_tokenizer,
        sha256_file,
        sha256_text,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "eval_outputs"
    / "projection"
    / "qwen3_tts_semantic_5500_raw_text_embeddings.pt"
)
WORDNET_COUNT = 4000
BROWN_SENTENCE_COUNT = 500
WORDNET_LABEL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9 '\-&.]{1,79}$")


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
    args = parser.parse_args()
    return args


def ensure_nltk_data() -> None:
    resources = [
        ("corpora/wordnet", "wordnet"),
        ("corpora/brown", "brown"),
    ]
    for resource, package in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.data.find(resource + ".zip")
            except LookupError:
                if not nltk.download(package, quiet=False):
                    raise RuntimeError(f"NLTK could not download {package!r}")


def normalized_key(text: str) -> str:
    return " ".join(text.casefold().split())


def clean_wordnet_label(label: str) -> str:
    label = label.replace("_", " ")
    label = re.sub(r"\s+", " ", label).strip()
    return label


def select_wordnet_concepts(
    count: int,
    excluded_texts: Iterable[str],
) -> Tuple[List[str], List[str], Dict[str, int]]:
    excluded = {normalized_key(text) for text in excluded_texts}
    groups: Dict[str, Dict[str, Tuple[int, str]]] = defaultdict(dict)

    for synset in wn.all_synsets():
        lemmas = synset.lemmas()
        if not lemmas:
            continue

        label = clean_wordnet_label(lemmas[0].name())
        key = normalized_key(label)
        if (
            key in excluded
            or len(label.split()) > 6
            or not WORDNET_LABEL_PATTERN.fullmatch(label)
        ):
            continue

        frequency = max(lemma.count() for lemma in lemmas)
        lexname = synset.lexname()
        previous = groups[lexname].get(key)
        candidate = (frequency, label)
        if previous is None or candidate[0] > previous[0]:
            groups[lexname][key] = candidate

    ranked: Dict[str, List[Tuple[int, str]]] = {}
    for lexname, values in groups.items():
        ranked[lexname] = sorted(
            values.values(),
            key=lambda item: (-item[0], normalized_key(item[1])),
        )

    lexnames = sorted(ranked)
    positions = {lexname: 0 for lexname in lexnames}
    selected: List[str] = []
    selected_lexnames: List[str] = []
    selected_keys = set(excluded)

    while len(selected) < count:
        made_progress = False
        for lexname in lexnames:
            candidates = ranked[lexname]
            position = positions[lexname]
            while position < len(candidates):
                _, label = candidates[position]
                position += 1
                key = normalized_key(label)
                if key in selected_keys:
                    continue
                selected.append(label)
                selected_lexnames.append(lexname)
                selected_keys.add(key)
                made_progress = True
                break
            positions[lexname] = position
            if len(selected) == count:
                break
        if not made_progress:
            raise RuntimeError(
                f"WordNet supplied only {len(selected)} usable unique concepts; "
                f"{count} were requested"
            )

    distribution = dict(Counter(selected_lexnames))
    return selected, selected_lexnames, distribution


def clean_brown_sentence(tokens: Sequence[str]) -> str:
    text = TreebankWordDetokenizer().detokenize(list(tokens))
    text = text.replace("``", '"').replace("''", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def select_brown_sentences(
    count: int,
    excluded_texts: Iterable[str],
) -> Tuple[List[str], List[str], Dict[str, int]]:
    excluded = {normalized_key(text) for text in excluded_texts}
    categories = sorted(brown.categories())
    candidates: Dict[str, List[str]] = {}

    for category in categories:
        category_sentences = []
        seen = set()
        for tokens in brown.sents(categories=category):
            word_count = sum(any(char.isalpha() for char in token) for token in tokens)
            if not 7 <= word_count <= 28:
                continue
            text = clean_brown_sentence(tokens)
            key = normalized_key(text)
            if (
                key in excluded
                or key in seen
                or len(text) < 30
                or len(text) > 220
            ):
                continue
            category_sentences.append(text)
            seen.add(key)
        candidates[category] = category_sentences

    positions = {category: 0 for category in categories}
    selected: List[str] = []
    selected_categories: List[str] = []
    selected_keys = set(excluded)

    # Walk through every genre in round-robin order. Advancing by a prime-sized
    # stride avoids taking one contiguous passage from the corpus.
    stride = 97
    ordered: Dict[str, List[str]] = {}
    for category in categories:
        values = candidates[category]
        if not values:
            ordered[category] = []
            continue
        indices = []
        used = set()
        index = 0
        while len(indices) < len(values):
            if index not in used:
                indices.append(index)
                used.add(index)
            index = (index + stride) % len(values)
            if index in used:
                index = next(
                    (candidate for candidate in range(len(values)) if candidate not in used),
                    0,
                )
        ordered[category] = [values[index] for index in indices]

    while len(selected) < count:
        made_progress = False
        for category in categories:
            values = ordered[category]
            position = positions[category]
            while position < len(values):
                text = values[position]
                position += 1
                key = normalized_key(text)
                if key in selected_keys:
                    continue
                selected.append(text)
                selected_categories.append(category)
                selected_keys.add(key)
                made_progress = True
                break
            positions[category] = position
            if len(selected) == count:
                break
        if not made_progress:
            raise RuntimeError(
                f"Brown supplied only {len(selected)} usable unique sentences; "
                f"{count} were requested"
            )

    distribution = dict(Counter(selected_categories))
    return selected, selected_categories, distribution


def build_corpus(
    label_source_id: str,
    baseline_text_path: Path,
) -> Tuple[List[str], List[str], Dict[str, object]]:
    ensure_nltk_data()
    if not baseline_text_path.is_file():
        raise FileNotFoundError(
            f"MiniMax exclusion file not found: {baseline_text_path}"
        )
    baseline_texts = load_baseline_texts(baseline_text_path)
    imagenet = load_imagenet_labels(label_source_id)

    wordnet, wordnet_lexnames, wordnet_distribution = select_wordnet_concepts(
        WORDNET_COUNT,
        excluded_texts=[*baseline_texts, *imagenet],
    )
    brown_sentences, brown_categories, brown_distribution = select_brown_sentences(
        BROWN_SENTENCE_COUNT,
        excluded_texts=[*baseline_texts, *imagenet, *wordnet],
    )

    texts = [*imagenet, *wordnet, *brown_sentences]
    text_kinds = (
        ["imagenet_class"] * len(imagenet)
        + ["wordnet_concept"] * len(wordnet)
        + ["brown_sentence"] * len(brown_sentences)
    )

    normalized = [normalized_key(text) for text in texts]
    if len(texts) != 5500 or len(set(normalized)) != len(texts):
        raise RuntimeError("The semantic corpus is not exactly 5,500 unique texts")

    baseline_keys = {normalized_key(text) for text in baseline_texts}
    overlap = sorted(set(normalized) & baseline_keys)
    if overlap:
        raise RuntimeError(f"Corpus contains exact MiniMax evaluation prompts: {overlap}")

    metadata = {
        "imagenet_count": len(imagenet),
        "wordnet_count": len(wordnet),
        "brown_sentence_count": len(brown_sentences),
        "wordnet_lexname_distribution": wordnet_distribution,
        "brown_category_distribution": brown_distribution,
        "wordnet_lexnames": wordnet_lexnames,
        "brown_categories": brown_categories,
    }
    return texts, text_kinds, metadata


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
                "text_kind": payload["text_kinds"][index],
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

    texts, text_kinds, corpus_metadata = build_corpus(
        args.label_source_id,
        args.baseline_text.resolve(),
    )

    print(
        "Corpus: 1,000 ImageNet classes + 4,000 WordNet concepts "
        "+ 500 Brown sentences",
        flush=True,
    )
    print("Loading tokenizer and only the raw text-embedding tensor", flush=True)
    tokenizer = load_tokenizer(args.model_id)
    embedding_weight, embedding_key, embedding_shard = find_embedding_weight(args.model_id)
    output_dtype = torch.float16 if args.output_dtype == "float16" else torch.float32

    embedding_parts = []
    token_id_parts = []
    character_offset_parts = []
    token_strings = []
    full_input_ids = []
    wrapped_texts = []
    row_offsets = [0]

    for index, text in enumerate(texts):
        content_ids, offsets, strings, all_ids, wrapped = content_layout(tokenizer, text)
        embedding_parts.append(embedding_weight[content_ids].to(dtype=output_dtype))
        token_id_parts.append(content_ids.to(dtype=torch.int32))
        character_offset_parts.append(offsets.to(dtype=torch.int32))
        token_strings.extend(strings)
        full_input_ids.append(all_ids)
        wrapped_texts.append(wrapped)
        row_offsets.append(row_offsets[-1] + len(content_ids))
        if (index + 1) % 250 == 0 or index + 1 == len(texts):
            print(f"[{index + 1:04d}/{len(texts):04d}]", flush=True)

    embeddings = torch.cat(embedding_parts).contiguous()
    token_ids = torch.cat(token_id_parts).contiguous()
    character_offsets = torch.cat(character_offset_parts).contiguous()
    row_offsets_tensor = torch.tensor(row_offsets, dtype=torch.int64)
    norms = embeddings.float().norm(dim=1)

    payload = {
        "format_version": 1,
        "kind": "qwen3_tts_semantic_5500_raw_text_embeddings",
        "purpose": "out_of_sample_projection_training",
        "benchmark_prompts_used_for_fitting": False,
        "evaluation_prompts_excluded": True,
        "text_dataset": "ImageNet-1K + balanced WordNet + balanced Brown",
        "text_kinds": text_kinds,
        "corpus_metadata": corpus_metadata,
        "label_source_id": args.label_source_id,
        "model_id": args.model_id,
        "target_representation": "talker_raw_text_embedding_before_text_projection",
        "target_layer": "talker.model.text_embedding",
        "checkpoint_embedding_key": embedding_key,
        "checkpoint_embedding_shard": embedding_shard,
        "target_hidden_dim": int(embeddings.shape[1]),
        "storage_dtype": args.output_dtype,
        "content_prefix_token_count": 3,
        "content_suffix_token_count": 5,
        "assistant_prefix": "<|im_start|>assistant\n",
        "assistant_suffix": "<|im_end|>\n<|im_start|>assistant\n",
        "alignment": "character_span_overlap_relative_to_raw_prompt",
        "prompt_texts": texts,
        "prompt_sha256": [sha256_text(text) for text in texts],
        "source_voices": [""] * len(texts),
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
    texts_path = output_path.with_suffix(".txt")
    write_manifest(manifest_path, payload)
    texts_path.write_text("\n".join(texts) + "\n", encoding="utf-8")
    artifact_hash = sha256_file(output_path)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{artifact_hash}  {output_path.name}\n",
        encoding="utf-8",
    )

    unique_token_count = len(set(token_ids.tolist()))
    print("\nExport complete", flush=True)
    print(f"Texts: {len(texts)}", flush=True)
    print(f"Active text tokens: {len(embeddings)}", flush=True)
    print(f"Unique Qwen token IDs: {unique_token_count}", flush=True)
    print(f"Embeddings: {tuple(embeddings.shape)} {embeddings.dtype}", flush=True)
    print(f"Artifact: {output_path}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Texts: {texts_path}", flush=True)
    print(f"SHA-256: {artifact_hash}", flush=True)


if __name__ == "__main__":
    main()
