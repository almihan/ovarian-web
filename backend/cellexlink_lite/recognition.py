"""Memory-bounded CellExLink NER inference for retrieved chunk records.

The span reconstruction follows CellExLink's offset-based BIO workflow:
long texts are tokenized into overlapping windows, logits for duplicate tokens
are averaged, and entity spans are reconstructed against raw-text offsets.
This module avoids Hugging Face ``datasets`` and ``Trainer`` so the Railway
worker keeps only the model, one small text batch, and its token windows in
memory.
"""

from __future__ import annotations

import gc
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

PathLike = str | Path


@dataclass(slots=True, frozen=True)
class EntitySpan:
    """One cell-type span relative to a source chunk string."""

    text: str
    start: int
    end: int
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention": self.text,
            "start": self.start,
            "end": self.end,
            "entity_type": "cell_type",
            "ner_label": self.label,
        }


def _canonical_id2label(raw_mapping: dict[Any, Any]) -> dict[int, str]:
    return {int(index): str(label) for index, label in raw_mapping.items()}


def _tag_parts(tag: str) -> tuple[str, str]:
    if tag == "O":
        return "O", ""
    if "-" in tag:
        prefix, label = tag.split("-", 1)
        return prefix.upper(), label
    return "B", tag


def reconstruct_entities(
    *,
    text: str,
    token_offsets: Sequence[tuple[int, int]],
    predicted_label_ids_by_offset: dict[tuple[int, int], int],
    id_to_label: dict[int, str],
    outside_label_id: int,
) -> list[EntitySpan]:
    """Convert averaged BIO token predictions into raw-text entity spans."""

    entities: list[EntitySpan] = []
    active_label: str | None = None
    active_start: int | None = None
    active_end: int | None = None

    def close_active() -> None:
        nonlocal active_label, active_start, active_end
        if active_label is None or active_start is None or active_end is None:
            return
        if active_end > active_start:
            entities.append(
                EntitySpan(
                    text=text[active_start:active_end],
                    start=active_start,
                    end=active_end,
                    label=active_label,
                )
            )
        active_label = None
        active_start = None
        active_end = None

    for token_start, token_end in token_offsets:
        if token_end <= token_start:
            continue
        label_id = predicted_label_ids_by_offset.get(
            (token_start, token_end), outside_label_id
        )
        tag = id_to_label.get(int(label_id), "O")
        prefix, entity_label = _tag_parts(tag)

        if prefix == "O":
            close_active()
            continue

        starts_new = (
            prefix == "B"
            or active_label is None
            or active_label != entity_label
        )
        if starts_new:
            close_active()
            active_label = entity_label
            active_start = token_start
            active_end = token_end
        else:
            active_end = max(active_end or token_end, token_end)

    close_active()
    return entities


def _set_torch_threads(cpu_threads: int) -> None:
    safe_threads = max(1, int(cpu_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(safe_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(safe_threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch

    torch.set_num_threads(safe_threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, safe_threads)))
    except RuntimeError:
        # PyTorch allows this setting only before parallel work begins.
        pass


def _effective_max_length(tokenizer: Any, config: Any, requested: int | None) -> int:
    candidates: list[int] = []
    if requested is not None and requested > 0:
        candidates.append(int(requested))

    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 8 <= tokenizer_limit < 1_000_000:
        candidates.append(tokenizer_limit)

    config_limit = getattr(config, "max_position_embeddings", None)
    if isinstance(config_limit, int) and config_limit >= 8:
        candidates.append(config_limit)

    return max(8, min(candidates or [512]))


class ChunkNER:
    """CellExLink NER model loaded only inside a short-lived worker process."""

    def __init__(
        self,
        *,
        model_name_or_path: PathLike,
        cache_dir: PathLike | None = None,
        max_seq_length: int | None = None,
        doc_stride: int = 128,
        window_batch_size: int = 4,
        cpu_threads: int = 2,
        trust_remote_code: bool = False,
    ) -> None:
        if window_batch_size < 1:
            raise ValueError("window_batch_size must be >= 1")
        if doc_stride < 0:
            raise ValueError("doc_stride must be >= 0")

        _set_torch_threads(cpu_threads)

        import torch
        from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.model_reference = str(model_name_or_path)
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.window_batch_size = int(window_batch_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        common_kwargs = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": trust_remote_code,
        }
        self.config = AutoConfig.from_pretrained(self.model_reference, **common_kwargs)
        if not getattr(self.config, "id2label", None):
            raise ValueError("The CellExLink NER checkpoint has no id2label mapping.")

        self.id_to_label = _canonical_id2label(dict(self.config.id2label))
        outside_candidates = [
            index for index, label in self.id_to_label.items() if label == "O"
        ]
        self.outside_label_id = outside_candidates[0] if outside_candidates else 0

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_reference,
            use_fast=True,
            **common_kwargs,
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("CellExLink NER requires a fast tokenizer with offsets.")

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_reference,
            config=self.config,
            **common_kwargs,
        )
        self.model.to(self.device)
        self.model.eval()

        self.max_seq_length = _effective_max_length(
            self.tokenizer, self.config, max_seq_length
        )
        max_stride = max(0, self.max_seq_length - 8)
        self.doc_stride = min(int(doc_stride), max_stride)

    def predict_texts(self, texts: Sequence[str]) -> list[list[EntitySpan]]:
        """Predict spans for a bounded batch of raw chunk strings."""

        if not texts:
            return []

        normalized_texts = [str(text or "") for text in texts]
        outputs: list[list[EntitySpan]] = [[] for _ in normalized_texts]
        nonempty_indices = [
            index for index, text in enumerate(normalized_texts) if text.strip()
        ]
        if not nonempty_indices:
            return outputs

        selected_texts = [normalized_texts[index] for index in nonempty_indices]
        encoded = self.tokenizer(
            selected_texts,
            truncation=True,
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            padding=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        sample_mapping = encoded.pop("overflow_to_sample_mapping").tolist()
        offset_mapping = encoded.pop("offset_mapping").tolist()
        special_tokens_mask = encoded.pop("special_tokens_mask").tolist()

        aggregated_logits: list[dict[tuple[int, int], np.ndarray]] = [
            {} for _ in selected_texts
        ]
        aggregated_counts: list[dict[tuple[int, int], int]] = [
            {} for _ in selected_texts
        ]

        window_count = len(sample_mapping)
        with self.torch.inference_mode():
            for start in range(0, window_count, self.window_batch_size):
                end = min(window_count, start + self.window_batch_size)
                model_inputs = {
                    key: value[start:end].to(self.device)
                    for key, value in encoded.items()
                    if hasattr(value, "shape")
                }
                logits = self.model(**model_inputs).logits.detach().cpu().float().numpy()

                for local_window, window_index in enumerate(range(start, end)):
                    sample_index = int(sample_mapping[window_index])
                    sample_logits = aggregated_logits[sample_index]
                    sample_counts = aggregated_counts[sample_index]
                    for token_logits, raw_offset, is_special in zip(
                        logits[local_window],
                        offset_mapping[window_index],
                        special_tokens_mask[window_index],
                    ):
                        token_start, token_end = int(raw_offset[0]), int(raw_offset[1])
                        if is_special or token_end <= token_start:
                            continue
                        key = (token_start, token_end)
                        if key in sample_logits:
                            sample_logits[key] += token_logits
                            sample_counts[key] += 1
                        else:
                            sample_logits[key] = np.asarray(token_logits, dtype=np.float32).copy()
                            sample_counts[key] = 1

        for selected_index, original_index in enumerate(nonempty_indices):
            score_sums = aggregated_logits[selected_index]
            counts = aggregated_counts[selected_index]
            predicted: dict[tuple[int, int], int] = {}
            for offset, score_sum in score_sums.items():
                averaged = score_sum / max(1, counts.get(offset, 1))
                predicted[offset] = int(np.argmax(averaged))

            ordered_offsets = sorted(score_sums, key=lambda item: (item[0], item[1]))
            outputs[original_index] = reconstruct_entities(
                text=normalized_texts[original_index],
                token_offsets=ordered_offsets,
                predicted_label_ids_by_offset=predicted,
                id_to_label=self.id_to_label,
                outside_label_id=self.outside_label_id,
            )

        return outputs

    def predict_records(
        self,
        records: Sequence[dict[str, Any]],
        *,
        text_key: str = "chunk",
        text_batch_size: int = 8,
    ) -> list[list[EntitySpan]]:
        """Predict records in small text batches while preserving input order."""

        if text_batch_size < 1:
            raise ValueError("text_batch_size must be >= 1")
        results: list[list[EntitySpan]] = []
        for start in range(0, len(records), text_batch_size):
            batch = records[start : start + text_batch_size]
            results.extend(
                self.predict_texts([str(record.get(text_key) or "") for record in batch])
            )
        return results

    def close(self) -> None:
        """Drop model objects before the worker process exits."""

        model = getattr(self, "model", None)
        tokenizer = getattr(self, "tokenizer", None)
        config = getattr(self, "config", None)
        self.model = None
        self.tokenizer = None
        self.config = None
        del model, tokenizer, config
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            try:
                self.torch.cuda.ipc_collect()
            except Exception:
                pass

    def __enter__(self) -> "ChunkNER":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def estimate_window_count(text_length: int, *, max_seq_length: int = 512) -> int:
    """Small diagnostic helper used only in tests and status estimates."""

    if text_length <= 0:
        return 0
    return max(1, math.ceil(text_length / max(1, max_seq_length * 3)))


__all__ = ["ChunkNER", "EntitySpan", "reconstruct_entities", "estimate_window_count"]
