"""
Multi-format dataloader for pretraining and SFT.

Reads many on-disk formats (txt, csv, jsonl/json, parquet) and HuggingFace hub
repos, normalizes each record into chat messages, then tokenizes for either
plain causal LM (``pretrain``) or instruction-tuning with the prompt masked
(``sft``). The ``datasets`` package is used for csv/parquet/hub; txt and jsonl
have a pure-Python fallback so the common case needs no extra dependency.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from torch.utils.data import DataLoader, Dataset

from cambium.exceptions import DataError

logger = logging.getLogger(__name__)

_FORMAT_BY_EXT = {
    ".txt": "txt",
    ".text": "txt",
    ".jsonl": "jsonl",
    ".json": "json",
    ".csv": "csv",
    ".parquet": "parquet",
}

# Schema hints select a formatter but are stored as jsonl on disk.
_SCHEMA_TO_FILE_FORMAT = {
    "alpaca": "jsonl",
    "chat": "jsonl",
    "messages": "jsonl",
}


@dataclass
class DataConfig:
    """Configuration for :func:`build_text_dataloader`.

    Args:
        source: Local file, local directory, or HuggingFace repo id.
        mode: ``"pretrain"`` (full causal LM over selected content) or ``"sft"``
            (chat-template render, loss on assistant turns only).
        format: ``"auto"`` (detect from extension / inspect record) or one of
            ``txt``, ``jsonl``, ``json``, ``csv``, ``parquet``, ``alpaca``,
            ``chat``.
        formatter: Callable mapping a record to a list of ``{role, content}``
            messages. If omitted, chosen from ``format``.
        text_fields: In pretrain mode, which message roles to keep in the text
            stream (``None`` = all). Ignored in sft mode.
        split: HuggingFace hub split name.
        streaming: Stream the hub dataset instead of downloading it.
        num_samples: Optional cap on records loaded.
        max_length: Maximum token length per example.
        add_eos: Append ``eos_token`` in pretrain mode.
        cache_dir: Optional HuggingFace cache directory.
    """

    source: str
    mode: str = "pretrain"
    format: str = "auto"
    formatter: Callable[[Any], list[dict]] | None = None
    text_fields: list[str] | None = None
    split: str = "train"
    streaming: bool = False
    num_samples: int | None = None
    max_length: int = 512
    add_eos: bool = True
    cache_dir: str | None = None


class TextFormatter:
    """Treat a record as plain text wrapped in a single user message."""

    def __init__(self, field: str | None = None):
        self.field = field

    def __call__(self, record: Any) -> list[dict]:
        if isinstance(record, str):
            return [{"role": "user", "content": record}]
        if not isinstance(record, dict):
            raise DataError(f"TextFormatter cannot handle record of type {type(record).__name__}")

        if self.field is not None:
            if self.field not in record:
                raise DataError(f"Field {self.field!r} not found in record keys {list(record)}")
            return [{"role": "user", "content": str(record[self.field])}]

        # Prefer a conventional "text" key, otherwise join all values.
        if "text" in record:
            return [{"role": "user", "content": str(record["text"])}]
        content = "\n".join(str(v) for v in record.values())
        return [{"role": "user", "content": content}]


class AlpacaFormatter:
    """Map an alpaca-style record (instruction/input/output) to two messages."""

    def __init__(
        self,
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output",
        input_optional: bool = True,
    ):
        self.instruction_field = instruction_field
        self.input_field = input_field
        self.output_field = output_field
        self.input_optional = input_optional

    def __call__(self, record: Any) -> list[dict]:
        if not isinstance(record, dict):
            raise DataError(f"AlpacaFormatter requires a dict record, got {type(record).__name__}")

        instruction = record.get(self.instruction_field)
        output = record.get(self.output_field)
        if not instruction:
            raise DataError(f"Alpaca record missing {self.instruction_field!r}")
        if not output:
            raise DataError(f"Alpaca record missing {self.output_field!r}")

        inp = record.get(self.input_field)
        if inp is None and not self.input_optional:
            raise DataError(f"Alpaca record missing required {self.input_field!r}")

        user_content = str(instruction)
        if inp:
            user_content = f"{instruction}\n\n{inp}"

        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": str(output)},
        ]


class ChatFormatter:
    """Pass through an OpenAI-style ``messages`` field."""

    def __init__(self, messages_field: str = "messages"):
        self.messages_field = messages_field

    def __call__(self, record: Any) -> list[dict]:
        if not isinstance(record, dict):
            raise DataError(f"ChatFormatter requires a dict record, got {type(record).__name__}")
        messages = record.get(self.messages_field)
        if not messages:
            raise DataError(f"Chat record missing {self.messages_field!r}")
        normalized = []
        for msg in messages:
            if "role" not in msg or "content" not in msg:
                raise DataError(f"Each message needs 'role' and 'content'; got {msg!r}")
            normalized.append({"role": msg["role"], "content": str(msg["content"])})
        return normalized


def _resolve_formatter(config: DataConfig, sample_record: Any) -> Callable[[Any], list[dict]]:
    """Pick a formatter from an explicit callable, format hint, or record shape."""
    if config.formatter is not None:
        return config.formatter

    fmt = config.format
    if fmt == "alpaca":
        return AlpacaFormatter()
    if fmt in ("chat", "messages"):
        return ChatFormatter()
    if fmt == "auto" and isinstance(sample_record, dict):
        if "messages" in sample_record:
            return ChatFormatter()
        if "instruction" in sample_record or "output" in sample_record:
            return AlpacaFormatter()
    return TextFormatter()


def _infer_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return _FORMAT_BY_EXT.get(ext, "jsonl")


def _expand_files(source: str) -> list[str]:
    if os.path.isdir(source):
        files: list[str] = []
        for ext in _FORMAT_BY_EXT:
            files.extend(glob.glob(os.path.join(source, f"*{ext}")))
        return sorted(files)
    return [source]


def _load_txt(files: list[str]) -> Iterator[str]:
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.strip():
                    yield line


def _load_jsonl(files: list[str]) -> Iterator[dict]:
    for path in files:
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        lines = [l for l in content.splitlines() if l.strip()]
        if len(lines) == 1:
            # A single line is a whole JSON array or object, not jsonl.
            data = json.loads(lines[0])
            if isinstance(data, list):
                yield from data
            else:
                yield data
            continue
        for line in lines:
            parsed = json.loads(line)
            if isinstance(parsed, list):
                yield from parsed
            else:
                yield parsed


def _load_with_datasets(kind: str, files: list[str]) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise DataError(
            f"{kind} format requires the 'datasets' package. "
            "Install with: pip install 'cambium-llm[train]'"
        ) from e

    ds = load_dataset(kind, data_files=files)
    split_name = "train" if "train" in ds else next(iter(ds))
    for ex in ds[split_name]:
        yield dict(ex)


def _load_hub(source: str, split: str, streaming: bool, cache_dir: str | None) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise DataError(
            "Loading from the HuggingFace hub requires the 'datasets' package. "
            "Install with: pip install 'cambium-llm[train]'"
        ) from e

    ds = load_dataset(source, split=split, streaming=streaming, cache_dir=cache_dir)
    for ex in ds:
        yield dict(ex)


def load_records(
    source: str,
    format: str = "auto",
    split: str = "train",
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Iterator[Any]:
    """Yield raw records from a local path/directory or HuggingFace hub repo."""
    if os.path.exists(source):
        files = _expand_files(source)
        if not files:
            raise DataError(f"No files found at {source}")
        fmt = format if format and format != "auto" else _infer_format(files[0])
        fmt = _SCHEMA_TO_FILE_FORMAT.get(fmt, fmt)
        if fmt in ("txt", "text"):
            yield from _load_txt(files)
        elif fmt in ("jsonl", "json"):
            yield from _load_jsonl(files)
        elif fmt == "csv":
            yield from _load_with_datasets("csv", files)
        elif fmt == "parquet":
            yield from _load_with_datasets("parquet", files)
        else:
            raise DataError(f"Unsupported format {fmt!r} for local source {source}")
    else:
        yield from _load_hub(source, split, streaming, cache_dir)


def _tokenize_pretrain(
    messages: list[dict],
    tokenizer: Any,
    max_length: int,
    add_eos: bool,
    text_fields: list[str] | None,
) -> dict[str, list[int]]:
    """Full causal LM over selected message content."""
    contents = [
        msg["content"] for msg in messages if text_fields is None or msg["role"] in text_fields
    ]
    text = "\n".join(contents)
    if not text:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    trunc_max = max_length - 1 if add_eos else max_length
    enc = tokenizer(text, truncation=True, max_length=trunc_max)
    input_ids = list(enc["input_ids"])

    if add_eos and tokenizer.eos_token_id is not None:
        input_ids.append(tokenizer.eos_token_id)

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": list(input_ids),
    }


def _render_ids(
    tokenizer: Any, messages: list[dict], add_generation_prompt: bool = False
) -> list[int]:
    """Token ids for a chat-template rendering (handles Encoding vs list returns)."""
    enc = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=True, add_generation_prompt=add_generation_prompt
    )
    return list(enc["input_ids"])


def _assistant_mask_via_prefixes(messages: list[dict], tokenizer: Any, total_len: int) -> list[int]:
    """Compute an assistant mask when the chat template lacks ``{% generation %}``.

    For each assistant turn, the tokens between the prompt-with-generation-header
    and the prompt-plus-the-assistant-turn are the assistant's output. ChatML-style
    templates put role markers on their own token IDs, so these spans align cleanly
    with the full rendered sequence.
    """
    mask = [0] * total_len
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix = _render_ids(tokenizer, messages[:i], add_generation_prompt=True)
        upto = _render_ids(tokenizer, messages[: i + 1])
        start = len(prefix)
        end = min(len(upto), total_len)
        for j in range(start, max(start, end)):
            mask[j] = 1
    return mask


def _template_marks_generation(tokenizer: Any) -> bool:
    """True only if the template uses the ``{% generation %}`` block tag."""
    tmpl = getattr(tokenizer, "chat_template", None) or ""
    return "{% generation" in tmpl


def _tokenize_sft(
    messages: list[dict],
    tokenizer: Any,
    max_length: int,
) -> dict[str, list[int]]:
    """Render via chat template; mask all non-assistant tokens from the loss."""
    if not getattr(tokenizer, "chat_template", None):
        raise DataError(
            "SFT mode requires a tokenizer with a chat_template. "
            "Set one on the tokenizer or use a model that ships one."
        )
    try:
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            truncation=True,
            max_length=max_length,
        )
    except (ValueError, TypeError) as e:
        raise DataError(f"Tokenizer cannot render the chat template: {e}") from e

    input_ids = list(enc["input_ids"])
    attention_mask = list(enc.get("attention_mask", [1] * len(input_ids)))

    # Prefer the template's own generation markers when present; otherwise derive
    # the assistant spans by comparing prefix renderings.
    assistant_mask = [0] * len(input_ids)
    if _template_marks_generation(tokenizer):
        try:
            with_mask = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
                truncation=True,
                max_length=max_length,
            )
            assistant_mask = list(with_mask.get("assistant_masks", [0] * len(input_ids)))
        except (ValueError, TypeError):
            assistant_mask = []

    if not any(assistant_mask):
        assistant_mask = _assistant_mask_via_prefixes(messages, tokenizer, len(input_ids))

    # Loss only on tokens the assistant produced.
    labels = [tok if masked else -100 for tok, masked in zip(input_ids, assistant_mask)]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class TextDataset(Dataset):
    """Holds pre-tokenized examples for dynamic-padding collation."""

    def __init__(self, examples: list[dict[str, list[int]]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]


def _make_collate(pad_token_id: int) -> Callable[[list[dict]], dict]:
    """Pad each batch to the longest example; labels pad to -100."""

    def collate(batch: list[dict]) -> dict:
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids, attention_mask, labels = [], [], []
        for ex in batch:
            pad = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [pad_token_id] * pad)
            attention_mask.append(ex["attention_mask"] + [0] * pad)
            labels.append(ex["labels"] + [-100] * pad)
        import torch

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def build_text_dataloader(
    tokenizer: Any,
    config: DataConfig | None = None,
    *,
    batch_size: int = 4,
    shuffle: bool = True,
    source: str | None = None,
    **overrides: Any,
) -> DataLoader:
    """Build a ready-to-train DataLoader from a multi-format data source.

    Args:
        tokenizer: A HuggingFace tokenizer. ``pad_token`` is set to
            ``eos_token`` if missing.
        config: A :class:`DataConfig`. If omitted, one is built from ``source``
            and ``overrides``.
        batch_size: DataLoader batch size.
        shuffle: Shuffle the dataset each epoch.
        source: Convenience kwarg used when ``config`` is None.
        **overrides: Extra :class:`DataConfig` fields when ``config`` is None.

    Returns:
        A ``torch.utils.data.DataLoader`` yielding ``{input_ids, attention_mask,
        labels}`` batches.
    """
    if config is None:
        if source is None:
            raise DataError("Provide a DataConfig or a source= argument")
        config = DataConfig(source=source, **overrides)

    if config.mode not in ("pretrain", "sft"):
        raise DataError(f"Unknown mode {config.mode!r}; use 'pretrain' or 'sft'")

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    import itertools

    record_iter = load_records(
        config.source, config.format, config.split, config.streaming, config.cache_dir
    )
    if config.num_samples is not None:
        records = list(itertools.islice(record_iter, config.num_samples))
    else:
        records = list(record_iter)
    if not records:
        raise DataError(f"No records loaded from {config.source!r}")

    formatter = _resolve_formatter(config, records[0])

    examples: list[dict[str, list[int]]] = []
    for record in records:
        messages = formatter(record)
        if config.mode == "pretrain":
            example = _tokenize_pretrain(
                messages, tokenizer, config.max_length, config.add_eos, config.text_fields
            )
        else:
            example = _tokenize_sft(messages, tokenizer, config.max_length)
        if not example["input_ids"]:
            continue
        examples.append(example)

    if not examples:
        raise DataError(f"All records from {config.source!r} tokenized to empty sequences")

    dataset = TextDataset(examples)
    collate = _make_collate(tokenizer.pad_token_id)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
