"""Unit tests for the multi-format dataloader (cambium.training.data).

No network: a tiny in-memory tokenizer is built via the ``tokenizers`` package
so tokenization, chat templates, and assistant masking can be tested offline.
"""

import json
import sys

import pytest
import torch
from torch.utils.data import DataLoader

from cambium.exceptions import DataError
from cambium.training.data import (
    AlpacaFormatter,
    ChatFormatter,
    DataConfig,
    TextFormatter,
    build_text_dataloader,
    load_records,
)


def _make_tokenizer():
    """Build a tiny offline tokenizer with a chat template that supports masks."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    words = [
        "[PAD]",
        "[EOS]",
        "[UNK]",
        "user",
        "assistant",
        "end",
        "hello",
        "world",
        "foo",
        "bar",
        "instruction",
        "input",
        "output",
    ]
    vocab = {w: i for i, w in enumerate(words)}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    # Mark assistant content as a generation block so return_assistant_tokens_mask works.
    fast.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'assistant' %}"
        "{% generation %}{{ message['content'] + ' end ' }}{% endgeneration %}"
        "{% else %}{{ message['role'] + ' ' + message['content'] + ' end ' }}{% endif %}"
        "{% endfor %}"
    )
    return fast


@pytest.fixture
def tokenizer():
    return _make_tokenizer()


class TestFormatters:
    def test_text_formatter_str(self):
        msgs = TextFormatter()("hello world")
        assert msgs == [{"role": "user", "content": "hello world"}]

    def test_text_formatter_dict_with_text_key(self):
        msgs = TextFormatter()({"text": "hello", "id": 7})
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_text_formatter_dict_with_field(self):
        msgs = TextFormatter(field="body")({"body": "abc", "extra": "x"})
        assert msgs == [{"role": "user", "content": "abc"}]

    def test_text_formatter_missing_field_raises(self):
        with pytest.raises(DataError, match="Field 'body'"):
            TextFormatter(field="body")({"text": "abc"})

    def test_alpaca_with_input(self):
        msgs = AlpacaFormatter()({"instruction": "do X", "input": "ctx", "output": "done"})
        assert msgs == [
            {"role": "user", "content": "do X\n\nctx"},
            {"role": "assistant", "content": "done"},
        ]

    def test_alpaca_without_input(self):
        msgs = AlpacaFormatter()({"instruction": "do X", "output": "done"})
        assert msgs == [
            {"role": "user", "content": "do X"},
            {"role": "assistant", "content": "done"},
        ]

    def test_alpaca_missing_instruction_raises(self):
        with pytest.raises(DataError, match="missing 'instruction'"):
            AlpacaFormatter()({"output": "done"})

    def test_chat_formatter_passthrough(self):
        record = {
            "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
        }
        msgs = ChatFormatter()(record)
        assert msgs == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]

    def test_chat_formatter_missing_messages_raises(self):
        with pytest.raises(DataError, match="missing 'messages'"):
            ChatFormatter()({"text": "hi"})


class TestLoadRecords:
    def test_load_txt(self, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("line one\n\nline two\nline three\n", encoding="utf-8")
        records = list(load_records(str(path), format="txt"))
        assert records == ["line one", "line two", "line three"]

    def test_load_jsonl(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text(
            json.dumps({"text": "a"}) + "\n" + json.dumps({"text": "b"}) + "\n",
            encoding="utf-8",
        )
        records = list(load_records(str(path)))
        assert records == [{"text": "a"}, {"text": "b"}]

    def test_load_json_array(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps([{"text": "a"}, {"text": "b"}]), encoding="utf-8")
        records = list(load_records(str(path)))
        assert records == [{"text": "a"}, {"text": "b"}]

    def test_load_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
        (tmp_path / "b.txt").write_text("beta\n", encoding="utf-8")
        records = list(load_records(str(tmp_path)))
        assert sorted(records) == ["alpha", "beta"]

    def test_csv_without_datasets_raises(self, tmp_path, monkeypatch):
        # Force the fallback path so csv raises DataError regardless of environment.
        monkeypatch.setitem(sys.modules, "datasets", None)
        path = tmp_path / "data.csv"
        path.write_text("text\nhello\n", encoding="utf-8")
        with pytest.raises(DataError, match="datasets"):
            list(load_records(str(path), format="csv"))


class TestBuildDataLoaderPretrain:
    def test_pretrain_full_text(self, tokenizer, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("hello world\nfoo bar\n", encoding="utf-8")
        loader = build_text_dataloader(
            tokenizer,
            DataConfig(source=str(path), mode="pretrain", max_length=16, add_eos=True),
            batch_size=2,
        )
        batch = next(iter(loader))
        assert batch["input_ids"].shape[0] == 2
        assert batch["attention_mask"].shape == batch["input_ids"].shape
        assert batch["labels"].shape == batch["input_ids"].shape
        # Loss covers every non-pad token.
        non_pad = batch["attention_mask"].bool()
        assert torch.equal(batch["labels"][non_pad], batch["input_ids"][non_pad])
        # Padding tokens are masked from the loss.
        pad = ~non_pad
        assert (batch["labels"][pad] == -100).all()

    def test_pretrain_num_samples_cap(self, tokenizer, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("hello world\nfoo bar\nuser hello\n", encoding="utf-8")
        loader = build_text_dataloader(
            tokenizer,
            DataConfig(source=str(path), mode="pretrain", num_samples=1),
            batch_size=4,
        )
        assert len(loader.dataset) == 1

    def test_pretrain_text_fields_subset(self, tokenizer, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text(
            json.dumps({"instruction": "instruction", "output": "output foo bar"}) + "\n",
            encoding="utf-8",
        )
        # Keep only the assistant (output) content.
        loader = build_text_dataloader(
            tokenizer,
            DataConfig(
                source=str(path), mode="pretrain", format="alpaca", text_fields=["assistant"]
            ),
            batch_size=1,
        )
        batch = next(iter(loader))
        ids = batch["input_ids"][0].tolist()
        foo_id = tokenizer.convert_tokens_to_ids("foo")
        bar_id = tokenizer.convert_tokens_to_ids("bar")
        instr_id = tokenizer.convert_tokens_to_ids("instruction")
        # Only the assistant (output) content is kept; the instruction is dropped.
        assert foo_id in ids
        assert bar_id in ids
        assert instr_id not in ids


class TestBuildDataLoaderSFT:
    def test_sft_masks_prompt(self, tokenizer, tmp_path):
        path = tmp_path / "data.jsonl"
        record = {
            "messages": [
                {"role": "user", "content": "hello world"},
                {"role": "assistant", "content": "foo bar"},
            ]
        }
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        loader = build_text_dataloader(
            tokenizer,
            DataConfig(source=str(path), mode="sft", format="chat", max_length=32),
            batch_size=1,
        )
        batch = next(iter(loader))
        input_ids = batch["input_ids"][0].tolist()
        labels = batch["labels"][0].tolist()

        user_word_id = tokenizer.convert_tokens_to_ids("user")
        hello_id = tokenizer.convert_tokens_to_ids("hello")
        foo_id = tokenizer.convert_tokens_to_ids("foo")
        bar_id = tokenizer.convert_tokens_to_ids("bar")

        # Prompt tokens are masked.
        assert labels[input_ids.index(user_word_id)] == -100
        assert labels[input_ids.index(hello_id)] == -100
        # Assistant tokens carry the loss.
        assert labels[input_ids.index(foo_id)] == foo_id
        assert labels[input_ids.index(bar_id)] == bar_id

    def test_sft_without_chat_template_raises(self, tmp_path):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        tok = Tokenizer(WordLevel({"[PAD]": 0, "[EOS]": 1, "[UNK]": 2}, unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        no_template = PreTrainedTokenizerFast(
            tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
        )
        path = tmp_path / "data.txt"
        path.write_text("hello world\n", encoding="utf-8")
        with pytest.raises(DataError, match="chat_template"):
            build_text_dataloader(no_template, DataConfig(source=str(path), mode="sft"))

    def test_sft_prefix_fallback_without_generation_block(self, tmp_path):
        # ChatML-style template WITHOUT the {% generation %} block, like real
        # SmolLM2-Instruct / Llama-3 templates. The prefix-comparison fallback
        # must still mask the prompt and keep only the assistant turn.
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        words = [
            "[PAD]",
            "[EOS]",
            "[UNK]",
            "user",
            "assistant",
            "end",
            "hello",
            "world",
            "foo",
            "bar",
        ]
        vocab = {w: i for i, w in enumerate(words)}
        tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        chat_tok = PreTrainedTokenizerFast(
            tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
        )
        chat_tok.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] + ' ' + message['content'] + ' end ' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
        )

        record = {
            "messages": [
                {"role": "user", "content": "hello world"},
                {"role": "assistant", "content": "foo bar"},
            ]
        }
        path = tmp_path / "data.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        loader = build_text_dataloader(
            chat_tok,
            DataConfig(source=str(path), mode="sft", format="chat", max_length=32),
            batch_size=1,
        )
        batch = next(iter(loader))
        input_ids = batch["input_ids"][0].tolist()
        labels = batch["labels"][0].tolist()

        user_word_id = chat_tok.convert_tokens_to_ids("user")
        hello_id = chat_tok.convert_tokens_to_ids("hello")
        foo_id = chat_tok.convert_tokens_to_ids("foo")
        bar_id = chat_tok.convert_tokens_to_ids("bar")

        # Prompt (user turn + assistant role header) is masked.
        assert labels[input_ids.index(user_word_id)] == -100
        assert labels[input_ids.index(hello_id)] == -100
        # Assistant content carries the loss.
        assert labels[input_ids.index(foo_id)] == foo_id
        assert labels[input_ids.index(bar_id)] == bar_id


class TestConfigErrors:
    def test_unknown_mode_raises(self, tokenizer, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("hello\n", encoding="utf-8")
        with pytest.raises(DataError, match="Unknown mode"):
            build_text_dataloader(
                tokenizer, DataConfig(source=str(path), mode="invalid"), batch_size=1
            )

    def test_empty_source_raises(self, tokenizer):
        with pytest.raises(DataError, match="source"):
            build_text_dataloader(tokenizer, None)

    def test_no_records_raises(self, tokenizer, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")
        with pytest.raises(DataError, match="No records"):
            build_text_dataloader(tokenizer, DataConfig(source=str(path), mode="pretrain"))
