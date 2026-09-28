"""A random Qwen2 small enough to train in seconds, with a 64-word tokenizer.

Same shape as ``test_interp_compatibility.build_model`` (vocab 64, hidden 32,
two layers). The WordLevel vocabulary covers every id so a greedy sample from
random weights always decodes to text rather than to nothing.
"""

from pathlib import Path
from typing import List

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from vivasecuris.aiasylum.weights.lora import LoraSpec
from vivasecuris.aiasylum.weights.train_data import TrainRow

WORDS = ["[UNK]", "[PAD]", "[EOS]"] + [f"w{i}" for i in range(61)]


def build_tokenizer(words=None, with_bos: bool = False):
    """The 64-word tokenizer; ``with_bos`` makes it insert a BOS token like a real one.

    The default has no BOS at all, which is why the BOS defect in prompt
    formatting was invisible to every fixture: ``add_special_tokens`` on or off
    gave identical ids. ``with_bos`` adds ``[BOS]`` (id 3, vocab still 64) and a
    post-processor that prepends it, so a test can count how many BOS tokens a
    capture or generation path actually put in front of the prompt.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    if words is None:
        words = (["[UNK]", "[PAD]", "[EOS]", "[BOS]"] + [f"w{i}" for i in range(60)]) if with_bos else list(WORDS)
    else:
        words = list(words)
    backend = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    extra = {}
    if with_bos:
        from tokenizers.processors import TemplateProcessing

        bos_id = words.index("[BOS]")
        backend.post_processor = TemplateProcessing(
            single="[BOS] $A", pair="[BOS] $A $B", special_tokens=[("[BOS]", bos_id)]
        )
        extra["bos_token"] = "[BOS]"
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        model_input_names=["input_ids", "attention_mask"], **extra,
    )


def build_model(seed: int = 17, family: str = "qwen2", **config_overrides):
    """Random tiny causal LM. ``qwen2_moe`` gives two experts per layer."""
    if family == "qwen2":
        cls = transformers.Qwen2Config
    elif family == "qwen2_moe":
        cls = getattr(transformers, "Qwen2MoeConfig", None)
        if cls is None:
            pytest.skip("Qwen2-MoE is unavailable in this transformers version")
    else:
        raise ValueError(family)
    options = dict(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
        bos_token_id=2, eos_token_id=2, pad_token_id=1, attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    if family == "qwen2_moe":
        options.update(num_experts=2, num_experts_per_tok=2, moe_intermediate_size=32,
                       shared_expert_intermediate_size=32)
    options.update(config_overrides)
    torch.manual_seed(seed)
    return transformers.AutoModelForCausalLM.from_config(cls(**options)).eval()


def save_tiny(directory, seed: int = 17) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    build_model(seed).save_pretrained(str(directory))
    build_tokenizer().save_pretrained(str(directory))
    return directory


def sample_rows(n: int = 4) -> List[TrainRow]:
    return [
        TrainRow(prompt=f"w{3 + i} w{4 + i}", response=f"w{10 + i} w{11 + i} w{12 + i}")
        for i in range(n)
    ]


def small_spec(**overrides) -> LoraSpec:
    base = dict(rank=2, alpha=4, max_steps=2, batch_size=1, grad_accum=1,
                eval_rows=1, eval_every=1, max_length=32, seed=0)
    base.update(overrides)
    return LoraSpec(**base)
