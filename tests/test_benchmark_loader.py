"""Native chat tokenization must preserve special tokens and disable reasoning."""

from vivasecuris.aiasylum.models.benchmark_loader import prepare_benchmark_inputs


class Inputs(dict):
    def to(self, device):
        self['device'] = device
        return self


def test_mistral_native_template_without_jinja_is_used():
    class MistralCommonBackend:
        chat_template = None

        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'q'}]
            assert kwargs['tokenize'] is True
            assert kwargs['return_dict'] is True
            return Inputs(input_ids=[1, 42, 2])

        def __call__(self, *args, **kwargs):
            raise AssertionError('Do not round-trip native control tokens through text')

    assert prepare_benchmark_inputs(MistralCommonBackend(), 'q', 's', None, 'cuda')['input_ids'] == [1, 42, 2]


def test_hybrid_reasoning_template_is_disabled():
    """A Jinja template is rendered to text (reasoning off) and tokenized once, specials off."""
    class HybridTokenizer:
        chat_template = '{% if enable_thinking %}...{% endif %}'

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['enable_thinking'] is False
            assert kwargs['tokenize'] is False          # rendered here; tokenized in __call__
            return 'rendered'

        def __call__(self, text, **kwargs):
            assert text == 'rendered'
            assert kwargs['add_special_tokens'] is False  # the template carries its own specials
            return Inputs(input_ids=[1, 2])

    assert prepare_benchmark_inputs(HybridTokenizer(), 'q', None, None, 'cpu')['device'] == 'cpu'
