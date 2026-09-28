"""Apply an evaluator's selected generation policy to every analysis subtask."""

from copy import deepcopy


class ConfiguredEvaluatorModel:
    """Run-local adapter; never mutate or share the provider's generation state."""

    def __init__(self, model, *, enable_cot=False, temperature=None, top_p=None, max_tokens=None):
        self.wrapped = model
        self.framework_cot_enabled = bool(enable_cot)
        self.options = {key: value for key, value in {
            "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
        }.items() if value is not None}
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    async def generate(self, prompt="", system_prompt=None, messages=None, **kwargs):
        from vivasecuris.aiasylum.cot import ReACTReasoner

        options = {**kwargs, **self.options}
        chat = deepcopy(messages or [])
        if system_prompt:
            chat.insert(0, {"role": "system", "content": system_prompt})
        if prompt:
            chat.append({"role": "user", "content": prompt})
        if not chat or chat[-1].get("role") != "user":
            raise ValueError("An evaluation request must end with a user task")
        if self.framework_cot_enabled:
            options.setdefault("temperature", getattr(self.wrapped, "temperature", 0.7))
            response = await ReACTReasoner(self.wrapped).reason(
                prompt=chat[-1]["content"], messages=chat[:-1], gen_overrides=options,
            )
        else:
            response = await self.wrapped.generate(prompt="", messages=chat, **options)
        response.metadata = dict(response.metadata or {})
        response.metadata.setdefault("request_system_prompts", [m["content"] for m in chat if m.get("role") == "system"])
        response.metadata.setdefault("request_system_prompts_source", "model_input")
        self.calls.append({"enable_cot": self.framework_cot_enabled, "generation": dict(options),
                           "finish_reason": response.finish_reason,
                           "metadata": deepcopy(response.metadata)})
        return response

    async def reason(self, prompt, messages=None, system_prompt=None, **kwargs):
        # Factuality's per-claim verifier can reuse this wrapper directly;
        # wrapping it in another ReACTReasoner would add the instructions twice.
        return await self.generate(prompt, system_prompt=system_prompt, messages=messages, **kwargs)
