"""Chat prompt formatting: Qwen3.5's ChatML written out by hand (without tool calling), or a model's own Jinja
chat template read from its files at load time (TemplateChat; Tiny Aya)."""
import json
from datetime import datetime

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
SPECIAL_TOKENS = ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token")


def _raise(message: str):
    raise ValueError(message)                          # a template's raise_exception(...): e.g. roles out of order


class TemplateChat:
    """A model's own chat template, read from tokenizer_config.json and rendered as transformers'
    apply_chat_template renders it: a sandboxed Jinja environment with trim_blocks, lstrip_blocks and loop controls,
    the same tojson / raise_exception / strftime_now helpers, and the special-token strings as variables. Tiny Aya's
    template carries a long fixed system preamble; it stays in the model's files and is never copied into this code."""

    def __init__(self, tokenizer_config: str, name: str = "default"):
        cfg = json.load(open(tokenizer_config, encoding="utf-8"))
        template = cfg["chat_template"]
        if isinstance(template, list):                 # several named templates: [{"name": ..., "template": ...}]
            template = next(t["template"] for t in template if t["name"] == name)
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
        env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
            x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
        env.globals["raise_exception"] = _raise
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self.template = env.from_string(template)
        self.tokens = {k: v["content"] if isinstance(v, dict) else v for k, v in cfg.items()
                       if k in SPECIAL_TOKENS and v is not None}

    def render(self, messages: list[dict], add_generation_prompt: bool = True, enable_thinking: bool = False) -> str:
        return self.template.render(messages=messages, tools=None, documents=None,
                                    add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking,
                                    **self.tokens)


def format_chat(messages: list[dict], add_generation_prompt: bool = True, style: "str | TemplateChat" = "qwen3.5",
                enable_thinking: bool = False) -> str:
    """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...] -> one prompt string.

    style: "qwen3.5", or a TemplateChat (the model's own template, rendered as transformers renders it).

    qwen3.5 (mirrors chat_template.jinja): no default system prompt; every message is trimmed; an assistant turn's
    text before its last </think> is reasoning (the model's own replies have no opening <think> because the prompt
    already opened it); reasoning is kept only for assistant turns after the last user query; the generation prompt
    opens a thinking block (enable_thinking) or closes an empty one. Any other style raises.
    """
    if isinstance(style, TemplateChat):
        return style.render(messages, add_generation_prompt, enable_thinking)
    if style != "qwen3.5":
        raise ValueError(f"unknown chat style {style!r}")
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
    out = []
    for i, m in enumerate(messages):
        content = m["content"].strip()
        if m["role"] == "assistant":
            reasoning = ""
            if "</think>" in content:
                reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n").strip()
                content = content.split("</think>")[-1].lstrip("\n")
            if i > last_user:
                out.append(f"{IM_START}assistant\n<think>\n{reasoning}\n</think>\n\n{content}{IM_END}\n")
            else:
                out.append(f"{IM_START}assistant\n{content}{IM_END}\n")
        else:
            out.append(f"{IM_START}{m['role']}\n{content}{IM_END}\n")
    if add_generation_prompt:
        out.append(f"{IM_START}assistant\n" + ("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"))
    return "".join(out)
