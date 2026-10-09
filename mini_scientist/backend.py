"""Batched local inference with transformers. Works on a single 24 GB GPU
(e.g. ModelScope's free notebook) for models up to 7B in bf16."""
from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .agent import TRUNCATED


def resolve(model_id: str) -> str:
    """Download from ModelScope when USE_MODELSCOPE=1 (default on ModelScope
    notebooks), otherwise let transformers use the Hugging Face cache."""
    if os.environ.get("USE_MODELSCOPE", "0") == "1":
        from modelscope import snapshot_download
        return snapshot_download(model_id)
    return model_id


def split_thinking(text: str, thinking: bool) -> tuple[str, str]:
    """Split a reasoning model's output into (thinking, answer). In thinking mode an
    output without </think> was cut off mid-thought: it has no answer, and must not
    be fed back into the history as if it were one."""
    head, sep, tail = text.rpartition("</think>")
    if sep:
        return head.replace("<think>", "").strip(), tail.strip()
    if thinking:
        return text, TRUNCATED
    return "", text.replace("<think>", "").strip()


class HFBackend:
    def __init__(self, model_id: str, max_new_tokens: int = 512, batch_size: int = 16, thinking: bool = False):
        path = resolve(model_id)
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if thinking:
            torch.manual_seed(0)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16).to("cuda").eval()
        self.thinking = thinking
        self.max_new_tokens = max(max_new_tokens, 8192) if thinking else max_new_tokens
        self.batch_size = batch_size
        self.think_chars = []

    @torch.inference_mode()
    def generate(self, conversations: list[list[dict]]) -> list[str]:
        out = []
        for i in range(0, len(conversations), self.batch_size):
            chunk = conversations[i:i + self.batch_size]
            kw = {"enable_thinking": self.thinking} if self.thinking else {}
            prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **kw) for m in chunk]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
            # greedy for instruct models; Qwen's recommended sampling for thinking (greedy makes it loop)
            sampling = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20) if self.thinking else \
                dict(do_sample=False, temperature=None, top_p=None, top_k=None)
            gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, pad_token_id=self.tok.pad_token_id,
                                      **sampling)
            for text in self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
                think, answer = split_thinking(text, self.thinking)
                self.think_chars.append(len(think))
                out.append(answer)
        return out

    def close(self):
        del self.model
        torch.cuda.empty_cache()


class VLLMBackend:
    """Faster backend for reasoning ("thinking") models such as Qwen3 / Qwen3.5.
    The <think> trace is dropped from the reply that enters the agent's history,
    but its length is kept in `self.think_chars` for analysis."""

    def __init__(self, model_id: str, thinking: bool = True, max_tokens: int = 8192,
                 max_model_len: int = 20480, gpu_memory_utilization: float = 0.85):
        from vllm import LLM, SamplingParams
        self.llm = LLM(resolve(model_id), max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, seed=0)
        self.thinking = thinking
        # Qwen's recommended sampling for thinking mode; greedy decoding makes thinking models loop
        self.params = SamplingParams(temperature=0.6 if thinking else 0.7, top_p=0.95 if thinking else 0.8,
                                     top_k=20, max_tokens=max_tokens if thinking else 1024, seed=0)
        self.think_chars = []

    def generate(self, conversations: list[list[dict]]) -> list[str]:
        outs = self.llm.chat(conversations, self.params, use_tqdm=False,
                             chat_template_kwargs={"enable_thinking": self.thinking})
        replies = []
        for o in outs:
            think, answer = split_thinking(o.outputs[0].text, self.thinking)
            self.think_chars.append(len(think))
            replies.append(answer)
        return replies

    def close(self):
        del self.llm
