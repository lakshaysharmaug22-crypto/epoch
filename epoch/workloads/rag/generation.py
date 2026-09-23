"""Answer generation backends.

extractive  CPU-only span reader (no model) — lets the whole pipeline run anywhere
hf          transformers causal LM; quantisation fp16 | int8 | nf4 (bitsandbytes); runtime eager | compile
ort         ONNX Runtime via optimum (CUDA EP when available)

Loaded models are LRU-cached (size 1 on GPU) so consecutive trials that share a model don't reload it;
load time is reported as build_s, never as request latency.
"""

from __future__ import annotations

import gc
import re
import threading
from collections import OrderedDict

from epoch.benchmark.metrics import normalize_answer
from epoch.workloads.rag.retrieval import Chunk

MODELS = {
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
}
PROMPTS = {
    "concise": "Answer the question using only the context. Reply with the exact value only — no sentence, no explanation.",
    "cite": "Answer the question using only the context. Reply with the exact value only, then the chunk id in square brackets.",
}
_LOCK = threading.Lock()
_MODEL_CACHE: OrderedDict = OrderedDict()


def build_context(chunks: list[Chunk], budget_words: int) -> str:
    parts, used = [], 0
    for c in chunks:
        words = c.text.split()
        if used + len(words) > budget_words and parts:
            break
        parts.append(f"[{c.id}] " + " ".join(words[: max(0, budget_words - used)]))
        used += len(words)
    return "\n".join(parts)


class Extractive:
    """Picks the sentence with the highest question overlap and returns the span after ' is '."""

    def answer(self, question: str, chunks: list[Chunk], budget_words: int) -> str:
        q = set(normalize_answer(question).split()) - {"what", "is", "the", "at", "does", "how", "of"}
        best, best_s = "", -1.0
        for c in chunks:
            for s in re.split(r"(?<=\.)\s+", c.text):
                toks = set(normalize_answer(s).split())
                sc = len(q & toks) / (len(toks) ** 0.3 + 1)
                if " is " in s and sc > best_s:
                    best, best_s = s, sc
        m = re.search(r"\bis (.+?)\.?$", best)
        return m.group(1) if m else best


def _dtype_kw(torch_dtype):
    import transformers
    from packaging.version import Version

    return {"dtype": torch_dtype} if Version(transformers.__version__) >= Version("4.56") else {"torch_dtype": torch_dtype}


def _load(name: str, quant: str, runtime: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mid = MODELS[name]
    tok = AutoTokenizer.from_pretrained(mid, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if runtime == "ort":
        from optimum.onnxruntime import ORTModelForCausalLM

        provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        model = ORTModelForCausalLM.from_pretrained(mid, export=True, provider=provider, use_cache=True)
        return tok, model
    kw = {}
    if quant in ("int8", "nf4"):
        if device != "cuda":
            raise RuntimeError(f"bitsandbytes {quant} not available on {device}")
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = (
            BitsAndBytesConfig(load_in_8bit=True) if quant == "int8" else
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
        )
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(mid, device_map=device, **_dtype_kw(dtype), **kw)
    model.eval()
    if runtime == "compile":
        model.generation_config.cache_implementation = "static"
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
    return tok, model


def _free() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class HFGenerator:
    def __init__(self, name: str, quant: str, runtime: str, max_new_tokens: int, prompt: str, device: str):
        key = (name, quant, runtime)
        with _LOCK:
            hit = _MODEL_CACHE.get(key)
            if hit is None:
                while _MODEL_CACHE:  # one resident model at a time on a single GPU
                    _MODEL_CACHE.popitem(last=False)
                    _free()
                hit = _load(name, quant, runtime, device)
                _MODEL_CACHE[key] = hit
        self.tok, self.model = hit
        self.max_new_tokens = max_new_tokens
        self.system = PROMPTS[prompt]
        self.device = device

    def answer_batch(self, questions: list[str], contexts: list[str]) -> list[str]:
        import torch

        msgs = [[{"role": "system", "content": self.system},
                 {"role": "user", "content": f"Context:\n{c}\n\nQuestion: {q}"}] for q, c in zip(questions, contexts, strict=True)]
        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        enc = self.tok(texts, return_tensors="pt", padding=True)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1] :]
        answers = self.tok.batch_decode(gen, skip_special_tokens=True)
        return [re.sub(r"\[[^\]]*\]", "", a).strip().split("\n")[0] for a in answers]
