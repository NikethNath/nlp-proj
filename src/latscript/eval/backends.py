"""Model backends. Every call reports its own latency and token counts.

* ``NLLB``  - a dedicated MT model run with transformers (CPU or GPU).
* ``Chat``  - any OpenAI-compatible chat endpoint: a local llama.cpp server,
  vLLM on a Colab GPU, or a hosted API (OpenAI, or Gemini's OpenAI-compatible
  endpoint) with the key taken from an environment variable.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests

FLORES_CODE = {"ml": "mal_Mlym", "ta": "tam_Taml"}


@dataclass
class Result:
    text: str
    seconds: float
    prompt_tokens: int
    completion_tokens: int


class NLLB:
    def __init__(self, model_id: str = "facebook/nllb-200-distilled-600M", batch: int = 16,
                 beams: int = 1, adapter: str | None = None, threads: int | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).eval()
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload().eval()
        self.batch, self.beams = batch, beams
        self.eng = self.tok.convert_tokens_to_ids("eng_Latn")

    def translate(self, texts: list[str], lang: str):
        """Yield one Result per text, a batch at a time."""
        self.tok.src_lang = FLORES_CODE[lang]
        for start in range(0, len(texts), self.batch):
            chunk = texts[start:start + self.batch]
            enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256)
            t0 = time.perf_counter()
            with self.torch.inference_mode():
                # Cap runaway loops (common on romanized input): an English
                # translation never needs 1.5x the source's token count.
                cap = min(200, int(1.5 * enc["input_ids"].shape[1]) + 10)
                gen = self.model.generate(**enc, forced_bos_token_id=self.eng,
                                          num_beams=self.beams, max_new_tokens=cap,
                                          max_length=None)
            per = (time.perf_counter() - t0) / len(chunk)
            hyps = self.tok.batch_decode(gen, skip_special_tokens=True)
            for i, h in enumerate(hyps):
                yield Result(h.strip(), per, int(enc["attention_mask"][i].sum()),
                             int((gen[i] != self.tok.pad_token_id).sum()))


class Chat:
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:8091/v1",
                 api_key_env: str | None = None, parallel: int = 4, timeout: int = 600):
        self.model, self.url = model, base_url.rstrip("/") + "/chat/completions"
        key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.parallel, self.timeout = parallel, timeout

    def complete(self, messages: list[dict], max_tokens: int) -> Result:
        body = {"model": self.model, "messages": messages, "temperature": 0,
                "max_tokens": max_tokens, "seed": 0}
        for attempt in range(5):
            t0 = time.perf_counter()
            try:
                r = requests.post(self.url, json=body, headers=self.headers, timeout=self.timeout)
                r.raise_for_status()
                d = r.json()
                usage = d.get("usage", {})
                return Result((d["choices"][0]["message"].get("content") or "").strip(),
                              time.perf_counter() - t0, usage.get("prompt_tokens", 0),
                              usage.get("completion_tokens", 0))
            except (requests.RequestException, KeyError, ValueError):
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    def map(self, jobs: list[list[dict]], max_tokens: int):
        """Yield (index, Result) as calls finish, `parallel` at a time."""
        with ThreadPoolExecutor(self.parallel) as pool:
            futures = {pool.submit(self.complete, m, max_tokens): i for i, m in enumerate(jobs)}
            for f in futures:
                yield futures[f], f.result()
