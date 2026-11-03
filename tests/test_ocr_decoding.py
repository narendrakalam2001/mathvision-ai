"""Tests for the shifted-label greedy decoder (reference vs KV-cache) on a tiny random model.

No pretrained weights are needed: a randomly initialised TrOCR-style
``VisionEncoderDecoderModel`` is enough to prove the cached decoder is
token-for-token identical to the reference implementation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import TrOCRConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel, ViTConfig  # noqa: E402

import src.ocr_engine as ocr_engine  # noqa: E402
from src.ocr_engine import _TrOCRBackend, shifted_greedy_decode  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model() -> VisionEncoderDecoderModel:
    torch.manual_seed(0)
    enc = ViTConfig(image_size=32, patch_size=16, hidden_size=32, num_hidden_layers=2,
                    num_attention_heads=2, intermediate_size=64)
    dec = TrOCRConfig(vocab_size=120, d_model=32, decoder_layers=2, decoder_attention_heads=2,
                      decoder_ffn_dim=64, max_position_embeddings=256, cross_attention_hidden_size=32,
                      use_cache=True)
    return VisionEncoderDecoderModel(VisionEncoderDecoderConfig.from_encoder_decoder_configs(enc, dec)).eval()


def _pixels(seed: int):
    torch.manual_seed(seed)
    return torch.randn(1, 3, 32, 32)


@pytest.mark.parametrize("max_len", [1, 2, 3, 6, 13, 30])
def test_cached_decoder_matches_reference(tiny_model, max_len: int) -> None:
    for seed in range(4):
        px = _pixels(seed)
        ref_ids, ref_probs = shifted_greedy_decode(tiny_model, px, 0, -1, max_len, use_cache=False)
        fast_ids, fast_probs = shifted_greedy_decode(tiny_model, px, 0, -1, max_len, use_cache=True)
        assert fast_ids == ref_ids and len(fast_ids) == max_len
        assert fast_probs == pytest.approx(ref_probs, abs=1e-4)


@pytest.mark.parametrize("position", [0, 1, 2, 5, 8])
def test_stops_at_eos_identically(tiny_model, position: int) -> None:
    """EOS may land on either token of a 2-token chunk; both paths must stop at the same place."""
    px = _pixels(3)
    free_run = shifted_greedy_decode(tiny_model, px, 0, -1, 20, use_cache=False)[0]
    eos = free_run[position]
    ref = shifted_greedy_decode(tiny_model, px, 0, eos, 20, use_cache=False)[0]
    fast = shifted_greedy_decode(tiny_model, px, 0, eos, 20, use_cache=True)[0]
    assert fast == ref and eos not in fast


def test_cached_decoder_uses_about_half_the_decoder_passes(tiny_model) -> None:
    calls = {"n": 0}
    handle = tiny_model.decoder.register_forward_hook(lambda *_: calls.__setitem__("n", calls["n"] + 1))
    try:
        px = _pixels(1)
        shifted_greedy_decode(tiny_model, px, 0, -1, 20, use_cache=False)
        reference_passes = calls["n"]
        calls["n"] = 0
        shifted_greedy_decode(tiny_model, px, 0, -1, 20, use_cache=True)
        assert reference_passes == 20 and calls["n"] <= 11
    finally:
        handle.remove()


class _FakeTokenizer:
    cls_token_id, sep_token_id = 0, 2

    def decode(self, ids, **_kwargs) -> str:
        return " ".join(str(i) for i in ids)


def _backend(tiny_model) -> _TrOCRBackend:
    backend = _TrOCRBackend.__new__(_TrOCRBackend)
    backend._model = tiny_model
    backend._processor = SimpleNamespace(tokenizer=_FakeTokenizer())
    backend._use_cache = True
    return backend


def test_backend_falls_back_when_cache_path_fails(tiny_model, monkeypatch) -> None:
    real = ocr_engine.shifted_greedy_decode

    def flaky(*args, use_cache=True, **kwargs):
        if use_cache:
            raise TypeError("simulated cache API difference")
        return real(*args, use_cache=False, **kwargs)

    monkeypatch.setattr(ocr_engine, "shifted_greedy_decode", flaky)
    backend = _backend(tiny_model)
    text, conf = backend._decode_shifted(_pixels(2), max_len=8)
    assert backend._use_cache is False          # switched off for good
    assert len(text.split()) == 8 and 0.0 < conf <= 1.0


def test_backend_decode_matches_reference_text(tiny_model) -> None:
    fast_backend, slow_backend = _backend(tiny_model), _backend(tiny_model)
    slow_backend._use_cache = False
    px = _pixels(5)
    assert fast_backend._decode_shifted(px, 12)[0] == slow_backend._decode_shifted(px, 12)[0]


def test_precomputed_encoder_states_give_the_same_tokens(tiny_model) -> None:
    px = _pixels(7)
    with torch.inference_mode():
        enc = tiny_model.encoder(pixel_values=px).last_hidden_state
    for use_cache in (False, True):
        direct = shifted_greedy_decode(tiny_model, px, 0, -1, 10, use_cache)[0]
        reused = shifted_greedy_decode(tiny_model, px, 0, -1, 10, use_cache, encoder_hidden=enc)[0]
        assert direct == reused
