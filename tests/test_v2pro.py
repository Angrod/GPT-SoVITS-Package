"""v2Pro/v2ProPlus inference checks with random weights (no downloads, CPU only).

Run from the repository root:  python tests/test_v2pro.py
"""
import io
import json
import os
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch


PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(PACKAGE, "GPT_SoVITS", "configs")
sys.path.insert(0, PACKAGE)

import torchaudio  # noqa: E402

import GPT_SoVITS.TTS_infer_pack.TTS as tts_module  # noqa: E402
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402
from GPT_SoVITS.eres2net.ERes2NetV2 import ERes2NetV2  # noqa: E402
from GPT_SoVITS.module.models import SynthesizerTrn  # noqa: E402
from GPT_SoVITS.utils import HParams  # noqa: E402


# torchaudio >= 2.9 needs torchcodec for load(); the app pins torchaudio 2.7, so read with soundfile here
def _load(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _load
tts_module.snapshot_download = lambda **kwargs: None  # no network
torch.set_grad_enabled(False)
failures = []


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        failures.append(name)


def save_sovits(path, config_name, version, header):
    """Writes a checkpoint the way upstream's process_ckpt.savee/my_save2 does."""
    with open(os.path.join(CONFIGS, config_name)) as f:
        hps_dict = json.load(f)
    kwargs = dict(hps_dict["model"], version=version, semantic_frame_rate="25hz")
    model = SynthesizerTrn(hps_dict["data"]["filter_length"] // 2 + 1,
                           hps_dict["train"]["segment_size"] // hps_dict["data"]["hop_length"],
                           n_speakers=hps_dict["data"]["n_speakers"], **kwargs)
    weights = {k: v.half() for k, v in model.state_dict().items() if "enc_q" not in k}
    opt = {"weight": weights, "config": HParams(**hps_dict), "info": "test"}
    bio = io.BytesIO()
    torch.save(opt, bio)
    data = bio.getvalue()
    with open(path, "wb") as f:
        f.write(header + data[2:])


def make_tts(pretrained_dir):
    cfg = TTS_Config({"version": "v2", "custom": {
        "device": "cpu", "is_half": False, "version": "v2",
        "t2s_weights_path": "", "vits_weights_path": "",
        "bert_base_path": os.path.join(pretrained_dir, "chinese-roberta-wwm-ext-large"),
        "cnhuhbert_base_path": os.path.join(pretrained_dir, "chinese-hubert-base"),
    }})
    # Same state TTS.__init__ sets up, without loading BERT/HuBERT/GPT weights
    tts = TTS.__new__(TTS)
    tts.configs = cfg
    tts.vits_model = None
    tts.sv_model = None
    tts.is_v2pro = False
    tts.vocoder = None
    tts.prompt_cache = {"ref_audio_path": None, "prompt_semantic": None, "refer_spec": [], "refer_sv_audio": [],
                        "prompt_text": None, "prompt_lang": None, "phones": None, "bert_features": None,
                        "norm_text": None, "aux_ref_audio_paths": []}
    tts.precision = torch.float32
    return tts


with tempfile.TemporaryDirectory() as d:
    pretrained = os.path.join(d, "engines", "gpt_sovits", "pretrained_models")
    os.makedirs(os.path.join(pretrained, "sv"))
    # As after the app's first auto-download; TTS_Config falls back to its own defaults if these are missing
    os.makedirs(os.path.join(pretrained, "chinese-roberta-wwm-ext-large"))
    os.makedirs(os.path.join(pretrained, "chinese-hubert-base"))
    torch.manual_seed(0)
    torch.save(ERes2NetV2(baseWidth=24, scale=4, expansion=4).state_dict(),
               os.path.join(pretrained, "sv", "pretrained_eres2netv2w24s4ep4.ckpt"))
    v2_path, pro_path = os.path.join(d, "voice_v2.pth"), os.path.join(d, "voice_v2ProPlus.pth")
    save_sovits(v2_path, "s2.json", "v2", b"01")
    save_sovits(pro_path, "s2v2ProPlus.json", "v2ProPlus", b"06")
    ref, aux = os.path.join(d, "ref.wav"), os.path.join(d, "aux.wav")
    sf.write(ref, (0.1 * np.sin(np.arange(int(22050 * 3.0)) / 9)).astype(np.float32), 22050)
    sf.write(aux, (0.1 * np.sin(np.arange(int(44100 * 2.0)) / 5)).astype(np.float32), 44100)

    # --- config version names ---
    check("TTS_Config accepts 'v2proplus' in any case",
          TTS_Config({"version": "V2ProPlus", "custom": make_tts(pretrained).configs.update_configs()}).version == "v2ProPlus")
    try:
        TTS_Config({"version": "v9", "custom": {}})
        check("TTS_Config rejects unknown versions", False)
    except AssertionError:
        check("TTS_Config rejects unknown versions", True)
    check("default configs include v2Pro and v2ProPlus", {"v2Pro", "v2ProPlus"} <= set(TTS_Config.default_configs))

    # --- v2 model: unchanged behaviour ---
    tts = make_tts(pretrained)
    tts.init_vits_weights(v2_path)
    check("v2 checkpoint: not v2Pro, no speaker model loaded", not tts.is_v2pro and tts.sv_model is None)
    tts._set_ref_spec(ref)
    check("v2: no speaker embeddings requested", tts._get_sv_embeddings() is None)
    codes, text = torch.randint(0, 1024, (1, 1, 30)), torch.randint(1, 300, (1, 20))
    audio = tts.vits_model.decode(codes, text, [s.float() for s in tts.prompt_cache["refer_spec"]], sv_emb=None)
    check(f"v2: decode works {tuple(audio.shape)}", audio.shape[-1] > 0)

    # --- switching the same pipeline to a v2ProPlus model ---
    tts.init_vits_weights(pro_path)
    check("v2ProPlus checkpoint detected from header byte 06", tts.is_v2pro and tts.configs.version == "v2ProPlus")
    check("v2ProPlus: SynthesizerTrn built with speaker layers", tts.vits_model.is_v2pro
          and tts.vits_model.sv_emb.out_features == 1024)
    check("v2ProPlus: speaker model loaded from <pretrained>/sv", tts.sv_model is not None)
    check("v2ProPlus: languages are the v2 set", "en" in tts.configs.languages and "ko" in tts.configs.languages)

    # Reference cached while the v2 model was loaded must still work (sv audio is always computed)
    embs = tts._get_sv_embeddings()
    check(f"cached reference reusable after switching models {tuple(embs[0].shape)}",
          len(embs) == 1 and embs[0].shape == (1, 20480))

    tts._set_ref_spec(ref)
    sv_audio = tts.prompt_cache["refer_sv_audio"][0]
    check(f"16 kHz reference audio for the speaker model {tuple(sv_audio.shape)}",
          sv_audio.shape[0] == 1 and abs(sv_audio.shape[1] - 16000 * 3) < 200)

    tts._set_aux_ref_audio([aux, os.path.join(d, "missing.wav"), ""])
    check("aux reference added in parallel to refer_spec",
          len(tts.prompt_cache["refer_spec"]) == 2 and len(tts.prompt_cache["refer_sv_audio"]) == 2)
    embs = tts._get_sv_embeddings()
    audio = tts.vits_model.decode(codes, text, [s.float() for s in tts.prompt_cache["refer_spec"]], sv_emb=embs)
    check(f"v2ProPlus: decode with main + aux reference {tuple(audio.shape)}",
          audio.shape[-1] > 0 and torch.isfinite(audio).all())

    tts._set_aux_ref_audio([])
    check("aux references cleared", len(tts.prompt_cache["refer_spec"]) == 1 and len(tts.prompt_cache["refer_sv_audio"]) == 1)

    # --- clear error when the speaker model was not downloaded ---
    os.remove(os.path.join(pretrained, "sv", "pretrained_eres2netv2w24s4ep4.ckpt"))
    tts = make_tts(pretrained)
    try:
        tts.init_vits_weights(pro_path)
        check("missing sv model gives a clear error", False)
    except FileNotFoundError as e:
        check("missing sv model gives a clear error", "pretrained_eres2netv2w24s4ep4.ckpt" in str(e))

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
