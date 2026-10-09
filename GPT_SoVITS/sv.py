import os

import torch

from GPT_SoVITS.eres2net.ERes2NetV2 import ERes2NetV2
from GPT_SoVITS.eres2net import kaldi as Kaldi

# Relative to the pretrained models folder (the local copy of huggingface.co/lj1995/GPT-SoVITS)
SV_MODEL_RELATIVE_PATH = os.path.join("sv", "pretrained_eres2netv2w24s4ep4.ckpt")


class SV:
    """Speaker verification embedding used by the v2Pro / v2ProPlus SoVITS models."""

    def __init__(self, device, is_half, sv_path):
        if not os.path.exists(sv_path):
            raise FileNotFoundError(
                f"Speaker verification model not found at {sv_path}. v2Pro/v2ProPlus models need "
                f"{SV_MODEL_RELATIVE_PATH} from https://huggingface.co/lj1995/GPT-SoVITS in the pretrained models folder."
            )
        pretrained_state = torch.load(sv_path, map_location="cpu", weights_only=True)
        embedding_model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
        embedding_model.load_state_dict(pretrained_state)
        embedding_model.eval()
        if is_half:
            embedding_model = embedding_model.half()
        self.embedding_model = embedding_model.to(device)
        self.is_half = is_half

    def compute_embedding3(self, wav):
        """wav: (batch, samples) at 16 kHz in [-1, 1]. Returns (batch, 20480)."""
        with torch.no_grad():
            if self.is_half:
                wav = wav.half()
            feat = torch.stack(
                [Kaldi.fbank(wav0.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for wav0 in wav]
            )
            sv_emb = self.embedding_model.forward3(feat)
        return sv_emb
