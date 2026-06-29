from kornia.feature.lightglue import LightGlue
from torch import nn
import torch
import os

# ================================================================
# Monkey-patch 1: fix apply_cached_rotary_emb broadcasting bug
# ================================================================
# Problem: posenc produces (2, 1, B, N, D), but q/k have (B, H, N, D).
# freqs[0] = (1, B, N, D) * q (B, H, N, D) → (B, B, N, D)  ← WRONG!
# Fix: permute freqs from (2, 1, B, N, D) to (2, B, 1, N, D)
# so freqs[0] = (B, 1, N, D) * q (B, H, N, D) → (B, H, N, D)  ← CORRECT
import kornia.feature.lightglue as _kg_lg

_orig_apply_cached_rotary_emb = _kg_lg.apply_cached_rotary_emb


def _patched_apply_cached_rotary_emb(freqs, t):
    # freqs: (2, 1, B, N, D) from LearnableFourierPositionalEncoding
    # t:     (B, H, N, D) from SelfBlock
    # When B > 1, the batch dims don't align → broadcasting creates (B, B, N, D)
    if freqs.ndim >= 3 and freqs.shape[2] == t.shape[0] and freqs.shape[2] > 1:
        freqs = freqs.permute(0, 2, 1, 3, 4)  # (2, 1, B, N, D) → (2, B, 1, N, D)
    return _orig_apply_cached_rotary_emb(freqs, t)


_kg_lg.apply_cached_rotary_emb = _patched_apply_cached_rotary_emb


# ================================================================
# Monkey-patch 2: TransformerLayer.masked_forward — add heads dim
# ================================================================
# Problem: masks have shape (B, N, N) but attention scores have shape
# (B, H, N, N). When B > 1, (8, 570, 570) cannot broadcast to
# (8, 1, 570, 570) because dim 1 sizes are 570 vs 1.
# Fix: unsqueeze(1) to add the heads dimension → (B, 1, N, N).

_orig_masked_forward = _kg_lg.TransformerLayer.masked_forward


def _patched_masked_forward(self, desc0, desc1, encoding0, encoding1, mask0, mask1):
    mask = (mask0 & mask1.transpose(-1, -2)).unsqueeze(1)   # (B, 1, N0, N1)
    mask0 = (mask0 & mask0.transpose(-1, -2)).unsqueeze(1)  # (B, 1, N0, N0)
    mask1 = (mask1 & mask1.transpose(-1, -2)).unsqueeze(1)  # (B, 1, N1, N1)
    desc0 = self.self_attn(desc0, encoding0, mask0)
    desc1 = self.self_attn(desc1, encoding1, mask1)
    return self.cross_attn(desc0, desc1, mask)


_kg_lg.TransformerLayer.masked_forward = _patched_masked_forward


# ================================================================
# Monkey-patch 3: _forward to accept external masks + disable
# point pruning when masks are used
# ================================================================
# We patch LightGlue._forward to:
#   (a) check data["image0"]["mask"] and data["image1"]["mask"]
#   (b) if present, use them directly as attention masks
#   (c) disable point pruning when external masks are active
#       (point pruning doesn't update masks, causing shape mismatches)

from kornia.core.check import KORNIA_CHECK  # noqa: E402
from kornia.feature.laf import laf_to_three_points, scale_laf  # noqa: E402
from kornia.feature.lightglue import (  # noqa: E402
    filter_matches, normalize_keypoints, pad_to_length,
)

_orig_lg_forward = _kg_lg.LightGlue._forward


def _patched_lg_forward(self, data: dict) -> dict:
    """Patched _forward: supports external masks in data['image0']['mask']."""
    for key in self.required_data_keys:
        KORNIA_CHECK(key in data, f"Missing key {key} in data")
    data0, data1 = data["image0"], data["image1"]
    kpts0, kpts1 = data0["keypoints"], data1["keypoints"]
    b, m, _ = kpts0.shape
    b, n, _ = kpts1.shape
    device = kpts0.device
    size0, size1 = data0.get("image_size"), data1.get("image_size")
    size0 = size0 if size0 is not None else data0["image"].shape[-2:][::-1]
    size1 = size1 if size1 is not None else data1["image"].shape[-2:][::-1]

    kpts0 = normalize_keypoints(kpts0, size0).clone()
    kpts1 = normalize_keypoints(kpts1, size1).clone()
    KORNIA_CHECK(torch.all(kpts0 >= -1).item() and torch.all(kpts0 <= 1).item(), "")
    KORNIA_CHECK(torch.all(kpts1 >= -1).item() and torch.all(kpts1 <= 1).item(), "")
    if self.conf.add_scale_ori:
        kpts0 = torch.cat([kpts0] + [data0[k].unsqueeze(-1) for k in ("scales", "oris")], -1)
        if self.conf.scale_coef != 1.0:
            kpts0[..., -2] = kpts0[..., -2] * self.conf.scale_coef
        kpts1 = torch.cat([kpts1] + [data1[k].unsqueeze(-1) for k in ("scales", "oris")], -1)
        if self.conf.scale_coef != 1.0:
            kpts1[..., -2] = kpts1[..., -2] * self.conf.scale_coef
    elif self.conf.add_laf:
        laf0 = scale_laf(data0["lafs"], self.conf.scale_coef)
        laf1 = scale_laf(data1["lafs"], self.conf.scale_coef)
        laf0 = laf_to_three_points(laf0)
        laf1 = laf_to_three_points(laf1)
        kpts0 = torch.cat(
            [kpts0,
             normalize_keypoints(laf0[..., 0], size0).clone().to(kpts0.dtype),
             normalize_keypoints(laf0[..., 1], size0).clone().to(kpts0.dtype)], -1)
        kpts1 = torch.cat(
            [kpts1,
             normalize_keypoints(laf1[..., 0], size1).clone().to(kpts1.dtype),
             normalize_keypoints(laf1[..., 1], size1).clone().to(kpts1.dtype)], -1)

    desc0 = data0["descriptors"].detach().contiguous()
    desc1 = data1["descriptors"].detach().contiguous()

    KORNIA_CHECK(desc0.shape[-1] == self.conf.input_dim, "Descriptor dimension does not match input dim in config")
    KORNIA_CHECK(desc1.shape[-1] == self.conf.input_dim, "Descriptor dimension does not match input dim in config")

    if torch.is_autocast_enabled():
        desc0 = desc0.half()
        desc1 = desc1.half()

    # ================================================================
    # PATCHED: accept external masks from data dict
    # ================================================================
    mask0, mask1 = None, None
    has_ext_mask = ("mask" in data0) and ("mask" in data1)
    if has_ext_mask:
        mask0 = data0.pop("mask")
        mask1 = data1.pop("mask")
        do_compile = False
    else:
        c = max(m, n)
        do_compile = self.static_lengths and c <= max(self.static_lengths)
        if do_compile:
            kn = min([k for k in self.static_lengths if k >= c])
            desc0, mask0 = pad_to_length(desc0, kn)
            desc1, mask1 = pad_to_length(desc1, kn)
            kpts0, _ = pad_to_length(kpts0, kn)
            kpts1, _ = pad_to_length(kpts1, kn)

    desc0 = self.input_proj(desc0)
    desc1 = self.input_proj(desc1)
    encoding0 = self.posenc(kpts0)
    encoding1 = self.posenc(kpts1)

    do_early_stop = self.conf.depth_confidence > 0
    # PATCHED: disable point pruning when external masks are used
    do_point_pruning = self.conf.width_confidence > 0 and not do_compile and mask0 is None
    pruning_th = self.pruning_min_kpts(device)
    if do_point_pruning:
        ind0 = torch.arange(0, m, device=device)[None]
        ind1 = torch.arange(0, n, device=device)[None]
        prune0 = torch.ones_like(ind0)
        prune1 = torch.ones_like(ind1)
    token0, token1 = None, None
    for i in range(self.conf.n_layers):
        desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1, mask0=mask0, mask1=mask1)
        if i == self.conf.n_layers - 1:
            continue

        if do_early_stop:
            token0, token1 = self.token_confidence[i](desc0, desc1)
            if self.check_if_stop(token0[..., :m, :], token1[..., :n, :], i, m + n):
                break
        if do_point_pruning and desc0.shape[-2] > pruning_th:
            scores0 = self.log_assignment[i].get_matchability(desc0)
            prunemask0 = self.get_pruning_mask(token0, scores0, i)
            keep0 = torch.where(prunemask0)[1]
            ind0 = ind0.index_select(1, keep0)
            desc0 = desc0.index_select(1, keep0)
            encoding0 = encoding0.index_select(-2, keep0)
            prune0[:, ind0] += 1
        if do_point_pruning and desc1.shape[-2] > pruning_th:
            scores1 = self.log_assignment[i].get_matchability(desc1)
            prunemask1 = self.get_pruning_mask(token1, scores1, i)
            keep1 = torch.where(prunemask1)[1]
            ind1 = ind1.index_select(1, keep1)
            desc1 = desc1.index_select(1, keep1)
            encoding1 = encoding1.index_select(-2, keep1)
            prune1[:, ind1] += 1

    desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
    scores, _ = self.log_assignment[i](desc0, desc1)
    m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)
    matches, mscores = [], []
    for k in range(b):
        valid = m0[k] > -1
        m_indices_0 = torch.where(valid)[0]
        m_indices_1 = m0[k][valid]
        if do_point_pruning:
            m_indices_0 = ind0[k, m_indices_0]
            m_indices_1 = ind1[k, m_indices_1]
        matches.append(torch.stack([m_indices_0, m_indices_1], -1))
        mscores.append(mscores0[k][valid])

    if do_point_pruning:
        m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
        m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
        m0_[:, ind0] = torch.where(m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
        m1_[:, ind1] = torch.where(m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
        mscores0_ = torch.zeros((b, m), device=mscores0.device)
        mscores1_ = torch.zeros((b, n), device=mscores1.device)
        mscores0_[:, ind0] = mscores0
        mscores1_[:, ind1] = mscores1
        m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
    else:
        prune0 = torch.ones_like(mscores0) * self.conf.n_layers
        prune1 = torch.ones_like(mscores1) * self.conf.n_layers

    pred = {
        "log_assignment": scores,
        "matches0": m0,
        "matches1": m1,
        "matching_scores0": mscores0,
        "matching_scores1": mscores1,
        "stop": i + 1,
        "matches": matches,
        "scores": mscores,
        "prune0": prune0,
        "prune1": prune1,
    }
    return pred


_kg_lg.LightGlue._forward = _patched_lg_forward


# ================================================================
# LighterGlue wrapper
# ================================================================

class LighterGlue(nn.Module):
    """
        Lighter version of LightGlue :)
        Patched to support batched (B > 1) inference with masks.
    """

    default_conf_xfeat = {
    "name": "xfeat",  # just for interfacing
    "input_dim": 64,  # input descriptor dimension (autoselected from weights)
    "descriptor_dim": 96,
    "add_scale_ori": False,
    "add_laf": False,  # for KeyNetAffNetHardNet
    "scale_coef": 1.0,  # to compensate for the SIFT scale bigger than KeyNet
    "n_layers": 6,
    "num_heads": 1,
    "flash": True,  # enable FlashAttention if available.
    "mp": False,  # enable mixed precision
    "depth_confidence": -1,  # early stopping, disable with -1
    "width_confidence": 0.95,  # point pruning, disable with -1
    "filter_threshold": 0.1,  # match threshold
    "weights": None,
    }

    def __init__(self, weights = os.path.abspath(os.path.dirname(__file__)) + '/../weights/xfeat-lighterglue.pt'):
        super().__init__()
        LightGlue.default_conf = self.default_conf_xfeat
        self.net = LightGlue(None)
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if os.path.exists(weights):
            state_dict = torch.load(weights, map_location=self.dev)
        else:
            state_dict = torch.hub.load_state_dict_from_url("https://github.com/verlab/accelerated_features/raw/main/weights/xfeat-lighterglue.pt")

        for i in range(self.net.conf.n_layers):
            pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            state_dict = {k.replace('matcher.', ''): v for k, v in state_dict.items()}

        self.net.load_state_dict(state_dict, strict=False)
        self.net.to(self.dev)

    @torch.inference_mode()
    def forward(self, data, min_conf = 0.1):
        self.net.conf.filter_threshold = min_conf

        img0 = {
            'keypoints': data['keypoints0'],
            'descriptors': data['descriptors0'],
            'image_size': data['image_size0'],
        }
        img1 = {
            'keypoints': data['keypoints1'],
            'descriptors': data['descriptors1'],
            'image_size': data['image_size1'],
        }

        # Pass external masks if present (for batch matching with padding)
        if 'mask0' in data:
            img0['mask'] = data['mask0']
        if 'mask1' in data:
            img1['mask'] = data['mask1']

        result = self.net({
            'image0': img0,
            'image1': img1,
        })
        return result