"""CUDA regression/gradient checks of the isolated exact-gradient prototype."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import warp_patch_ncc as legacy
from research import ncc_exact_backend as corrected
from research.ncc_reference import score, warped_patches


def run_case(contrast):
    device = "cuda"
    iy, ix = torch.meshgrid(torch.arange(96, device=device), torch.arange(96, device=device), indexing="ij")
    image_r = (contrast * (torch.sin(ix * .37 + iy * .13) + .3 * torch.cos(ix * .21 - iy * .19)) + contrast * 2).float()
    image_n = (contrast * (.8 * torch.sin(ix * .37 + iy * .13 + .3) + .24 * torch.cos(ix * .21 - iy * .19 - .2)) + contrast * 2).float()
    uv = torch.stack((ix[8:88:4, 8:88:4].flatten(), iy[8:88:4, 8:88:4].flatten()), -1).int()
    d = torch.full((len(uv),), 1.2, device=device)
    n = torch.tensor([.1, .03, -1.], device=device).expand(len(uv), 3).contiguous()
    rotation = torch.eye(3, device=device)
    translation = torch.tensor([.03, .007, .002], device=device)
    intrinsics = (80., 80., 47.5, 47.5)

    def call(module):
        dd, nn = d.clone().requires_grad_(), n.clone().requires_grad_()
        s, valid = module.warp_patch_ncc(dd, nn, uv, rotation.T.contiguous(), translation, image_r, image_n, *intrinsics, *intrinsics, False)
        gd, gn = torch.autograd.grad(s.sum(), (dd, nn))
        return s.detach(), valid, gd, gn

    ls, lv, ld, ln = call(legacy)
    cs, cv, cd, cn = call(corrected)
    dd, nn = d.double().requires_grad_(), n.double().requires_grad_()
    xx, yy, valid = warped_patches(dd, nn, uv, rotation.double(), translation.double(), image_r.double(), image_n.double(), intrinsics, intrinsics)
    oracle = score(xx, yy)
    od, on = torch.autograd.grad(oracle.sum(), (dd, nn))
    common = lv & cv & valid
    assert int(common.sum()) > 300, "Synthetic fixture should contain valid, nonconstant patches."
    assert torch.equal(lv, cv), "A backward-only correction must preserve validity."
    assert torch.equal(ls, cs), "A backward-only correction must preserve forward values bit for bit."
    legacy_error = (ld.double() - od).abs()[common]
    corrected_error = (cd.double() - od).abs()[common]
    if contrast < .01:
        assert corrected_error.median() < legacy_error.median() / 10
    step = 1e-7
    with torch.no_grad():
        def at(offset):
            x, y, _ = warped_patches(dd.detach() + offset, nn.detach(), uv, rotation.double(), translation.double(), image_r.double(), image_n.double(), intrinsics, intrinsics)
            return score(x, y)
        fd = (at(step) - at(-step)) / (2 * step)
    return {
        "contrast": contrast, "valid_patches": int(common.sum()), "forward_bitwise_equal": True,
        "depth_gradient_median_abs_error_legacy": float(legacy_error.median()),
        "depth_gradient_median_abs_error_corrected": float(corrected_error.median()),
        "normal_gradient_median_l2_error_legacy": float((ln.double() - on).norm(dim=-1)[common].median()),
        "normal_gradient_median_l2_error_corrected": float((cn.double() - on).norm(dim=-1)[common].median()),
        "oracle_fd_depth_median_abs_error": float((fd - od).abs()[common].median()),
        "scope": "Synthetic warp gradients; not DTU mesh or image quality.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "research/results/ncc_backend_validation.json")
    args = parser.parse_args()
    corrected.load_backend()
    result = {"cases": [run_case(c) for c in (.004, .1)], "torch": torch.__version__,
              "device": torch.cuda.get_device_name(), "all_passed": True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
