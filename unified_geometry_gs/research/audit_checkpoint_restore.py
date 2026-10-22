"""Reproduce stale appearance binding on a real checkpoint and test research fix."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from arguments import OptimizationParams
from argparse import ArgumentParser
from scene import GaussianModel
from research.checkpoint_state import restore_bound,verify_binding,state_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error('Fresh output required.')
    p=ArgumentParser()
    opt_builder=OptimizationParams(p)
    opt=opt_builder.extract(p.parse_args([]))
    cases={}
    for mode in ('original','corrected'):
        captured,iteration=torch.load(args.checkpoint,map_location='cuda',weights_only=False)
        if captured[16]!=GaussianModel.App_model.PGSR:
            raise ValueError('This regression test targets the actual PGSR recipe.')
        g=GaussianModel(3,0)
        g.create_app_model(len(captured[-1]),captured[16])
        if mode=='original':g.restore(captured,opt)
        else:restore_bound(g,captured,opt)
        bound=True
        try:verify_binding(g)
        except ValueError:bound=False
        expected=state_digest(captured)
        actual=state_digest(g.capture())
        before=g._appearance_embeddings.detach().clone()
        g._appearance_embeddings.sum().backward()
        g.optimizer.step()
        change=float((g._appearance_embeddings-before).abs().max())
        cases[mode]={'parameter_binding_correct':bound,
            'captured_state_content_identical':expected==actual,
            'appearance_optimizer_step_max_change':change,'iteration':iteration}
        del captured,g
        torch.cuda.empty_cache()
    assert not cases['original']['parameter_binding_correct']
    assert cases['original']['appearance_optimizer_step_max_change']==0
    assert cases['corrected']['parameter_binding_correct']
    assert cases['corrected']['captured_state_content_identical']
    assert cases['corrected']['appearance_optimizer_step_max_change']>0
    result={'cases':cases,'source_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (args.checkpoint,Path(__file__),ROOT/'research/checkpoint_state.py',ROOT/'scene/gaussian_model.py')},
        'limitations':['One existing PGSR checkpoint; other appearance modes were not exercised.',
        'Supplemental filter, RNG and camera queue require the separate runtime sidecar; this test only checks captured state and binding.',
        'Historical from-scratch NCC runs are unaffected by the restore-only bug.']}
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(cases,indent=2))


if __name__=='__main__':
    main()
