# v25 renderer-cache to packed RGB to FP64 image-loss integration

2026-10-21. Authorized fixed-input mathematical/small CUDA diagnostics only. No training,
backward,accepted scene edit,optimizer,new download or new scientific quality dataset.
All32 v23 images/metric outcomes are known. Do not call this independent replication.

Reuse original v21 RGB control,v22 box planner and immutable v24 loss binary. Implement
a separate packed-patch replay with the same positive-depth stable merge,parent removal,
four own children,window/alpha/cap/skip-current early stop. No fixed list truncation.
Every pixel of each bounding rectangle is written:active tiles are replayed;holes with
no parent/child membership copy the original RGB. Return compact CHW patch buffers and
diagnostic flags,never clone K complete images as an intermediate. ROI may itself span
most/all of an image. No alpha/depth/path/backward outputs or GT-dependent edit acceptance.

For each of the two known cameras per flowers/room,make one original control forward
and one32-child bank forward:4views*2=8 new native/control forwards. Discard bank mixed
RGB. Original baseline,bank coordinates/attributes/colors,computed pixel rectangles,
packed RGB,full reconstructed image hashes and Kx3 FP64 score vectors must reproduce
the existing v20/v23/v24 records. Score vectors must be literal to v24 prepared-patch
results;CPU-reference tolerances remain those already frozen in v24. Keep all cases.
No new complete candidate render is needed because its lossless full reference is
already sealed. Hash source/models/cameras/GT/references before execution;check model
input hashes after each view. Persist per-view receipts before final aggregation.

Software-only fixtures first:stable ties,independent candidates,deleted parent,empty
ROI,257 retained original layers,and a three-tile rectangle with an unaffected middle
tile. Middle-tile RGB must copy the original rather than use uninitialized output.
These fixtures are not image-quality evidence. Keep failures;do not widen literal gates.

The integrated endpoint must include dynamic child forward,geometry unpack,ROI plan,
GPU-to-CPU rectangle synchronization,CPU layout,GPU layout transfers,packed replay and
FP64 loss/all allocations. Current CPU layout is a prototype cost,not a free prepared
artifact in any later speed comparison. Model/GT/proposal/color preparation is separate.
Only after correctness passes,freeze a balanced full-pipeline timing protocol against
full-scene+same-loss,cached-image+same-loss,sequential packed+same-loss,and batched packed.
Persist baseline setup immediately to avoid the v22 final-metadata loss. Compute forward
budgets from the actual arm/count design. No speed,quality,policy or novelty conclusion
is presupposed;the full scientific objective remains open.
