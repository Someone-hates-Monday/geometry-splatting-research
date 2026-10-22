# v24 FP64 batched finite-support image-loss module

2026-10-21. Development stage within authorized mathematical/small CUDA diagnostics.
No training,backward,accepted edit,new download,or native TGS forward in this phase.
All32 actual v23 candidate/view images and scores have already been observed. This is
an implementation correctness study,not a new quality test or untouched validation set.

Implement a separate CUDA module consuming original float32 RGB,GT uint8 RGB,known
pixel ROIs and concatenated float32 counterfactual RGB patches. Keep original pixels
outside each ROI. Return Kx3 FP64 normalized changes:raw MSE,raw L1,raw SSIM. Form .8
pixel-.2 SSIM changes outside the kernel without asserting the checkpoint's loss branch.
Normal mode returns scalar scores/stats,not full RGB or SSIM maps. Debug maps expose
the exact same moment and loss path for comparison. Inputs and ROI layout are prepared
outside this API. No renderer/cache integration or timing benefit is implied here.

Two separable radius5 filters compute eight moments for each RGB channel:means of
baseline,trial,GT,squares of each,and baseline*GT/trial*GT. Vertical outputs cover the
SSIM output region R+5 in y and R+10 in x;vertical input fetches reach R+10 in y.
Horizontal outputs cover R+5. Clip to global image and preserve zero padding. Empty
ROIs produce zero change. GT uses double(byte)/255. Gaussian weights are v11 float64.
No image clamp or quantization enters these raw loss terms. Compile without fast math
or FMA contraction;this is not a bitwise reproduction of FP32 training convolution.

Sum per-output signed differences using fixed256-thread trees and deterministic
per-proposal final block reductions. No floating atomic reduction. Count nonfinite
inputs/maps and require zero flags. Same code handles full-frame reference ROIs.
The full-frame baseline receives the complete reconstructed trial,so local/full
differences test finite support and packing against identical GPU arithmetic.

Before real cases,software fixtures cover empty/full ROI,corners,thin border edits,
duplicate proposals,unchanged patches and raw values outside[0,1]. Mark them synthetic.
For all32 real cases,verify lossless full RGB hashes against v23;run all8 alternatives
per known view in one local call;verify normal/debug scalar literal equality;compare
local maps with full-frame maps literally at every retained pixel and require exact
zero full-frame map changes outside R+5. Preserve every case including small/zero effects.

Compare per-element GPU SSIM maps to independent SciPy v11 FP64 maps with max absolute
tolerance1e-10. Compare normalized scalar changes to prior v23 full-image references
with abs tolerance1e-13+1e-9*abs(reference). Freeze tolerances before building/running.
No outcome-dependent threshold widening. Report actual errors,all zeros,blend signs
and any ambiguous sign within the declared scalar error allowance. GPU full/local
map literal equality is a separate gate that tolerance cannot rescue. Scalar reductions
can round differently because local/full block boundaries differ;do not call them
integer-exact. Full-image input packing,CPU layout and H2D preparation are excluded
from this functional module study;no timing/speed/memory advantage is claimed.

Freeze source,runner,adapter,protocol,Gaussian implementation and all v23 RGB/GT inputs
before compile/fixtures/real execution. Use retained failure records and distinct
derivative attempts if needed;never rerun a live job because an observation timed out.
Keep goal incomplete:renderer integration,actual cost,policy utility,independent data,
geometry/image quality and novelty remain separate unmet requirements.
