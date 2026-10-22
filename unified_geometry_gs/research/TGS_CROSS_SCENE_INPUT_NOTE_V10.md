# v10 input checks before GPU output

The published flowers GT PNG has raster dimensions1256x828. Its COLMAP camera JSON has
width5025,height3312 and cfg images_4. The old stump-only check ceil(width/4) is not a universal
loader rule:5025/4=1256.25, while its actual PNG width is1256. No GPU outputs had been observed
when this was detected by the PNG header check.

The pinned official `sources/tgs_real_checkpoint_v1/source/utils/camera_utils.py:20` reads
decoded `cam_info.image.size`. With resolution=-1 and image width<=1600, it retains that raster
size. `source/scene/dataset_readers.py:98-113` obtains FoV from the COLMAP focal/dimensions
and the pixel data from the selected image folder. These are separate sources of information.

v10 therefore uses the published PNG dimensions, checks their consistency with the scene's
image-factor metadata to within its integer rounding bracket, and retains COLMAP width/focal
ratios for FoV. It does not force a ceil/floor rule or scale the focal angle twice. The other training
image bytes are not individually decoded; assuming a shared scene raster size is explicit in the
input audit. The PNG's camera identity remains unverified and it is not used for quality metrics.
This corrects an input-assumption implementation before execution; no precision threshold,
camera selection or render result was changed to make a failed candidate pass.

Acquisition first stopped with an actual RemoteDisconnected exception on the final room PNG
local ZIP header, after nine complete members had been CRC/SHA checked. Session71236 exit1
is recorded in acquisition_attempts.json. The same frozen extractor is resumed without changing
its source, overwriting partial data or reselecting scenes; complete files are verified and skipped.
The acquisition audit counts all successful new cached Range payloads across both attempts,
not just the final attempt's smaller byte count. No full-archive download fallback is allowed.
