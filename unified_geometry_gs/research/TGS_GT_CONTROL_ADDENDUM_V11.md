# Additional archived-source control, fixed before GT payloads or scores

2026-10-20. Directory-only catalogue is complete; no original/GT image payloads or quality
scores have been inspected. Add exactly one fourth forward-only baseline to the three in
TGS_GT_IDENTITY_QUALITY_PROTOCOL_V11.md: archived_native, the existing unedited source
checkout binary, SHA9d974f09b7217a026aaad3ebc48af16c6a9571fcd17a885711b0517b9d409da1.
Receipt: experiments/tgs_subdivision_v1/native_build/receipt_9d974f09b721.json, native commit
6d61f4c0a571c635e37ffb740250706fa30a5541 and source_edits=false.

This supplies a compatibility control for published weights because historical piecewise and
native64 contain prior fixes. It must not be called a proven reproduction of the actual training
code: that training commit is unknown. Do not rebuild or modify the source, replace a camera,
change metrics/quantization or optimize its parameters. All four baselines are retained even if
an unfixed source happens to give better zero-step quality. This comparison diagnoses operator
compatibility; it cannot isolate one fix or establish the quality of a retrained method.
