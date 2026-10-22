# Pre-execution input construction correction, v4.1

Before v4 native output, code inspection found the first16 new opacity-target
constructions added EPS to a float32 depth denominator before dividing double
numerators. EPS may round away. The original111 frozen inputs are retained
byte-for-byte; they are near-threshold cases, not asserted exact CPU64 targets.
Append16 paired corrected-denominator inputs using float64 depth+EPS, the same
colors, signed patterns, depths, winding and ULP shifts. The extended127-case
plan references the original plan hash. No native outcome selection occurred.
Both compositing families use adaptive-v4 versus oracle-v4 gates; CPU64 branch
disagreements remain visible. No protocol thresholds or old inputs are changed.
