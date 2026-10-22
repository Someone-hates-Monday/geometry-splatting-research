# Follow-up diagnosis of retained raw v1 failures

Written after v1 completed. Its original verdict remains factor891/891,
dense890/891, direction35/42. Do not replace these with tuned-step counts.
All seven failed direction pairs are retained, with regenerated directions
checked against the originally recorded predicted derivative. Use240-digit
independent raw-parameter forward arithmetic (true covariance inverse;
same captured-base continuous extension; real kernel0) and re-solved roots
at +/-h for h=1e-6,1e-8,1e-10,1e-12. Compare to the MP analytic raw chain
contracted with that fixed direction; last relative tolerance1e-5 with
absolute1e-9 floor, matching the original test. Also record Float64
directional differences at all four micro steps and cap-mask changes.
This diagnoses truncation/conditioning/chain errors, not new confirmation.

The dense failure at scan55_0034 pixel167992 remains a rejected Float64
comparison implementation; no artificial damping of its covariance or
replacement of original input is allowed. Factor and independent MP may
still provide evidence about the mathematical derivative. No CUDA or
training implementation begins on the basis of a changed gate.
