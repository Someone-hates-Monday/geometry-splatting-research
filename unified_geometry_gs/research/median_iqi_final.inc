            if constexpr (decltype(last_const)::value) {
                const int neighbor = start_id > 0 ? start_id - 1 : start_id + 2;
                const float k = static_cast<float>(neighbor-start_id);
                const float y0 = T_p[start_id]-0.5f;
                const float y1 = T_p[start_id+1]-0.5f;
                const float y2 = T_p[neighbor]-0.5f;
                const float linear_s = __saturatef(y0/(y0-y1));
                float s = linear_s;
                // Pairwise differences avoid products of tiny denominators.
                // Nearly coincident T values or extrapolation use the original
                // linear interpolant; this is an engineering safeguard.
                if (fabsf(y1-y0) > 1e-7f && fabsf(y1-y2) > 1e-7f && fabsf(y2-y0) > 1e-7f) {
                    const float candidate = (y0/(y1-y0))*(y2/(y1-y2)) +
                        k*(y0/(y2-y0))*(y1/(y2-y1));
                    if (isfinite(candidate) && candidate >= 0.f && candidate <= 1.f)
                        s = candidate;
                }
                inverse_quad_depth = depth_min + (start_id+s)*interval;
            }
