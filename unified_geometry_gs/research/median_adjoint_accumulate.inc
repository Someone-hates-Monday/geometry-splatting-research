                dT_dtm += -0.25f * Gt / (1.f - Gt) * fabsf(t_delta) * rsigma;
                const float solver_g = rsigma > 0.f ? Gt : 0.f;
                const float solver_v = 1.f-solver_g;
                solver_T *= (mDepth>t_peak ? (1.f-alpha) : solver_v)*rsqrtf(solver_v);
                if (rsigma>0.f) {
                    const float side = t_delta>0.f ? 1.f : (t_delta<0.f ? -1.f : 0.f);
                    solver_dR += 0.25f*Gt*rsigma*rsigma*side*
                        ((1.f-Gt)-t_delta*t_delta)/((1.f-Gt)*(1.f-Gt));
                    solver_peak_distance = fminf(solver_peak_distance,fabsf(mDepth-t_peak));
                }
