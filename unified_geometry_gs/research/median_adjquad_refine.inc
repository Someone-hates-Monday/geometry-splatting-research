        // Halley step for F=T-.5, T'=-T H_t, T''=T(H_t^2-H_tt).
        // R=.5 H_t, dR=.5 H_tt, ddR=.5 H_ttt. The half-distance
        // guard keeps the accepted segment away from every peak. It does not
        // certify that the true root is in that segment or bound roundoff.
        const float solver_R=-dT_dtm;
        if (mDepth>0.f && dL_dpixel_mt!=0.f && solver_R>1e-7f && solver_T>0.f) {
            const float residual_ratio=(solver_T-0.5f)/solver_T;
            const float halley_den=1.f-0.5f*residual_ratio*(1.f-0.5f*solver_dR/(solver_R*solver_R));
            const float correction=0.5f*residual_ratio/(solver_R*halley_den);
            const float predicted_R=solver_R+solver_dR*correction+0.5f*solver_ddR*correction*correction;
            if (isfinite(correction) && isfinite(predicted_R) && halley_den>0.f && predicted_R>1e-7f &&
                fabsf(correction)<0.5f*solver_peak_distance && fabsf(correction)<0.0002f) {
                mDepth+=correction;
                dT_dtm=-predicted_R;
            }
        }
