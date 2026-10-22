        // R=q*H_t, q=.5. Use T'(t)=-T*H_t for the Newton correction,
        // then a first-order update of R before the existing numerator pass.
        // No new contributor traversal. Keep the historical derivative floor.
        const float solver_R=-dT_dtm;
        if (mDepth>0.f && dL_dpixel_mt!=0.f && solver_R>1e-7f && solver_T>0.f) {
            const float correction=0.5f*(solver_T-0.5f)/(solver_T*solver_R);
            const float predicted_R=solver_R+solver_dR*correction;
            if (isfinite(correction) && isfinite(predicted_R) && predicted_R>1e-7f &&
                fabsf(correction)<0.25f*solver_peak_distance && fabsf(correction)<0.0002f) {
                mDepth+=correction;
                dT_dtm=-predicted_R;
            }
        }
