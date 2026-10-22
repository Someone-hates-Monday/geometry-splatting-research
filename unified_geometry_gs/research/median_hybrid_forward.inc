        // Retain original nine-point first pass and its original validity gate.
        iter(std::true_type{});
        float interpolated = depth_min + (depth_max - depth_min) *
            __saturatef((T_p[0] - 0.5f) / (T_p[0] - T_p[SPLIT]));
        float test_depths[3], test_T[3], test_Ht = 0.f;

        // All threads participate in shared-memory barriers. Inactive pixels
        // only skip arithmetic, never leave the collective loop early.
        auto evaluate_local = [&] __device__(int count) {
            test_T[0] = test_T[1] = test_T[2] = 1.f;
            test_Ht = 0.f;
            bool done = !in_range || !inside;
            int toDo = max_contributor;
            contributor = 0;
            for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
                block.sync();
                int progress = i * BLOCK_SIZE + block.thread_rank();
                if (progress < max_contributor) {
                    int coll_id = point_list[range.x + progress];
                    collected_xy[block.thread_rank()] = points_xy_image[coll_id];
                    collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
                    collected_ray_planes[block.thread_rank()] = ray_planes[coll_id];
                }
                block.sync();
                for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
                    contributor++;
                    done = contributor >= last_contributor;
                    float2 xy = collected_xy[j];
                    float2 d = {xy.x - pixf.x, xy.y - pixf.y};
                    float4 con_o = collected_conic_opacity[j];
                    float power = -0.5f * (con_o.x*d.x*d.x + con_o.z*d.y*d.y) - con_o.y*d.x*d.y;
                    if (power > 0.f) continue;
                    float alpha = fminf(0.99f, con_o.w * expf(power));
                    if (alpha < 1.f/255.f) continue;
                    float4 plane = collected_ray_planes[j];
                    float peak = plane.x*d.x + plane.y*d.y + plane.z;
                    float rsigma = plane.w;
                    for (int k = 0; k < count; k++) {
                        float delta = (test_depths[k] - peak) * rsigma;
                        float gg = rsigma > 0.f ? alpha * expf(-0.5f*delta*delta) : 0.f;
                        float vacancy2 = 1.f - gg;
                        test_T[k] *= (test_depths[k] > peak ? (1.f-alpha) : vacancy2) * rsqrtf(vacancy2);
                        if (k == 0 && rsigma > 0.f)
                            test_Ht += gg * fabsf(delta) * rsigma / (2.f * vacancy2);
                    }
                }
            }
        };

        test_depths[0] = interpolated;
        evaluate_local(1);
        float slope = test_T[0] * test_Ht;
        float candidate = interpolated + (test_T[0] - 0.5f) / fmaxf(slope, 1e-20f);
        bool candidate_ok = isfinite(candidate) && slope > 0.f &&
            candidate > depth_min && candidate < depth_max;
        candidate = candidate_ok ? candidate : interpolated;
        constexpr float LOCAL_TOL = 1e-6f;
        constexpr float T_MARGIN = 2e-6f;
        float probe_lo = fmaxf(depth_min, candidate - LOCAL_TOL);
        float probe_hi = fminf(depth_max, candidate + LOCAL_TOL);
        test_depths[0] = 0.5f * (probe_lo + probe_hi);
        test_depths[1] = probe_lo;
        test_depths[2] = probe_hi;
        evaluate_local(3);
        bool accepted = in_range && inside && candidate_ok &&
            test_T[1] >= 0.5f + T_MARGIN && test_T[2] <= 0.5f - T_MARGIN;
        float fast_depth = test_depths[0];
        float fast_slope = 0.5f * test_Ht;

        // Fallback is identical to the historical remaining four passes.
        // A tile can avoid the fallback fetches only when all its pixels pass.
        int all_accepted = __syncthreads_and(accepted || !in_range || !inside);
        if (!all_accepted) {
            for (int i = 0; i < SPLIT_ITERATIONS - 1; i++)
                iter(std::false_type{}, accepted || !inside);
        }
        float w_max = __saturatef((T_p[0] - 0.5f) / (T_p[0] - T_p[SPLIT]));
        mDepth = in_range ? (accepted ? fast_depth : depth_min + w_max*(depth_max-depth_min)) : 0.f;
        if (in_range) {
            tslope = accepted ? fast_slope : fabsf(T_p[0]-T_p[SPLIT]) / fmaxf(depth_max-depth_min, 1e-6f);
        }
