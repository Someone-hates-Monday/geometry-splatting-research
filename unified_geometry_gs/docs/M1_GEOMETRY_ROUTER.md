# M1 geometry-aware soft router（低风险试探）

## 思路（与常见 MoE 的差异）

- **不是**按场景 ID、不是拆多套高斯参数、不是加深主干网络。
- **是**在同一次渲染得到的 `median_depth` / `alpha` 上，按 **patch 级几何统计**（粗分辨率下的深度、α、半透明歧义 `4α(1−α)`、边界强度）算特征。
- 一个小 MLP 输出 **3 个专家的 softmax 权重**；每个专家对 M1 四项（`L_occ`, `L_depth_occ`, `L_rank`, `L_thin`）有一组**固定的、可解释**的倍率。
- 最终每个迭代的有效权重 = **patch 平均后的倍率 × 你在 CLI 里设的 `lambda_*`**。
- 额外加 **`router_balance_loss`**，惩罚「永远只选一个专家」的塌缩。

这与「表示层 MoE」不同：我们只做 **物理可解释的 M1 损失调度**，便于消融与叙事。

## 专家预设（倍率表）

列顺序：`occ`, `depth_occ`, `rank`, `thin`

| 专家 | 含义（解释性标签） | 作用倾向 |
|------|---------------------|----------|
| 0 | smooth / flat | 弱化整组 M1，减少平坦区过正则 |
| 1 | boundary | 强化 depth_occ + rank |
| 2 | thin / cavity proxy | 强化 thin 项 |

初始化时会对列做归一化，使 **均匀路由时倍率≈1**，接近原始 `lambda_*`。

## 用法

```bash
cd unified_geometry_gs
source configs/dtu_path.env
export PYTHON=python

./scripts/run_m1_router.sh "$UGGS_DTU_ROOT/scan105" ./output_compare/m1_router/scan105
```

关键参数：

- `--use_m1_geometry_router`
- `--m1_router_balance`（默认 0.02，负载均衡强度）
- `--m1_router_lr`（路由 MLP 学习率）
- `--m1_router_temperature`（softmax 温度，>1 更平滑）
- `--m1_router_grid_h` / `--m1_router_grid_w`（默认 4×4 patch）

## 停止条件（试探）

- TensorBoard 或日志里若某专家长期占比 →0，先增大 `m1_router_balance` 或温度。
- 若 mesh 均收益不如 **固定 M1-3**，关掉路由对照 `run_m1.sh` / `run_m1_thin.sh`。

## 训练诊断（无需 TensorBoard）

启用路由时，每隔 `--m1_router_log_interval`（默认 500）步在：

`output_compare/<run>/debug/m1_router_usage.csv`

追加一行：`e0,e1,e2` 为三专家平均使用率，`w_*` 为有效倍率，`balance_loss` 为负载均衡项大小。  
若某 `e*` 长期接近 0，说明塌缩或温度/均衡权重需调。

## scan105 若 mesh 变差而图像略好

优先尝试（单独或组合）：

1. **关掉 thin**：`LAMBDA_OCC_THIN=0 ./scripts/run_m1_router.sh ...`（脚本默认带 thin，易与路由叠加过激）
2. **提高温度**：`M1_ROUTER_TEMP=1.2`（路由更平滑）
3. **略降均衡**：`M1_ROUTER_BALANCE=0.01`
4. 与固定 **M1-3** 并排对比：`scripts/compare_scan_methods.sh`

## 限制（已知）

- 路由 checkpoint 未与 `chkpnt*.pth` 绑定；短期实验可接受，正式跑需另存 `router.pt`（后续可加）。
