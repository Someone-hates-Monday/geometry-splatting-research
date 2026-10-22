"""Refresh the existing managed research canvas from completed result files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANVAS = Path("C:/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/canvases/GGGS-research-audit.canvas.tsx")


def main():
    text = CANVAS.read_text(encoding="utf-8")
    if '"完整 NCC pilot"' in text:
        raise ValueError("Canvas already has this result update; inspect before rewriting.")
    pilot = json.loads((ROOT / "research/results/exact37_report.json").read_text())
    gpu = json.loads((ROOT / "research/results/median_cuda_summary_v1.json").read_text())
    replacements = {
        '"求根原型"];': '"求根原型", "完整 NCC pilot", "GPU 求根对照"];',
        "已有证据支持配方，新的学术结论仍需验证": "NCC 未过几何门槛；效率控制正在验证几何代价",
        "已找到稳定化 NCC 的导数不一致，并构造 GGGS 等价条件之外的反例。正确性检查已通过；两组 50 步训练集成已通过；30k 成对实验正在运行，真实重建质量收益尚无完整结果。新增三场景 CPU 射线诊断，尚待 CUDA 核对。":
            "完整 30k NCC 对照已完成：scan37 overall 退化约 0.01119 mm，训练视角 PSNR 增加 0.155 dB。CPU/CUDA 抽样核对通过。GPU Newton 原型更慢；四轮细分控制的冻结渲染延迟降低约 14%，正在核验网格代价。",
        "但混合 Newton 的评估次数中位数为 33，二分为 32，排序成本未计入：当前没有加速证据。":
            "旧混合 Newton 的评估次数中位数为 33，二分为 32；新停止规则另列于求根原型页，GPU 原型另列于 GPU 对照页。",
        "CPU float64 仿射投影尚待 CUDA 核对，不能解释为生产错误率或质量改善。":
            "3072 条射线通过 CPU/CUDA 前向抽样核对，深度和透明度在预定容差内，有效性零差异；这些机制量不等于生产错误率或质量改善。",
        "CPU 仿射模型待 CUDA 核对；不是 GPU 遍历数或耗时。":
            "CPU 仿射前向抽样核对已通过；函数评估次数不是 GPU 遍历数或耗时。",
        "尚无 GPU 提速结论": "GPU 迁移结果见独立对照页",
    }
    for old, new in replacements.items():
        if old not in text:
            raise ValueError(f"Missing canvas anchor: {old[:50]}")
        text = text.replace(old, new)
    image_rows = []
    for key, label in (("psnr_global_rgb_db", "PSNR（dB，训练视角）"), ("ssim_repository", "SSIM（训练视角）")):
        a, b = (pilot["training_view_image_means"][arm][key] for arm in ("legacy", "exact"))
        image_rows.append([label, f"{a:.6f}", f"{b:.6f}", f"{b-a:+.6f}"])
    mesh = pilot["unseeded_mesh_evaluation"]
    metric_rows = [[label, f"{mesh['legacy'][key]:.6f}", f"{mesh['exact'][key]:.6f}",
        f"{mesh['delta_exact_minus_legacy'][key]:+.6f}"] for key, label in (("overall", "overall（mm）"), ("mean_d2s", "d2s（mm）"), ("mean_s2d", "s2d（mm）"))]
    categories = [s["model"].rsplit("/", 1)[-1] for s in gpu["scenes"]]
    timing_rows = []
    for category, scene in zip(categories, gpu["scenes"]):
        v = scene["variants"]
        timing_rows.append([category, f"{v['control']['mean_view_gpu_ms']:.3f}",
            f"{v['four']['mean_view_gpu_ms']:.3f}", f"{v['hybrid']['mean_view_gpu_ms']:.3f}",
            f"{100*(1-v['four']['median_view_latency_ratio_to_control']):.1f}%"])
    series = [{"name": label, "data": [s["variants"][key]["mean_view_gpu_ms"] for s in gpu["scenes"]], "tone": tone}
        for key, label, tone in (("control", "同源五轮控制", "neutral"), ("four", "四轮控制", "info"), ("hybrid", "Newton hybrid", "neutral"))]
    dump = lambda value: json.dumps(value, ensure_ascii=False)
    addition = '''    {tab === 5 && <Stack gap={18}>
      <H2>scan37 完整 NCC pilot：几何未达采纳门槛</H2>
      <Text>各 30k；训练种子 0；no_dn + trunc4。距离越低越好，图像指标越高越好。</Text>
      <Table headers={["指标", "legacy", "exact", "exact−legacy"]} rows={METRICS} />
      <Text>三个预定 evaluator seed 的平均 Δoverall 为 +0.01119016 mm，范围 +0.01110359～+0.01125731 mm，均退化；legacy seed 0 重复得到相同数值。评估重复不能冒充训练重复。</Text>
      <Text>首个 NCC 调用的输入块数为 333,162 与 325,858。在任何 NCC 反传前状态已不同，所以相同种子没有提供相同 warmup；当前对照不能严格归因于导数修正。</Text>
      <Text>保留导数正确性修复；暂不扩大 NCC 训练。没有新视角、统计显著性或训练加速结论。</Text>
      <Text tone="secondary" size="small">来源：NCC_PILOT_RESULT.md、results/exact37_report.json、exact37_v1/postprocess_v2、seeded_eval_v2；2026-10-12。相同完整 49 个训练相机，未留出测试相机。</Text>
    </Stack>}
    {tab === 6 && <Stack gap={18}>
      <H2>GPU 实测：当前 Newton 原型更慢，四轮控制约快 14%</H2>
      <Text>冻结旧模型，每场景预定 8 个视角。5 轮 × 20 次 CUDA event 计时，预热后随机化变体和 depth on/off 顺序；下列为每视角轮次中位数再跨视角求均值。</Text>
      <Text tone="secondary" size="small">纵轴：带深度渲染 GPU 时间（ms）；横轴：DTU 场景。图例为不同隔离后端。</Text>
      <BarChart categories={CATEGORIES} series={SERIES} height={260} valueSuffix=" ms" />
      <Table headers={["场景", "五轮（ms）", "四轮（ms）", "hybrid（ms）", "四轮延迟降低中位数"]} rows={TIMINGS} />
      <Text>所有 24 个完整图像的有效性掩码一致，RGB、alpha、法向最大差为 0。同源 JIT 与已安装五轮后端的深度逐位一致。四轮相对五轮的最大深度差约 5.866×10⁻⁵（归一化相机深度单位），抽样 CPU 误差均在 10⁻⁴ 内。</Text>
      <Text>hybrid 延迟增加约 22%～25%，拒绝当前效率实现。四轮为简单控制，不能作为新颖性声明；三个模型正在重新提取网格，尚不能认定几何代价通过。此处没有训练吞吐结论。</Text>
      <Text tone="secondary" size="small">来源：results/median_cuda_frozen_v1.json、median_cuda_summary_v1.json、CUDA_SOLVER_RESULTS.md；RTX 5090 Laptop，2026-10-12。GPU 端点检查非严格区间证书，hybrid 斜率通道与历史 secant 有差异。</Text>
    </Stack>}
'''.replace("METRICS", dump(metric_rows + image_rows)).replace("CATEGORIES", dump(categories)).replace("SERIES", dump(series)).replace("TIMINGS", dump(timing_rows))
    anchor = '    <Text size="small" tone="tertiary">所有数字'
    if text.count(anchor) != 1:
        raise ValueError("Missing insertion anchor.")
    text = text.replace(anchor, addition + anchor)
    CANVAS.write_text(text, encoding="utf-8")
    print(json.dumps({"canvas": str(CANVAS), "sha256": hashlib.sha256(CANVAS.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
