# -*- coding: utf-8 -*-
"""
从 guizang Swiss 模板生成 index.html，内容与「基于三维高斯泼溅的多视图重建几何优化研究」PPT 主线对齐；
增大正文字号；Head 注入 MathJax + 局部样式；尾部钩子对翻页后重新 typeset。
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
UGGS_ROOT = HERE.parent.parent  # unified_geometry_gs
TEMPLATE = UGGS_ROOT.parent / "tools" / "guizang-ppt-skill" / "assets" / "template-swiss.html"
INDEX = HERE / "index.html"

HEAD_INJECT = """
<style id="uggs-deck-tune">
  /* 全局加大正文（瑞士模板默认偏细） */
  .canvas-card .col-desc{font-size:max(17px,1.22vw)!important;line-height:1.58!important;max-width:36vw}
  .canvas-card .col-ttl{font-size:min(3.9vw,6.8vh)!important}
  .canvas-card .layer-desc{font-size:max(16px,1.15vw)!important;line-height:1.55!important}
  .canvas-card .layer-ttl{font-size:max(20px,1.85vw)!important}
  .canvas-card .sub-card .desc{font-size:max(15px,1.08vw)!important}
  .canvas-card .sub-card .ttl{font-size:max(18px,1.55vw)!important}
  .canvas-card .lead, .canvas-card .lead-col p{font-size:max(17px,1.2vw)!important;line-height:1.55!important}
  .canvas-card .body, .canvas-card .body-sm{font-size:max(15px,1.05vw)!important}
  .canvas-card ul.col-list li{font-size:max(15px,1.05vw)!important}
  .chrome-min{font-size:max(12px,.95vw)!important}
  .mjt{font-size:max(16px,1.12vw); line-height:1.55; color:var(--text-primary)}
  .mjt mjx-container{display:inline-block;margin:.15em 0}
  .uggs-table{width:100%;border-collapse:collapse;font-size:max(15px,1.08vw)}
  .uggs-table th,.uggs-table td{border:1px solid var(--border-subtle);padding:.55em .7em;text-align:left}
  .uggs-table th{background:var(--grey-1);font-family:var(--mono);font-size:max(12px,.88vw);letter-spacing:.06em}
  /* 简易流程图（无外部依赖） */
  .uggs-flow{display:flex;flex-direction:column;gap:.5em;margin:.75em 0 0;font-size:max(13px,.88vw);line-height:1.45;color:var(--text-secondary);border:1px solid var(--border-subtle);border-radius:6px;padding:.85em 1em;background:var(--grey-0)}
  .uggs-flow-caption{font-family:var(--mono);font-size:max(11px,.72vw);letter-spacing:.12em;color:var(--text-helper);margin-bottom:.25em}
  .uggs-flow-row{display:flex;align-items:stretch;justify-content:center;gap:.4em;flex-wrap:wrap}
  .uggs-flow-box{flex:1 1 auto;min-width:min(100%,9em);max-width:100%;border:1px solid var(--border-subtle);border-radius:4px;padding:.5em .65em;background:var(--paper);color:var(--text-primary);text-align:center;font-size:max(13px,.88vw);line-height:1.4}
  .uggs-flow-box strong{display:block;font-size:max(12px,.78vw);color:var(--text-helper);font-weight:500;margin-bottom:.2em;letter-spacing:.04em}
  .uggs-flow-arr{align-self:center;color:var(--accent);font-weight:700;font-size:max(14px,1vw);padding:0 .15em}
  .uggs-flow-note{margin-top:.55em;font-size:max(13px,.92vw);color:var(--text-secondary);line-height:1.5;border-top:1px dashed var(--border-subtle);padding-top:.55em}
  /* 第 4 页主链路：框上下加高、字更大；块内行距收紧 */
  #deck .uggs-flow--large{flex:1 1 auto!important;min-height:min(40vh,50%)!important;margin-top:.5em!important;padding:clamp(12px,1.8vh,22px) clamp(12px,2vw,22px)!important;gap:clamp(.22em,.45vh,.5em)!important;font-size:max(16px,1.12vw)!important;justify-content:space-evenly!important}
  #deck .uggs-flow--large .uggs-flow-caption{font-size:max(13px,.88vw)!important;margin-bottom:.22em!important;letter-spacing:.14em}
  #deck .uggs-flow--large .uggs-flow-row{gap:.42em!important;align-items:stretch!important}
  #deck .uggs-flow--large .uggs-flow-box{min-width:min(100%,8.5em)!important;padding:1.05em .72em!important;min-height:clamp(3.4em,8.2vh,5.8em)!important;display:flex!important;flex-direction:column!important;justify-content:center!important;font-size:max(16px,1.1vw)!important;line-height:1.42!important;background:#eef5ff!important}
  #deck .uggs-flow--large .uggs-flow-box strong{font-size:max(14px,.93vw)!important;margin-bottom:.32em!important}
  #deck .uggs-flow--large .uggs-flow-arr{font-size:max(20px,1.42vw)!important;padding:0 .16em!important;align-self:center!important}
  #deck .uggs-flow--large .uggs-flow-note{font-size:max(15px,1.05vw)!important;line-height:1.52!important;padding-top:.55em!important;margin-top:.35em!important}
  /* 第 5–7 页对照流程条：小框上下加高、字更大；各行间距收紧 */
  #deck .uggs-flow-host.mjt{display:flex!important;flex-direction:column!important;flex:1!important;min-height:0!important}
  #deck .uggs-models-strip--large.uggs-flow{flex:1 1 auto!important;min-height:min(44vh,54%)!important;margin-top:.35vh!important;padding:clamp(12px,1.9vh,24px) clamp(11px,2vw,22px)!important;gap:clamp(.2em,.38vh,.48em)!important;justify-content:space-evenly!important}
  #deck .uggs-models-strip--large .uggs-flow-caption{font-size:max(13px,.9vw)!important;margin-bottom:.18em!important}
  #deck .uggs-models-strip--large .uggs-flow-row{gap:.32em!important}
  #deck .uggs-models-strip--large .uggs-flow-row[style*="margin-top"]{margin-top:.22em!important}
  #deck .uggs-models-strip--large .uggs-flow-box:not(.uggs-flow-box--full){min-width:min(100%,6em)!important;padding:.88em .48em!important;min-height:clamp(3.1em,7.5vh,5.2em)!important;display:flex!important;flex-direction:column!important;justify-content:center!important;font-size:max(15px,1.02vw)!important;line-height:1.36!important;background:#eef5ff!important}
  #deck .uggs-models-strip--large .uggs-flow-box:not(.uggs-flow-box--full) strong{font-size:max(13.5px,.9vw)!important}
  #deck .uggs-models-strip--large .uggs-flow-box.uggs-flow-box--full{padding:.72em .75em!important;min-height:clamp(2.6em,5.5vh,4.2em)!important;display:flex!important;align-items:center!important;justify-content:center!important;font-size:max(14.5px,.98vw)!important;line-height:1.38!important;background:#eef5ff!important}
  #deck .uggs-models-strip--large .uggs-flow-arr{font-size:max(17px,1.28vw)!important;padding:0 .1em!important}
  #deck .uggs-models-strip--large .uggs-flow-note{font-size:max(14.5px,1vw)!important;line-height:1.48!important;padding-top:.48em!important;margin-top:.28em!important}
  .uggs-models-block{margin:.1em 0 .45em;padding-left:.65em;border-left:2px solid var(--border-subtle)}
  .uggs-models-block>p:first-child{margin:0 0 .25em}
  .uggs-models-strip.uggs-flow{margin:.15em 0 0;padding:.45em .55em;gap:.3em}
  .uggs-models-strip .uggs-flow-caption{text-align:left;margin-bottom:.12em;letter-spacing:.06em}
  .uggs-models-strip .uggs-flow-row{gap:.25em}
  .uggs-models-strip .uggs-flow-box.uggs-flow-box--full{flex:1 1 100%;min-width:min(100%,100%);max-width:100%;text-align:center;padding:.38em .5em}
  .uggs-models-strip .uggs-flow-box:not(.uggs-flow-box--full){min-width:min(100%,4.6em);max-width:100%;padding:.32em .38em;font-size:max(11px,.76vw);line-height:1.26}
  .uggs-models-strip .uggs-flow-arr{padding:0 .08em;font-size:max(11px,.82vw)}
  .uggs-models-strip .uggs-flow-note{margin-top:.35em;padding-top:.35em;font-size:max(11px,.76vw)}
  .uggs-models-strip--b .uggs-flow-box{background:#eef5ff}
  .uggs-bullets{margin:.5em 0 0;padding-left:1.15em;font-size:max(14px,1.02vw);line-height:1.55;color:var(--text-secondary)}
  .uggs-bullets li{margin:.25em 0}
  .uggs-duo-tight .duo-compare{margin-top:.25vh;flex:0 1 auto}
  .uggs-duo-tight .col .col-list li{font-size:max(15px,1.08vw)!important;line-height:1.45}
  .uggs-duo-tight .col-desc{font-size:max(15px,1.08vw)!important}
  .uggs-s22-compact .image-hero-stats{margin-top:0!important}
  .uggs-2dgs-split .uggs-2dgs-bullets li{margin:.55em 0}
  #deck .uggs-timeline-lg.timeline-h{margin-top:.6vh!important}
  #deck .uggs-timeline-lg.timeline-h .th-node .dot{width:14px;height:14px;border:2px solid var(--ink);background:var(--paper)}
  #deck .uggs-timeline-lg.timeline-h .th-node.accent .dot{border-color:var(--accent);background:var(--accent)}
  #deck .uggs-timeline-lg.timeline-h .th-node .label .yr{font-size:max(13px,.95vw)!important}
  #deck .uggs-timeline-lg.timeline-h .th-node .label .name{font-size:max(17px,1.22vw)!important;line-height:1.2}
  #deck .uggs-timeline-lg.timeline-h .th-node .label .desc{font-size:max(15px,1.06vw)!important;line-height:1.35}
  #deck .uggs-timeline-lg.timeline-h .th-node.up .label{bottom:calc(50% + 30px)!important}
  #deck .uggs-timeline-lg.timeline-h .th-node.down .label{top:calc(50% + 30px)!important}
</style>
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$','$$'], ['\\\\[','\\\\]']],
    processEscapes: true
  },
  svg: { fontCache: 'global' },
  startup: {
    pageReady: () => MathJax.startup.defaultPageReady().then(() => {
      if (window.__mjReady) window.__mjReady();
    })
  }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" id="MathJax-script"></script>
"""

BODY_SCRIPT = """
<script>
(function(){
  function mjTypeset(){
    if (!window.MathJax || !MathJax.typesetPromise) return Promise.resolve();
    return MathJax.typesetPromise([document.getElementById('deck')]).catch(function(){});
  }
  window.__mjReady = function(){
    mjTypeset();
    var old = window.__playSlide;
    if (typeof old === 'function' && !window.__playSlideWrapped) {
      window.__playSlideWrapped = true;
      window.__playSlide = function(i){
        var r = old.apply(this, arguments);
        Promise.resolve(r).then(function(){ return mjTypeset(); });
        return r;
      };
    }
  };
  document.addEventListener('DOMContentLoaded', function(){
    if (window.MathJax && MathJax.typesetPromise) mjTypeset();
  });
})();
</script>
"""


def main() -> None:
    if not TEMPLATE.is_file():
        raise SystemExit(f"Missing template: {TEMPLATE}")
    text = TEMPLATE.read_text(encoding="utf-8")
    if "</head>" not in text:
        raise SystemExit("template: no </head>")
    text = text.replace("</head>", HEAD_INJECT + "\n</head>", 1)
    if "</body>" not in text:
        raise SystemExit("template: no </body>")
    text = text.replace("</body>", BODY_SCRIPT + "\n</body>", 1)
    text = text.replace(
        "<title>[必填] 替换为 PPT 标题 · Deck Title</title>",
        "<title>基于三维高斯泼溅的多视图重建几何优化研究</title>",
    )

    deck_open = text.find("<div id=\"deck\">") + len("<div id=\"deck\">")
    deck_close = text.find("</div>\n\n<div id=\"nav\">")
    if deck_open < 0 or deck_close < 0:
        raise SystemExit("deck markers not found")

    slides = SLIDES
    out = text[:deck_open] + "\n" + slides + "\n" + text[deck_close:]
    INDEX.write_text(out, encoding="utf-8")
    print("Wrote", INDEX)


# 共 17 页，与原版 PPT 章节对应；字号在 HTML 内联中再次加大关键标题
SLIDES = r"""
<section class="slide accent" data-layout="SWISS-COVER-ASCII" data-animate="hero">
  <div class="canvas-card">
    <canvas class="ascii-bg" aria-hidden="true"></canvas>
    <div style="flex:1;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;padding:4vh 8vw;gap:2.8vh">
      <h1 data-anim="title" style="font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(5.4vw,10vh);line-height:1.06;letter-spacing:-.02em;color:#fff;margin:0">基于三维高斯泼溅的<br/>多视图重建几何优化研究</h1>
      <p data-anim="lead" class="lead" style="margin:0;max-width:56ch;color:rgba(255,255,255,.9);font-weight:300;font-size:max(18px,1.28vw);line-height:1.55">负责人：刘宇新 · 成员：朱正泽、孙浩成、黄宜恒 · 指导教师：秦红星 教授</p>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S19" data-animate="four-cards">
  <div class="canvas-card">
    <div class="chrome-min"><div class="l">汇报大纲</div><div class="r">02 / 17</div></div>
    <div style="flex:1;padding:0;display:flex;flex-direction:column;gap:1.6vh;min-height:0">
      <div data-anim="line" style="border-bottom:2px solid var(--accent);padding-bottom:1vh">
        <div class="t-meta">OUTLINE</div>
        <h2 class="h-xl-zh" style="font-size:min(4.2vw,7.5vh);font-weight:200;margin:.2em 0 0">内容结构</h2>
      </div>
      <div data-anim="up" style="flex:1;overflow:auto;padding-right:.5vw">
        <ol style="margin:0;padding-left:1.25em;font-size:max(17px,1.15vw);line-height:1.55;font-weight:300;color:var(--text-primary);display:flex;flex-direction:column;gap:1vh">
          <li><strong>背景介绍</strong> — 多视图几何重建与<strong>三维高斯泼溅（3D Gaussian Splatting, 3DGS）</strong>在新视图合成上的优势，及其与<strong>可交付三角网格</strong>之间的鸿沟。</li>
          <li><strong>数据处理与评估管线</strong> — 从多视图输入到位姿估计、辐射场优化、曲面提取及 DTU 定量评估；附<strong>自观测到指标的主链路顺序示意</strong>（第 4 页内嵌框图，非截图）。</li>
          <li><strong>对照方法</strong> — 基线 3DGS、SuGaR 类后处理、M1 训练内正则；<strong>第 5–7 页</strong>分述 A/B/C 流程条；<strong>第 8 页</strong>合并总损失式与各正则项定义。</li>
          <li><strong>实验结果</strong> — Chamfer 距离（overall / mean d2s / mean s2d）与 PSNR；并附顶点–真值距离上色的网格目视，区分<strong>几何修补有效</strong>与<strong>统计变化但目视不可分</strong>两类情形。</li>
          <li><strong>相关工作与启示</strong> — SuGaR、2DGS、广义高斯泼溅（GGGS）及<strong>单目深度先验、语义分割掩膜</strong>等可插拔先验。</li>
          <li><strong>后续工作</strong> — 原生 2DGS 曲面表示路径，或在 3DGS / GGGS 框架下改进透射率–表面定义（含 \(T=\frac12\) 阈值问题）。</li>
        </ol>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S04" data-animate="grid-reveal">
  <div class="canvas-card">
    <div class="chrome-min"><div class="l">背景介绍</div><div class="r">03 / 17</div></div>
    <div style="flex:1;padding:0;display:flex;flex-direction:column;gap:1.4vh">
      <div data-anim="line">
        <div class="t-meta">CONTEXT</div>
        <h2 class="h-xl-zh" style="font-size:min(4.5vw,8vh);font-weight:200">研究动机：合成质量与曲面可交付性</h2>
        <p style="margin:0 0 .55em;font-size:max(15px,1.05vw);color:var(--text-secondary);line-height:1.55;max-width:92ch">3DGS 通过显式高斯原语与可微光栅化获得快速收敛的新视图合成，但在<strong>屏幕空间不透明度 \(\alpha\)</strong>与深度估计欠约束区域，常出现<strong>体积式半透明堆积</strong>；经曲面提取后表现为飞丝、孔洞与虚假厚度，难以满足 CAD、检测或下游仿真对<strong>水密、可编辑三角网格</strong>的要求。</p>
        <p style="margin:0;font-size:max(15px,1.05vw);color:var(--text-secondary);line-height:1.55;max-width:92ch">本工作在<strong>不改变推理阶段光栅化主干</strong>的前提下，引入一组与占据、深度及法线相关的<strong>可微正则项</strong>，与光度项、多视图几何一致性项在训练阶段<strong>同步反传</strong>。评估采用 DTU 标准脚本：网格–真值间 Chamfer 距离（报告 overall、<strong>mean d2s</strong>：网格顶点→真值曲面，<strong>mean s2d</strong>：真值采样点→重建网格）及 PSNR；并以<strong>顶点着色网格</strong>（至真值 STL 的测地近似距离，统一 colormap 与 \(v_{\max}\)）作定性核对。</p>
      </div>
      <div data-anim="up" class="sub-grid-3-2">
        <article class="sub-card"><span class="nb-corner">1</span><h4 class="ttl">问题定位</h4><p class="desc">多视图稠密重建 → 显式辐射场 → <strong>可提取曲面</strong>与可复现误差度量。</p></article>
        <article class="sub-card accent"><span class="nb-corner">2</span><h4 class="ttl">3DGS 优势</h4><p class="desc">可微光栅、参数更新高效；损失多可写为<strong>屏幕空间可微泛函</strong>。</p></article>
        <article class="sub-card ink"><span class="nb-corner">3</span><h4 class="ttl">几何退化</h4><p class="desc">高 \(\alpha\) 堆叠、浮片、薄壁不连续；提取网格质量不稳定。</p></article>
        <article class="sub-card"><span class="nb-corner">4</span><h4 class="ttl">正则策略</h4><p class="desc">占据（Occ）、深度–不透明度耦合（Doc）、法线门控项（M1-B）及边缘加权支路；约束 \(\alpha\) 与深度梯度。</p></article>
        <article class="sub-card"><span class="nb-corner">5</span><h4 class="ttl">评估协议</h4><p class="desc">DTU：Chamfer 三分量 + PSNR；JSON 记录；顶点误差上色 PLY 与固定视角渲染作补充材料。</p></article>
        <article class="sub-card"><span class="nb-corner">6</span><h4 class="ttl">方法边界</h4><p class="desc">正则项改善训练景观，但受<strong>原语可表达性</strong>约束；后续需讨论表示层替换（如 2DGS）或透射率建模（GGGS）。</p></article>
      </div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S19" data-animate="four-cards">
  <div class="canvas-card">
    <div class="chrome-min"><div class="l">数据处理与评估管线</div><div class="r">04 / 17</div></div>
    <div style="flex:1;padding:0;display:flex;flex-direction:column;gap:0.65vh;min-height:0">
      <div data-anim="line" style="flex-shrink:0">
        <div class="t-meta">PIPELINE</div>
        <h2 class="h-xl-zh" style="font-size:min(3.85vw,6.9vh);font-weight:200;margin:.08em 0 .22em">多视图输入至定量评估</h2>
        <p style="margin:0 0 .45em;font-size:max(14px,1.02vw);color:var(--text-secondary);line-height:1.55;max-width:94ch">对照实验须固定<strong>训练/验证划分、迭代次数、初始化与随机种子、网格提取命令行及评估脚本版本</strong>。对齐网格 <code>recon_aligned.ply</code> 由 <code>evaluate_dtu_mesh.py</code> 生成，用于顶点–STL 距离上色；<code>recon_post.ply</code> 用于与相机系一致的渲染快照。</p>
        <p style="margin:0;font-size:max(14px,1.02vw);color:var(--text-secondary);line-height:1.52;max-width:94ch">以下为<strong>主链路依赖顺序</strong>的版面内嵌示意（框体与箭头，非整页截图）：视角为「观测—位姿—显式辐射场—可交付三角网格—评估落盘」，与<strong>第 5–7 页</strong>对照方法中各损失路径示意<strong>互补、不重复</strong>。</p>
      </div>
      <div data-anim="up" class="uggs-flow uggs-flow--large" style="flex:1;min-height:0;margin-top:.25em;display:flex;flex-direction:column;justify-content:space-evenly">
        <div class="uggs-flow-caption">主链路</div>
        <div class="uggs-flow-row" role="list">
          <div class="uggs-flow-box" role="listitem"><strong>输入</strong>多视图 RGB（+ 深度先验）</div>
          <span class="uggs-flow-arr" aria-hidden="true">→</span>
          <div class="uggs-flow-box" role="listitem"><strong>位姿</strong>COLMAP / 已知位姿</div>
          <span class="uggs-flow-arr" aria-hidden="true">→</span>
          <div class="uggs-flow-box" role="listitem"><strong>3DGS</strong>局部参数 + 损失优化</div>
          <span class="uggs-flow-arr" aria-hidden="true">→</span>
          <div class="uggs-flow-box" role="listitem"><strong>网格</strong>SuGaR / TSDU 等提取</div>
          <span class="uggs-flow-arr" aria-hidden="true">→</span>
          <div class="uggs-flow-box" role="listitem"><strong>指标</strong>三角网格 + JSON 指标</div>
        </div>
        <p class="uggs-flow-note">M1 各项在<strong>3DGS 训练优化</strong>内与 rgb / ncc / geo 等主损失<strong>同迭代、并行反传</strong>；<strong>不修改</strong>推理阶段可微光栅化的前馈路径。定性材料基于自编渲染资产、orbit 视轨与<strong>顶点距离上色</strong>的 PLY（MeshLab / CloudCompare 截屏），与 <code>evaluate_dtu_mesh.py</code> 对齐网格输出一致。</p>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S19" data-animate="four-cards">
  <div class="canvas-card">
    <div class="chrome-min"><div class="l">对照方法 · A</div><div class="r">05 / 17</div></div>
    <div style="flex:1;padding:0;display:flex;flex-direction:column;gap:0.55vh;min-height:0;justify-content:flex-start">
      <div data-anim="line" style="flex-shrink:0">
        <div class="t-meta">MODELS</div>
        <h2 class="h-xl-zh" style="font-size:min(3.85vw,6.9vh);font-weight:200;margin:.06em 0 .2em">基线 3DGS（对照）</h2>
        <p style="margin:0;font-size:max(14px,1.05vw);color:var(--text-secondary);line-height:1.52;max-width:94ch">在相同 DTU 场景与训练预算下，<strong>基线</strong>指光度项 + PatchMatch / 深度先验的 3DGS；网格提取与 DTU 评估脚本为<strong>参照零点</strong>，表中 \(\Delta\) 均相对该配置。</p>
        <p style="margin:.45em 0 0;font-size:max(13px,.95vw);color:var(--text-helper);line-height:1.45">B、C 路径见<strong>第 6–7 页</strong>；总损失与各正则项定义见<strong>第 8 页</strong>。</p>
      </div>
      <div data-anim="up" class="mjt uggs-flow-host" style="flex:1;display:flex;flex-direction:column;justify-content:stretch;min-height:0;margin-top:.2vh">
        <div class="uggs-flow uggs-models-strip uggs-models-strip--large">
          <div class="uggs-flow-caption">主损失（与第 8 页总式符号一致）</div>
          <div class="uggs-flow-row" role="list">
            <div class="uggs-flow-box uggs-flow-box--full" role="listitem">\(L = L_{\mathrm{rgb}} + \lambda_{\mathrm{dn}}L_{\mathrm{dn}} + \lambda_{\mathrm{ncc}}L_{\mathrm{ncc}} + \lambda_{\mathrm{geo}}L_{\mathrm{geo}}\)</div>
          </div>
          <div class="uggs-flow-row" role="list" style="margin-top:.4em">
            <div class="uggs-flow-box" role="listitem"><strong>多视图</strong>RGB</div>
            <span class="uggs-flow-arr" aria-hidden="true">→</span>
            <div class="uggs-flow-box" role="listitem"><strong>位姿</strong>COLMAP</div>
            <span class="uggs-flow-arr" aria-hidden="true">→</span>
            <div class="uggs-flow-box" role="listitem"><strong>可微</strong>光栅化</div>
            <span class="uggs-flow-arr" aria-hidden="true">→</span>
            <div class="uggs-flow-box" role="listitem"><strong>Adam</strong>反传</div>
            <span class="uggs-flow-arr" aria-hidden="true">→</span>
            <div class="uggs-flow-box" role="listitem"><strong>高斯</strong>\(\Theta\)</div>
          </div>
          <p class="uggs-flow-note">网格：marching / Poisson 等提取 → 三角网格 → <code>evaluate_dtu_mesh.py</code> → JSON 指标。</p>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S19" data-animate="four-cards">
  <div class="canvas-card">
    <div class="chrome-min"><div class="l">对照方法 · B</div><div class="r">06 / 17</div></div>
    <div style="flex:1;padding:0;display:flex;flex-direction:column;gap:0.55vh;min-height:0;justify-content:flex-start">
      <div data-anim="line" style="flex-shrink:0">
        <div class="t-meta">MODELS</div>
        <h2 class="h-xl-zh" style="font-size:min(3.85vw,6.9vh);font-weight:200;margin:.06em 0 .2em">SuGaR 类后处理（训练环外）</h2>
        <p style="margin:0;font-size:max(14px,1.05vw);color:var(--text-secondary);line-height:1.52;max-width:94ch">在<strong>冻结或微调</strong>高斯参数的前提下最小化表面对齐能量；对 \(\alpha\) 体积的约束为<strong>隐式</strong>，可与 M1 训练所得高斯场<strong>串联</strong>后再导出与评估。</p>
        <p style="margin:.45em 0 0;font-size:max(13px,.95vw);color:var(--text-helper)">基线流程见<strong>第 5 页</strong>；M1 训练内正则见<strong>第 7 页</strong>。</p>
      </div>
      <div data-anim="up" class="mjt uggs-flow-host" style="flex:1;display:flex;flex-direction:column;justify-content:stretch;min-height:0;margin-top:.2vh">
        <div class="uggs-flow uggs-models-strip uggs-models-strip--b uggs-models-strip--large">
          <div class="uggs-flow-caption">表面对齐目标（概念）</div>
          <div class="uggs-flow-row" role="list">
            <div class="uggs-flow-box uggs-flow-box--full" role="listitem">压缩高斯并绑定显式曲面 → 三角网格</div>
          </div>
          <div class="uggs-flow-row" role="list" style="margin-top:.4em">
            <div class="uggs-flow-box" role="listitem"><strong>已训练</strong>3DGS</div>
            <span class="uggs-flow-arr" aria-hidden="true">→</span>
            <div class="uggs-flow-box" role="listitem"><strong>密度场</strong>椭球支撑</div>
            <span class="uggs-flow-arr" aria-hidden="true">→</span>
            <div class="uggs-flow-box" role="listitem"><strong>表面对齐</strong>能量最小化</div>
            <span class="uggs-flow-arr" aria-hidden="true">→</span>
            <div class="uggs-flow-box" role="listitem"><strong>导出</strong>网格</div>
          </div>
          <p class="uggs-flow-note">导出后复用与基线相同的 DTU 评估脚本。</p>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S19" data-animate="four-cards">
  <div class="canvas-card">
    <div class="chrome-min"><div class="l">对照方法 · C</div><div class="r">07 / 17</div></div>
    <div style="flex:1;padding:0;display:flex;flex-direction:column;gap:0.55vh;min-height:0;justify-content:flex-start">
      <div data-anim="line" style="flex-shrink:0">
        <div class="t-meta">MODELS</div>
        <h2 class="h-xl-zh" style="font-size:min(3.85vw,6.9vh);font-weight:200;margin:.06em 0 .2em">M1 训练内正则</h2>
        <p style="margin:0;font-size:max(14px,1.05vw);color:var(--text-secondary);line-height:1.52;max-width:94ch"><strong>M1</strong> 指在训练中与主损失并行加入 Occ、Doc、rank、thin、法线门控（M1-B）等项的配置族；<strong>M1-3</strong> 为本组主消融（占据–深度耦合）。在<strong>同一反传步骤</strong>中叠加各项，<strong>不修改</strong>推理光栅化前馈路径。</p>
        <p style="margin:.45em 0 0;font-size:max(13px,.95vw);color:var(--text-helper)">基线与 SuGaR 路径见<strong>第 5–6 页</strong>；总式与逐项定义见<strong>第 8 页</strong>。</p>
      </div>
      <div data-anim="up" class="mjt uggs-flow-host" style="flex:1;display:flex;flex-direction:column;justify-content:stretch;min-height:0;margin-top:.2vh">
        <div class="uggs-flow uggs-models-strip uggs-models-strip--large">
          <div class="uggs-flow-caption">附加项求和（与主损失并行反传）</div>
          <div class="uggs-flow-row" role="list">
            <div class="uggs-flow-box uggs-flow-box--full" role="listitem">\(L_{\mathrm{extra}}=\lambda_{\mathrm{occ}}L_{\mathrm{occ}}+\lambda_{\mathrm{doc}}L_{\mathrm{doc}}+\lambda_{\mathrm{rank}}L_{\mathrm{rank}}+\lambda_{\mathrm{thin}}L_{\mathrm{thin}}+\lambda_{\mathrm{ng}}L_{\mathrm{ng}}\)</div>
          </div>
          <div class="uggs-flow-row" role="list" style="margin-top:.4em">
            <div class="uggs-flow-box" role="listitem"><strong>屏幕</strong>\(\alpha,\,d\)</div>
            <span class="uggs-flow-arr" aria-hidden="true">→</span>
            <div class="uggs-flow-box" role="listitem">\(L_{\mathrm{occ}}\)</div>
            <span class="uggs-flow-arr" aria-hidden="true">→</span>
            <div class="uggs-flow-box" role="listitem">\(L_{\mathrm{doc}}\)</div>
            <span class="uggs-flow-arr" aria-hidden="true">→</span>
            <div class="uggs-flow-box" role="listitem"><strong>rank</strong> / <strong>thin</strong></div>
            <span class="uggs-flow-arr" aria-hidden="true">→</span>
            <div class="uggs-flow-box" role="listitem"><strong>M1-B</strong><br>\(L_{\mathrm{ng}}\)</div>
            <span class="uggs-flow-arr" aria-hidden="true">→</span>
            <div class="uggs-flow-box" role="listitem"><strong>并入</strong>主损失梯度</div>
          </div>
          <p class="uggs-flow-note">与 \(L_{\mathrm{rgb}}\)、\(L_{\mathrm{ncc}}\)、\(L_{\mathrm{geo}}\) 共用 Adam 反传；推理路径不变。</p>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S04" data-animate="grid-reveal">
  <div class="canvas-card">
    <div class="chrome-min"><div class="l">M1 总式与各项定义</div><div class="r">08 / 17</div></div>
    <div style="flex:1;padding:0;display:flex;flex-direction:column;gap:0.9vh;min-height:0;overflow:auto">
      <div data-anim="line" style="flex-shrink:0">
        <div class="t-meta">M1</div>
        <h2 class="h-xl-zh" style="font-size:min(3.75vw,6.8vh);font-weight:200;margin:.08em 0 .2em">总损失与正则项</h2>
        <p style="margin:0;font-size:max(14px,1.02vw);color:var(--text-secondary);line-height:1.5;max-width:94ch">下列为训练阶段<strong>总目标</strong>及各项在屏幕空间或可微渲染链上的含义；各 \(\lambda\) 与消融组合见实验 JSON。对照方法流程条见<strong>第 5–7 页</strong>。</p>
      </div>
      <div data-anim="up" class="mjt" style="flex-shrink:0;border-bottom:1px dashed var(--border-subtle);padding-bottom:.65em;margin-bottom:.15em">
        $$L = L_{\mathrm{rgb}}+\lambda_{\mathrm{dn}}L_{\mathrm{dn}}+\lambda_{\mathrm{ncc}}L_{\mathrm{ncc}}+\lambda_{\mathrm{geo}}L_{\mathrm{geo}}+\lambda_{\mathrm{occ}}L_{\mathrm{occ}}+\lambda_{\mathrm{doc}}L_{\mathrm{doc}}+\lambda_{\mathrm{rank}}L_{\mathrm{rank}}+\lambda_{\mathrm{thin}}L_{\mathrm{thin}}+\lambda_{\mathrm{ng}}L_{\mathrm{ng}}$$
        <ul class="uggs-bullets" style="margin-top:.5em;font-size:max(14px,1.03vw)">
          <li>\(L_{\mathrm{occ}}\)：驱动 \(\alpha\) 远离 \(\frac12\) 附近的模糊平台。</li>
          <li>\(L_{\mathrm{doc}}\)：\(\nabla_{x,y}\tilde d\) 与 \(\nabla_{x,y}\alpha\) 在图像平面上一致，固定遮挡边界。</li>
          <li>\(L_{\mathrm{rank}},L_{\mathrm{thin}}\)：射线一致性及薄结构先验（启用时）。</li>
          <li>\(L_{\mathrm{ng}}\)：门控权重 \(e^{-\beta e}\) 下惩罚 \(4\alpha(1-\alpha)\)，\(e\) 为法线–深度一致性标量；\(\beta\) 控制门控硬度。</li>
        </ul>
      </div>
      <div data-anim="up" class="sub-grid-3-2" style="margin-top:.2vh">
        <article class="sub-card"><span class="nb-corner">1</span><h4 class="ttl">\(L_{\mathrm{occ}}\)</h4><p class="desc"><strong>占据正则</strong>：惩罚屏幕空间 \(\alpha\) 的熵及高不透明度像素占比，抑制无支撑半透明层。</p></article>
        <article class="sub-card"><span class="nb-corner">2</span><h4 class="ttl">\(L_{\mathrm{doc}}\)</h4><p class="desc"><strong>深度–不透明度耦合</strong>：令渲染深度 \(\tilde d\) 与 \(\alpha\) 在图像平面上的梯度方向一致。</p></article>
        <article class="sub-card accent"><span class="nb-corner">3</span><h4 class="ttl">M1-3（主消融）</h4><p class="desc">Occ + Doc（及配置内其它固定项）在 DTU 三场景上取得<strong>平均 Chamfer 最优</strong>，见<strong>第 9 页</strong>表。</p></article>
        <article class="sub-card"><span class="nb-corner">4</span><h4 class="ttl">\(L_{\mathrm{rank}},L_{\mathrm{thin}}\)</h4><p class="desc"><strong>射线排序 / 薄结构</strong>辅助项；仅在特定场景启用，见配置文件。</p></article>
        <article class="sub-card"><span class="nb-corner">B</span><h4 class="ttl">\(L_{\mathrm{ng}}\)（M1-B）</h4><p class="desc">法线门控下对 \(4\alpha(1-\alpha)\) 的惩罚；须监控 <strong>mean s2d</strong> 以防网格外推。单独调节 \(\lambda_{\mathrm{ng}}\) 与 \(\beta\)。</p></article>
        <article class="sub-card ink"><span class="nb-corner">+</span><h4 class="ttl">边缘支路</h4><p class="desc"><code>edge_dn</code>、<code>pm_edge_soft</code> 等；<strong>与主表结论解耦</strong>，仅作探索性记录。</p></article>
      </div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S04" data-animate="grid-reveal">
  <div class="canvas-card">
    <div class="chrome-min"><div class="l">尝试结果 · M1-3</div><div class="r">09 / 17</div></div>
    <div style="flex:1;padding:0;display:flex;flex-direction:column;gap:1vh;min-height:0">
      <div data-anim="line" style="flex-shrink:0">
        <div class="t-meta">DTU · Δ vs 基线</div>
        <h2 class="h-xl-zh" style="font-size:min(3.8vw,6.8vh);font-weight:200">三场景 Chamfer 与 PSNR</h2>
        <p style="margin:.4em 0 0;font-size:max(14px,1.02vw);color:var(--text-secondary);line-height:1.52;max-width:92ch">表中 \(\Delta\) 为相对基线的<strong>差值</strong>；Chamfer 为<strong>有符号改进</strong>时取负号。量级约 \(10^{-4}\) 时，须结合<strong>第 10–11 页</strong>目视材料阅读，避免仅凭 overall 判断局部几何。</p>
      </div>
      <div data-anim="up" style="flex:1;overflow:auto">
        <table class="uggs-table">
          <thead><tr><th>场景</th><th>Δ overall</th><th>Δ mean_d2s</th><th>Δ mean_s2d</th><th>Δ PSNR</th></tr></thead>
          <tbody>
            <tr><td>scan24</td><td>\(-8.5\times10^{-4}\)</td><td>\(-9.1\times10^{-4}\)</td><td>\(-7.9\times10^{-4}\)</td><td>\(+0.15\)</td></tr>
            <tr><td>scan105</td><td>\(-7.2\times10^{-4}\)</td><td>\(-7.8\times10^{-4}\)</td><td>\(-6.6\times10^{-4}\)</td><td>\(+0.12\)</td></tr>
            <tr><td>scan122</td><td>\(-7.8\times10^{-4}\)</td><td>\(-8.3\times10^{-4}\)</td><td>\(-7.3\times10^{-4}\)</td><td>\(+0.18\)</td></tr>
            <tr><td><strong>平均</strong></td><td>\(\mathbf{-7.83\times10^{-4}}\)</td><td>\(-8.4\times10^{-4}\)</td><td>\(-7.27\times10^{-4}\)</td><td>\(+0.15\)</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S22" data-animate="image-hero">
  <div class="canvas-card uggs-s22-compact" style="padding:0;display:flex;flex-direction:column;overflow:hidden;min-height:0">
    <div data-anim="img" style="position:relative;flex:0 0 44%;min-height:22vh;max-height:48vh;overflow:hidden;background:var(--grey-0)">
      <img src="images/deck_effect_hole_patch_triptych.png" alt="顶点距离上色网格：三例局部修补（横向拼接）" data-image-slot="s22-hero-21x9" loading="lazy" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;object-position:center center">
    </div>
    <div data-anim="kpi" class="image-hero-body uggs-s22-body" style="flex:1;min-height:0;overflow:auto;display:flex;flex-direction:column;padding:1.2vh 3.2vw 2vh;justify-content:flex-start;gap:1.35vh">
      <div class="chrome-min" style="margin-bottom:0"><div class="l">目视 · 顶点误差上色（有效修补）</div><div class="r">10 / 17</div></div>
      <div style="max-width:62ch;font-size:max(14px,1.05vw);line-height:1.55;color:var(--text-primary);font-weight:300;margin:0">
        颜色表示对齐网格顶点到 DTU 真值 STL 的测地近似距离（与评估脚本一致，<strong>统一 colormap 与 \(v_{\max}\)</strong>）。左→右三例为答辩素材原图拼接：<strong>(1)</strong> scan65，阈值化边缘 + basis smooth 后，大空洞周边误差分布重分配；<strong>(2)</strong> scan65，<code>edge_to_fix</code> 与基线对照，可见镂空区域新增三角片；<strong>(3)</strong> scan24 与 scan65 在相同上色脚本下的局部对照。上述为<strong>定性正例</strong>；统计显著性仍以<strong>第 9 页</strong>表为准。</div>
      <div class="image-hero-stats uggs-s22-stats" style="gap:2.2vw;margin-top:.2vh;padding-top:.5vh">
        <div style="display:flex;flex-direction:column;gap:.5vh"><div style="height:1px;background:var(--ink)"></div><div class="t-meta">(1)</div><div style="font-family:var(--sans);font-weight:500;font-size:max(15px,1.05vw);line-height:1.35">scan65 阈值+smooth</div><div style="height:1px;background:var(--border-subtle);margin-top:.35vh"></div><p class="body-sm" style="margin:0;font-size:max(14px,.98vw)">误差热力重分配</p></div>
        <div style="display:flex;flex-direction:column;gap:.5vh"><div style="height:1px;background:var(--ink)"></div><div class="t-meta">(2)</div><div style="font-family:var(--sans);font-weight:500;font-size:max(15px,1.05vw);line-height:1.35">fix vs baseline</div><div style="height:1px;background:var(--border-subtle);margin-top:.35vh"></div><p class="body-sm" style="margin:0;font-size:max(14px,.98vw)">补片与噪声权衡</p></div>
        <div style="display:flex;flex-direction:column;gap:.5vh"><div style="height:1px;background:var(--ink)"></div><div class="t-meta">(3)</div><div style="font-family:var(--sans);font-weight:500;font-size:max(15px,1.05vw);line-height:1.35">scan24 ∥ scan65</div><div style="height:1px;background:var(--border-subtle);margin-top:.35vh"></div><p class="body-sm" style="margin:0;font-size:max(14px,.98vw)">跨场景对照</p></div>
      </div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S22" data-animate="image-hero">
  <div class="canvas-card uggs-s22-compact" style="padding:0;display:flex;flex-direction:column;overflow:hidden;min-height:0">
    <div data-anim="img" style="position:relative;flex:0 0 44%;min-height:22vh;max-height:48vh;overflow:hidden;background:var(--grey-0)">
      <img src="images/deck_no_effect_scan65_teddy_twin.png" alt="scan65：两视窗顶点上色对照，目视几乎无差异" data-image-slot="s22-hero-21x9" loading="lazy" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;object-position:center center">
    </div>
    <div data-anim="kpi" class="image-hero-body uggs-s22-body" style="flex:1;min-height:0;overflow:auto;display:flex;flex-direction:column;padding:1.2vh 3.2vw 2vh;justify-content:flex-start;gap:1.35vh">
      <div class="chrome-min" style="margin-bottom:0"><div class="l">目视 · 统计变化但不可分（反例）</div><div class="r">11 / 17</div></div>
      <div style="max-width:62ch;font-size:max(14px,1.05vw);line-height:1.55;color:var(--text-primary);font-weight:300;margin:0">
        <strong>与第 10 页对读。</strong>同一 scan65 场景、同一上色尺度下，两视窗为不同分支输出；<strong>热力与外形在目视上几乎重合</strong>，说明该分支对顶点误差分布的改善<strong>不足以形成可汇报的定性差异</strong>。若表中仍存在小幅 \(\Delta\)，须在文中明确标注为<strong>非主要结论</strong>，与 M1-3 主消融区分。</div>
      <div class="image-hero-stats uggs-s22-stats" style="gap:2.2vw;margin-top:.2vh;padding-top:.5vh">
        <div style="display:flex;flex-direction:column;gap:.5vh"><div style="height:1px;background:var(--ink)"></div><div class="t-meta">数据</div><div style="font-family:var(--sans);font-weight:500;font-size:max(15px,1.05vw)">scan65</div><div style="height:1px;background:var(--border-subtle);margin-top:.35vh"></div><p class="body-sm" style="margin:0;font-size:max(14px,.98vw)">上色 PLY 截屏</p></div>
        <div style="display:flex;flex-direction:column;gap:.5vh"><div style="height:1px;background:var(--ink)"></div><div class="t-meta">判读</div><div style="font-family:var(--sans);font-weight:500;font-size:max(15px,1.05vw)">目视不可分</div><div style="height:1px;background:var(--border-subtle);margin-top:.35vh"></div><p class="body-sm" style="margin:0;font-size:max(14px,.98vw)">须结合表与局部放大</p></div>
        <div style="display:flex;flex-direction:column;gap:.5vh"><div style="height:1px;background:var(--ink)"></div><div class="t-meta">写作</div><div style="font-family:var(--sans);font-weight:500;font-size:max(15px,1.05vw)">单独讨论</div><div style="height:1px;background:var(--border-subtle);margin-top:.35vh"></div><p class="body-sm" style="margin:0;font-size:max(14px,.98vw)">不得并入主结论句</p></div>
      </div>
    </div>
  </div>
</section>

<section class="slide grey" data-layout="S08" data-animate="duo-mirror">
  <div class="canvas-card">
    <div class="chrome-min"><div class="l">尝试结果 · M1-B</div><div class="r">12 / 17</div></div>
    <div style="flex:1;padding:0;display:flex;flex-direction:column;gap:0.45vh" class="uggs-duo-tight">
      <div data-anim="line">
        <div class="t-meta">TRADE-OFF</div>
        <h2 class="h-xl-zh" style="font-size:min(4vw,7.2vh);font-weight:200">法线门控与 mean s2d</h2>
        <p style="margin:.4em 0 0;font-size:max(14px,1.02vw);color:var(--text-secondary);line-height:1.52;max-width:92ch">当法线–深度估计一致时，\(L_{\mathrm{ng}}\) 可抑制薄片状半透明几何；若深度噪声显著或 \(\lambda_{\mathrm{ng}}\) 过大，优化器可能通过<strong>外推网格</strong>降低渲染误差，表现为 <strong>mean s2d</strong>（真值点到重建面）恶化。<strong>第 10–11 页</strong>说明：目视与标量须<strong>联合判读</strong>。</p>
      </div>
      <div class="duo-compare">
        <div class="col accent">
          <div class="col-tag"><span class="num">+</span> scan24</div>
          <h2 class="col-ttl">三向改进</h2>
          <p class="col-desc">overall、mean d2s、mean s2d 同时优于基线；与法线–深度一致性假设相符。</p>
          <ul class="col-list"><li>PSNR 不降</li><li>\(\lambda_{\mathrm{ng}},\beta\) 在验证集上调参</li></ul>
        </div>
        <span class="vrule"></span>
        <div class="col">
          <div class="col-tag"><span class="num">!</span> scan105 / 122</div>
          <h2 class="col-ttl">s2d 风险</h2>
          <p class="col-desc">mean s2d 相对基线变差；需减小 \(\lambda_{\mathrm{ng}}\) 或分阶段启用边缘支路。</p>
          <ul class="col-list"><li>先固定 M1-3</li><li>再以 M1-B 精调</li></ul>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S19" data-animate="four-cards">
  <div class="canvas-card">
    <div class="chrome-min"><div class="l">相关工作与启示</div><div class="r">13 / 17</div></div>
    <div style="flex:1;padding:0;display:flex;flex-direction:column;gap:1.2vh">
      <div data-anim="line" style="flex-shrink:0">
        <div style="height:2px;background:var(--accent);width:100%"></div>
        <h2 class="h-xl-zh" style="font-size:min(3.8vw,6.8vh);font-weight:200">表示、后处理与可插先验</h2>
        <p style="margin:.35em 0 0;font-size:max(14px,1.02vw);color:var(--text-secondary);line-height:1.52;max-width:94ch">下列方向与当前 3DGS+M1 管线<strong>正交可叠加</strong>：可在保持评估协议不变的前提下，引入单目深度网络输出的伪深度图作<strong>尺度对齐后的深度一致性正则</strong>；或使用语义分割网络生成掩膜，在天空及远景区域<strong>屏蔽几何监督</strong>以降低虚假薄壳。</p>
      </div>
      <div data-anim="up" style="display:grid;grid-template-columns:repeat(2,1fr);gap:1.1vw;flex:1;align-content:start">
        <div class="card-fill" style="padding:1.6vh 1.1vw">
          <span class="t-meta">— SuGaR / M1</span>
          <h3 style="font-size:max(16px,1.35vw);margin:.3em 0">后处理 vs 训练内</h3>
          <p style="font-size:max(14px,1.02vw);color:var(--text-secondary);line-height:1.52">SuGaR 系方法在收敛场上优化表面对齐；M1 在训练中直接塑形 \(\alpha\) 与深度梯度。二者可<strong>顺序组合</strong>，但须在文中分别报告贡献。</p>
        </div>
        <div class="card-fill" style="padding:1.6vh 1.1vw">
          <span class="t-meta">— 2DGS / GGGS</span>
          <h3 style="font-size:max(16px,1.35vw);margin:.3em 0">表示层差异</h3>
          <p style="font-size:max(14px,1.02vw);color:var(--text-secondary);line-height:1.52">2DGS 以盘面逼近薄层几何；GGGS 在连续透射率 \(T(t)\) 上定义中值表面。二者与 3DGS 的误差结构不同，适合作为<strong>受控对照实验</strong>。</p>
        </div>
        <div class="card-fill" style="padding:1.6vh 1.1vw">
          <span class="t-meta">— 单目深度</span>
          <h3 style="font-size:max(16px,1.35vw);margin:.3em 0">Depth Anything 等</h3>
          <p style="font-size:max(14px,1.02vw);color:var(--text-secondary);line-height:1.52">预训练单目深度经仿射对齐后，与渲染深度构造 \(L_2\) 或梯度一致性项，可缓解弱纹理区域梯度消失。</p>
        </div>
        <div class="card-fill" style="padding:1.6vh 1.1vw">
          <span class="t-meta">— SAM</span>
          <h3 style="font-size:max(16px,1.35vw);margin:.3em 0">语义掩膜</h3>
          <p style="font-size:max(14px,1.02vw);color:var(--text-secondary);line-height:1.52">对天空、远景类别施加掩膜，可在训练或提取阶段抑制<strong>无真实几何支撑</strong>的薄片。</p>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S19" data-animate="four-cards">
  <div class="canvas-card">
    <div class="chrome-min"><div class="l">后续工作 · 路径 A</div><div class="r">14 / 17</div></div>
    <div style="flex:1;padding:0;display:flex;flex-direction:column;gap:0.75vh;min-height:0;overflow:hidden">
      <div data-anim="line" style="flex-shrink:0">
        <div style="height:2px;background:var(--accent);width:100%"></div>
        <h2 class="h-xl-zh" style="font-size:min(3.65vw,6.5vh);font-weight:200;margin:.06em 0 .12em">二维高斯泼溅（2DGS）曲面表示</h2>
        <p style="margin:.25em 0 0;font-size:max(15px,1.08vw);color:var(--text-secondary);line-height:1.55;max-width:94ch">以定向<strong>2D 盘面元</strong>逼近场景薄层，光栅化沿射线对盘面积分；常改善初始网格质量。若将外观简化为<strong>叠置带色盘元</strong>而忽略镜面与折射的物理约束，提取阶段仍可能出现<strong>伪高频几何</strong>。须与当前 3DGS 管线在<strong>相同数据与评估脚本</strong>下对照。</p>
      </div>
      <div data-anim="up" class="uggs-2dgs-split" style="flex:1;display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,0.95fr);gap:1.2vw;min-height:0;align-items:stretch;padding-top:.35vh">
        <div style="display:flex;flex-direction:column;gap:1.1vh;min-height:0;overflow:auto">
          <ul class="uggs-bullets uggs-2dgs-bullets" style="margin:0;font-size:max(15px,1.08vw);line-height:1.62">
            <li><strong>优化噪声</strong>：盘面法向与深度在平滑区抖动 → 网格假粗糙。</li>
            <li><strong>镜面</strong>：高光被分解为半透明上层 → 法线场污染、纹理烘焙失真。</li>
            <li><strong>弱纹理</strong>：盘面定位退化 → 孔洞或层状薄壳。</li>
            <li><strong>透明介质</strong>：单层盘面难以同时表达反射与折射界面。</li>
          </ul>
        </div>
        <figure class="uggs-2dgs-fig" style="margin:0;display:flex;flex-direction:column;align-items:center;justify-content:stretch;background:var(--grey-0);border:1px solid var(--border-subtle);border-radius:6px;padding:1vh .8vw 1vh;min-height:min(52vh,100%)">
          <div style="flex:1;width:100%;display:flex;align-items:center;justify-content:center;min-height:36vh">
            <svg viewBox="0 0 220 320" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" style="width:100%;max-width:280px;height:auto;max-height:54vh">
              <defs>
                <linearGradient id="uggs2dgsG" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#0b57d0" stop-opacity=".35"/><stop offset="100%" stop-color="#0b57d0" stop-opacity=".08"/></linearGradient>
              </defs>
              <line x1="110" y1="280" x2="110" y2="40" stroke="#bbb" stroke-width="1.2" stroke-dasharray="4 3"/>
              <ellipse cx="110" cy="210" rx="72" ry="14" fill="url(#uggs2dgsG)" stroke="#0b57d0" stroke-width="1.2" transform="rotate(-8 110 210)"/>
              <ellipse cx="110" cy="150" rx="68" ry="13" fill="#e8eef8" stroke="#0b57d0" stroke-width="1" transform="rotate(6 110 150)"/>
              <ellipse cx="110" cy="95" rx="64" ry="12" fill="#f4f4f2" stroke="#333" stroke-width="1" transform="rotate(-4 110 95)"/>
              <circle cx="110" cy="52" r="4" fill="var(--accent)"/>
            </svg>
          </div>
          <figcaption style="margin:0;padding:.5em .4em 0;font-size:max(12px,.85vw);color:var(--text-helper);line-height:1.45;text-align:center;border-top:1px dashed var(--border-subtle);width:100%">原理示意：射线与多层<strong>定向椭盘</strong>求交并累积颜色/不透明度（无文字标注，说明见左栏）。</figcaption>
        </figure>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S08" data-animate="duo-mirror">
  <div class="canvas-card">
    <div class="chrome-min"><div class="l">后续工作 · 路径 B</div><div class="r">15 / 17</div></div>
    <div style="flex:1;padding:0;display:flex;flex-direction:column;gap:0.45vh" class="uggs-duo-tight">
      <div data-anim="line">
        <div class="t-meta">GGGS / 透射率表面</div>
        <h2 class="h-xl-zh" style="font-size:min(3.85vw,6.9vh);font-weight:200">中值透射率与界面定位</h2>
        <p style="margin:.4em 0 0;font-size:max(14px,1.02vw);color:var(--text-secondary);line-height:1.55;max-width:92ch">在连续透射率 \(T(t)\) 沿射线的模型中，常取满足 \(T(t^\ast)=\frac12\) 的 \(t^\ast\) 作为表面深度。对玻璃等多层透射界面，该定义可能<strong>合并或跳过</strong>真实界面，需在透射率场上增加<strong>平滑约束、自适应阈值</strong>或检测<strong>密度次峰</strong>以定位内层界面。</p>
      </div>
      <div class="duo-compare">
        <div class="col accent">
          <div class="col-tag"><span class="num">问题</span> \(T=\frac12\)</div>
          <h2 class="col-ttl">多层 / 玻璃</h2>
          <p class="col-desc">单一等值面难以表达多个光学界面。</p>
          <ul class="col-list"><li>学习化阈值或分段 \(T\)</li><li>多假设表面</li></ul>
        </div>
        <span class="vrule"></span>
        <div class="col">
          <div class="col-tag"><span class="num">结合</span> M1</div>
          <h2 class="col-ttl">互补思路</h2>
          <p class="col-desc">在射线域对透射率施加 TV 或稀疏先验；与占据正则协同抑制假厚度。</p>
          <ul class="col-list"><li>椭球密度峰检测</li><li>深度 / 分割先验</li></ul>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="slide" data-layout="S11" data-animate="timeline-walk">
  <div class="canvas-card">
    <div class="chrome-min"><div class="l">工作计划</div><div class="r">16 / 17</div></div>
    <div style="flex:1;padding:0;display:flex;flex-direction:column;gap:1vh">
      <div data-anim="line">
        <div class="t-meta">PLAN</div>
        <h2 class="h-xl-zh" style="font-size:min(3.8vw,6.8vh);font-weight:200">先验插入与表示路线分叉</h2>
        <p style="margin:.4em 0 0;font-size:max(15px,1.08vw);color:var(--text-secondary);line-height:1.52">近期：单目深度一致性 + 语义掩膜；中期：在固定评估协议下比较 <strong>2DGS</strong> 与 <strong>3DGS+GGGS</strong>；文稿中保持<strong>表与顶点上色图同页引用</strong>，避免孤立数值。</p>
      </div>
      <div class="timeline-h uggs-timeline-lg" style="margin-top:0.9vh;flex:1">
        <div class="tl-row">
          <div class="th-node up"><span class="label"><span class="yr">近期</span><span class="name">深度 + 掩膜</span><span class="desc">弱纹理 / 天空</span></span><span class="dot"></span></div>
          <div class="th-node down"><span class="label"><span class="yr">分支 A</span><span class="name">2DGS</span><span class="desc">曲面对照</span></span><span class="dot"></span></div>
          <div class="th-node up accent"><span class="label"><span class="yr">分支 B</span><span class="name">GGGS</span><span class="desc">\(T\) 与界面</span></span><span class="dot"></span></div>
          <div class="th-node down"><span class="label"><span class="yr">写作</span><span class="name">消融</span><span class="desc">JSON 归档</span></span><span class="dot"></span></div>
          <div class="th-node up"><span class="label"><span class="yr">交付</span><span class="name">材料</span><span class="desc">图表一致</span></span><span class="dot"></span></div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="slide split" data-layout="SWISS-CLOSING-ASCII" data-animate="split-statement">
  <div class="canvas-card">
    <div class="split-half">
      <div class="half b-accent" style="padding:5.6vh 3.6vw 4.4vh;justify-content:space-between;position:relative;overflow:hidden">
        <canvas class="ascii-bg" aria-hidden="true"></canvas>
        <div class="chrome-min" style="margin-bottom:0;position:relative;z-index:1"><div class="l">17 / 17</div><div class="r">致谢</div></div>
        <div data-anim="manifesto" style="display:flex;flex-direction:column;gap:2vh;position:relative;z-index:1">
          <div class="t-meta" style="color:rgba(255,255,255,.78);letter-spacing:.2em">THANKS</div>
          <h2 style="font-family:var(--sans),var(--sans-zh);font-size:min(5.2vw,9.5vh);line-height:1.05;font-weight:200;color:#fff">谢谢各位老师</h2>
          <p style="font-size:max(16px,1.12vw);line-height:1.55;color:rgba(255,255,255,.86);max-width:40ch">指导老师：秦红星 教授</p>
        </div>
        <div data-anim="signature" style="display:flex;justify-content:space-between;align-items:end;border-top:1px solid rgba(255,255,255,.22);padding-top:2vh;position:relative;z-index:1">
          <div class="t-meta" style="color:rgba(255,255,255,.65)">刘宇新 · 朱正泽 · 孙浩成 · 黄宜恒</div>
          <div class="t-meta" style="color:rgba(255,255,255,.65)">Q &amp; A</div>
        </div>
      </div>
      <div class="half" style="padding:5.6vh 3.6vw 4.4vh;justify-content:space-between">
        <div class="chrome-min"><div class="l">要点回顾</div><div class="r">3</div></div>
        <div data-anim="rules" style="display:flex;flex-direction:column;gap:0">
          <div style="display:grid;grid-template-columns:auto 1fr;gap:2vw;padding:2vh 0;border-top:1px solid var(--border-subtle)">
            <div style="font-family:var(--sans);font-weight:200;font-size:min(3.8vw,6.5vh)">1</div>
            <div><p style="font-size:max(15px,1.05vw);color:var(--text-secondary);line-height:1.5;margin:0">M1-3：DTU 平均 Chamfer 相对基线改进约 \(10^{-4}\)；须与第 10–11 页<strong>顶点上色目视</strong>联合陈述。</p></div>
          </div>
          <div style="display:grid;grid-template-columns:auto 1fr;gap:2vw;padding:2vh 0;border-top:1px solid var(--border-subtle)">
            <div style="font-family:var(--sans);font-weight:200;font-size:min(3.8vw,6.5vh)">2</div>
            <div><p style="font-size:max(15px,1.05vw);color:var(--text-secondary);line-height:1.5;margin:0">对照方法流程条分<strong>第 5–7 页</strong>；总式与逐项定义见<strong>第 8 页</strong>；避免与第 4 页主链路示意重复插图。</p></div>
          </div>
          <div style="display:grid;grid-template-columns:auto 1fr;gap:2vw;padding:2vh 0;border-top:1px solid var(--border-subtle);border-bottom:2px solid var(--accent)">
            <div style="font-family:var(--sans);font-weight:200;font-size:min(3.8vw,6.5vh);color:var(--accent)">3</div>
            <div><p style="font-size:max(15px,1.05vw);color:var(--text-secondary);line-height:1.5;margin:0">后续：单目深度与语义掩膜先验；2DGS 或 GGGS 透射率建模二选一深入。</p></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>
"""


if __name__ == "__main__":
    main()
