# 中期汇报 · 瑞士风网页 PPT（guizang-ppt-skill）

本目录由 **[guizang-ppt-skill](https://github.com/op7418/guizang-ppt-skill)** 的 **风格 B（`template-swiss.html`）** 生成：**单文件 `index.html`**，横向翻页，含 WebGL 网格背景与 Motion One 动效。

## 如何打开

- 用 **Chrome / Edge** 直接双击打开 `index.html`，或拖到浏览器。
- **← / →** 翻页，**滚轮** 翻页，**ESC** 缩略索引，**B** 低功耗（关 WebGL 动画）。

## 内容与版式

当前 `index.html` 共 **17 页**（内容由 `_deck_slides_body.txt` 经 `apply_deck_slides.py` 写入 `build_deck.py` 后生成）。`data-layout` 按 Swiss locked 规则标注。主题色为模板默认 **IKB 克莱因蓝**。

## 重新生成

修改 `_deck_slides_body.txt` 后执行 `py -3 apply_deck_slides.py`，再执行：

```bash
py -3 presentations/uggs-m1-guizang/build_deck.py
node tools/guizang-ppt-skill/scripts/validate-swiss-deck.mjs presentations/uggs-m1-guizang/index.html
```

（需已克隆 `tools/guizang-ppt-skill` 仓库。）

## 导出 PowerPoint / 保留动效

**结论（技术）**：Motion One、WebGL 背景等与 **PowerPoint 原生动画（OOXML 时间轴）不是同一套模型**，无法自动「无损翻译」为 PPT 里可编辑的逐形状动画。

本目录提供脚本 **`export_swiss_deck_to_pptx.py`**，两种产物二选一或同时使用：

1. **`.pptx`（默认）**：用无头 Chromium 打开本地 `index.html`，对每一页在「翻页过渡 + 入场动效」结束后截 **整视口 PNG**，再写入 16:9 空白版式。**每张幻灯片是一张位图**，观感接近最终帧，但 **没有** PPT 内置的「飞入 / 渐显」动画轨道。
2. **`--video-out deck.webm`（可选）**：同一次浏览过程 **整段录屏**（Playwright），**页内动效会留在视频里**；可在 PowerPoint 中「插入 → 视频」作为补充，或单独播放。

依赖（任选其一安装方式；**国内网络**可加上 ``-i https://pypi.tuna.tsinghua.edu.cn/simple``）：

```bash
py -3 -m pip install -r requirements-deck-export.txt
```

脚本会**优先使用本机已安装的 Edge / Chrome**（Playwright ``channel``），一般**不必**再下载约 180 MB 的 Playwright Chromium。若使用 ``--video-out`` 录 WebM，还需安装 FFmpeg 组件（体积约 1 MB）：

```bash
py -3 -m playwright install ffmpeg
```

若提示 ``__dirlock`` 锁文件，先结束其它正在安装的 Playwright 进程，或删除该锁目录后再执行 ``playwright install ffmpeg``。

示例：

```bash
cd unified_geometry_gs/presentations/uggs-m1-guizang
py -3 build_deck.py
py -3 export_swiss_deck_to_pptx.py --out deck_export.pptx
py -3 export_swiss_deck_to_pptx.py --out deck_export.pptx --video-out deck_full.webm
```

若导出时提示 **Permission denied**，请先关闭 PowerPoint 中已打开的同名 `.pptx`，或改用 `--out deck_export_new.pptx` 等新文件名。

若公式或大图仍未渲染完，可加大等待：`--wait-ms 5000`。

模板中已暴露 `window.__exportGoToSlide(n)`（与点击底部圆点等价）；**重新 `build_deck.py` 后** `index.html` 才会带上该钩子。

## 说明

- 主交付物仍是 **单文件网页幻灯片**；会议现场优先 **浏览器全屏** 以保留全部动效与交互。
- 若仅需静态分发，除上述 PPTX 外，亦可用浏览器 **打印 → 另存为 PDF**。
- Skill 未作为 Cursor 内置 Skill 安装；模板来自仓库 `tools/guizang-ppt-skill/assets/template-swiss.html`。
