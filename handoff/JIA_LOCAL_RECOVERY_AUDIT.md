# 甲：本地资产恢复查找回执

2026-11-16T15:36:05.549910+00:00。只读查找。未批量解压、未清空/自动还原回收站、未加载不明 pickle、未重复下载 90 卷。

## 结论

**NO_TRUSTED_ORIGINAL_GAUSSIAN_FOUND_IN_BOUNDED_LOCAL_SEARCH**

write rebuild protocol; new paired experiment; do not use old trunc4 JSON as control

区分：旧 `ggs_ex_trunc4/*/vis/results.json` 与 `recon*.ply` **仍在**（旧结果存在）；对应 `point_cloud/iteration_30000` **本地与已查备份中未找到**（原模型当前不可恢复）。

## 搜索覆盖

- `/mnt/c/Users/Someone/Desktop`
- `/mnt/c/Users/Someone/Documents`
- `/mnt/c/Users/Someone/Downloads`
- `/mnt/c/Users/Someone/Desktop/SuGaR-main`
- `/home/someone`
- `/mnt/d (maxdepth 3 name filter + candidate subtree)`
- `/mnt/e (missing)`
- `/mnt/f (missing)`

工具：find (name point_cloud.ply / chkpnt30000.pth)；find archives by sugar/ggs/uggs name；zipfile member listing for WeChat uggs-m1-guizang*.zip；df -h / ls home；prior snapshot ZIP member scan (39 under research/snapshots)

局限：
- no bulk archive extract
- recycle bin not fully enumerable / not restored
- D: not full recursive file walk (size/time bound); maxdepth-3 name filter + candidate subtrees
- inaccessible locations not counted as absence
- did not re-download 90 release volumes

## 旧结果仍在（对照 JSON）

- scan24: overall=0.35496639222779447 sha_ok=True trunc4_pc=False no_dn_pc=False
- scan105: overall=0.47048834938509165 sha_ok=True trunc4_pc=False no_dn_pc=False
- scan122: overall=0.3242774839490299 sha_ok=True trunc4_pc=False no_dn_pc=False

## 强原模型候选：0

无。未发现 `ggs_ablate_no_dn` 或 `ggs_ex_trunc4` 下 scan24/105/122 的 `point_cloud.ply` / `chkpnt30000.pth`。

## 微信压缩包（成员目录）

- `/mnt/c/Users/Someone/Documents/xwechat_files/wxid_felf1lyud2ms22_ca91/msg/file/2026-05/uggs-m1-guizang.zip` exists=True members=18 needle_hits=0 error=None
- `/mnt/c/Users/Someone/Documents/xwechat_files/wxid_felf1lyud2ms22_ca91/msg/file/2026-05/uggs-m1-guizang(1).zip` exists=True members=19 needle_hits=0 error=None
- `/mnt/c/Users/Someone/Documents/xwechat_files/wxid_felf1lyud2ms22_ca91/msg/file/2026-05/uggs-m1-guizang(2).zip` exists=True members=19 needle_hits=0 error=None
- `/mnt/c/Users/Someone/Documents/xwechat_files/wxid_felf1lyud2ms22_ca91/msg/file/2026-05/uggs-m1-guizang(3).zip` exists=True members=19 needle_hits=0 error=None
是否含试点高斯成员：False（命中 needle 不等于含完整 point_cloud；见 JSON）。

## 已见但排除为「原 trunc4 对应高斯」的类别

- `ggs_pipeline` 非试点场景高斯：P5 试点未使用这些目录。
- `ggs_tau_*`：τ 不同，禁止替代。
- `research/experiments/*`：factor / history-gap / exact NCC 等分叉，不是归档 no_dn。
- `input.ply` / `recon*.ply`：不是可渲染完整高斯。

机器可读：`jia_local_recovery_audit.json`。git HEAD：`8387442ef901329fae0cfcd238561f23c2a540b3`。

下一步：按授权进入「找不到原模型 → 新配对重建实验」；旧 trunc4 JSON 仅历史参考。
