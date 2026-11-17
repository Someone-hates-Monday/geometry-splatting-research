# 甲重建路线：scan24 冒烟审查包（STOP）

2026-11-17。协议：`JIA_TRUST_REBUILD_PROTOCOL_20261116.md`。  
工作树未提交。**在组织者确认前：不做 trust 全量提取、不评 105/122 trust、不写 gates/verdict。**

## 身份

| 项 | 值 |
|---|---|
| run | `ggs_trust_rebuild_20261116` |
| git HEAD | `8387442ef901329fae0cfcd238561f23c2a540b3` |
| 脏改动 | `loss_utils.py` / `mesh_extract.py` / `evaluate_dtu_mesh.py`（已修改）；`scripts/smoke_jia_trust_fuse_scan24.py`（未跟踪） |
| 核心 diff | `unified_geometry_gs/output_compare/ggs_trust_rebuild_20261116/scan24/logs/impl_diff.patch`（20771 bytes） |

## 三场景模型（iteration_30000，均可加载）

| 场景 | ply bytes | sha256 |
|---|---|---|
| scan24 | 235676240 | `e772631d0800f5b49ac32e3ecc5f7d608d733e9d5483851d2e2ea0831303625c` |
| scan105 | 111001508 | `75281dabb4b8327224b672f01a500013405c10e03ecafcda684f359a419acf0f` |
| scan122 | 121405328 | `bd7239447fb43af4dbe50f910e426f59cb06b46628362226be2921576da0958d` |

训练日志正常结束：`scan24/logs/train.log` → `Training complete.`；队列 `queue_105_122.log` → `queue complete`。

## scan24 control_trunc4（无掩码）

路径：`.../scan24/control_trunc4/`

```json
{
 "mean_d2s": 0.3045067887651643,
 "mean_s2d": 0.40637700950029426,
 "overall": 0.3554418991327293
}
```

- `recon_post.ply` sha256：见下方资产表  
- 旧归档 trunc4 JSON **不作主控制组**（重建协议）；本 overall 是本 run 的配对对照基线。

## scan24 冒烟

命令（WSL `uggs`）：

```bash
python scripts/smoke_jia_trust_fuse_scan24.py \
  -m output_compare/ggs_trust_rebuild_20261116/scan24/model \
  -s "$UGGS_DTU_ROOT/scan24" -r 2 --iteration 30000 \
  --out_dir output_compare/ggs_trust_rebuild_20261116/scan24/smoke
```

修复：冒烟改为 `get_combined_args`（与 `mesh_extract` 一致），否则 `cfg_args` 的 `multi_view_*` 未合并会在 `Scene` 初始化报 `float < NoneType`。

| 检查 | 结果 |
|---|---|
| 预览图 | `image_name=0000`，列表索引 0；`nearest_id[0]=1` → `0001` |
| 默认返回 arity（None / nearest / `return_trust_maps=False`） | 均为 2-tuple，`default_return_arity_ok=true` |
| 两次信任掩码一致 | `two_run_consistent=true`（统计与 K/error/evaluated/warp_valid/depth 摘要相同） |
| 计数完备 | `n_B=249922`；keep+ncc_reject+geom_or_warp = B；`depth_nonzero_eq_keep` |
| 保留率 | keep≈74.76%；ncc_reject≈6.00%；geom/warp≈19.24% |
| 产物 | `smoke_receipt.json` / `smoke_arrays.npz` / `smoke_redmask.png` |

### 已知缺口（审查须知）

§6.4 要求的「默认关闭路径与原版 loss/上游梯度在小输入上一致」**尚未做数值对照**：冒烟只断言返回项数为 2，注释里的 grad check 未落地。若审查要求补齐，应在进入 trust 全量前提交补测回执，仍停在本门。

## 资产哈希

| 相对路径 | bytes | sha256 |
|---|---|---|
| `.../scan24/control_trunc4/vis/results.json` | 101 | `d0d163417837912350cd4f69ec9965195cf5703e525d063c3960ce0032792d4b` |
| `.../scan24/control_trunc4/recon_post.ply` | 53969688 | `81a0877ba9c27c5c6c6bcbfff1ac7a859640d4d521dc2148bb6e1b1d5bb082fd` |
| `.../scan24/control_trunc4/recon.ply` | 54479845 | `c48c5d28ba7371144efef78ecc903c26d1b9eb322e9aaee0164fd57daadc2b91` |
| `.../scan24/control_trunc4/recon_aligned.ply` | 29281504 | `542143453ce77ebfcbaf29ef65bba332aec12869548c42c01da071c56cf347c4` |
| `.../scan24/smoke/smoke_receipt.json` | 2165 | `d90977448aa8cb408540ad7b1ace665c8c3eecb3711335d653bb9a5b25a6c5f5` |
| `.../scan24/smoke/smoke_arrays.npz` | 1644502 | `15dd032f07b05e7fa1c7c005c4644d321863d3ca1e3db44b468c88b0ead02393` |
| `.../scan24/smoke/smoke_redmask.png` | 448181 | `d60364ec0ba2609039507669c00cafecd9a6777444acdf8e5c0bff1682375dcc` |
| `.../scan24/logs/impl_diff.patch` | 20771 | `c46a8c2a71efb414dc89f1e6dfb5804be3f6bcdac74140ae774e3dc55db847ea` |
| `unified_geometry_gs/scripts/smoke_jia_trust_fuse_scan24.py` | 6530 | `6d912688b8da01fc7030b1d8e60a6cf1b9dfb64ea82782dd37c54114ae1255dd` |

复算：`handoff/_jia_hash_review_assets.py`。

## 停点请求

请审查：实现 diff、control JSON、冒烟回执/数组/红图、上述缺口是否可接受。

- **通过** → 授权后再跑 scan24→105→122 的 `trust_trunc4` 全量提取与统一评测/gates。  
- **不通过 / 需补梯度对照** → 先修冒烟或实现，仍不做 trust 全量。
