# 使用 dtu/DTU 跑 baseline（不复制数据）

已把数据根目录指向 **`dtu/DTU`**（未改动 dtu 文件夹），训练直接读该路径。

**若你是 RTX 5090 / 50 系显卡**，先执行一次：  
`sed -i 's/\r$//' scripts/install_submodules_blackwell.sh && ./scripts/install_submodules_blackwell.sh`  
再跑下面训练命令。详见 **`docs/RTX5090_BLACKWELL.md`**。

## 1. 环境（WSL）

```bash
conda activate uggs
cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs
```

## 2. fused_ssim（可选）

未安装时会用 PyTorch 版 SSIM，稍慢但可正常训练。若需加速可安装：

```bash
pip install "git+https://github.com/rahul-goel/fused-ssim/" --no-build-isolation
```

## 3. 修脚本换行符（若出现 `bash\r: Permission denied`）

```bash
sed -i 's/\r$//' scripts/run_baseline.sh scripts/run_baseline_scan105.sh scripts/dtu.sh
```

## 4. 跑 scan105 baseline

```bash
source configs/dtu_path.env
./scripts/run_baseline_scan105.sh
```

或手动指定路径：

```bash
./scripts/run_baseline.sh /mnt/c/Users/Someone/Desktop/SuGaR-main/dtu/DTU/scan105 ./output/dtu/scan105
```

## 5. 批量跑所有 DTU scan

```bash
./scripts/dtu.sh
```

输出目录：`output/dtu/scan105/` 等，mesh 在 `recon_post.ply`。
