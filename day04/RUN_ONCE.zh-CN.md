# Day04：服务器运行一次，下载后离线学习

适用当前项目的 Linux 双 A100 环境；本地终端为 Windows PowerShell。
这份清单不会连接、启动或关闭你的服务器。端口、地址以 Vast.ai 当前实例页面为准：仓库里同时出现了旧地址 `23.127.144.217:26936` 和预跑地址 `ssh6.vast.ai:20163`，不能据此认定现在哪个有效。

## 1. 这一天要取得哪些证据

前三天的基础是：单卡可复现训练 → 手动梯度同步 → bucket + autograd hook + DDP。
Day04 接着回答：一个 optimizer step 分成 K 个 microbatch 后，如何只在最后同步，同时保持大 batch 的梯度、参数更新、LR 和 Adam 计数一致。

| 阶段 | 要取得的结果 | 是否需要服务器 GPU |
| --- | --- | --- |
| environment / probe | 实际 torch/CUDA/NCCL、GPU、拓扑和运行前后状态；代码与源码副本 | 是 |
| semantics | G = W × K × m、短窗口权重、SGD/AdamW 对缩放错误的反应 | 本次在 GPU 跑；脚本也支持 CPU |
| parity | A 梯度、B 首次更新参数、C 初始 loss、D LR/Adam 步数、E 短窗口梯度；五个错误对照必须被检测到 | 是 |
| count | K=4 时，每个 optimizer step 的 NCCL AllReduce/Broadcast 数量，保留两个 rank 的 trace | 是 |
| timing | K=1/2/4/8；无通信基线、flat、MiniDDP、DDP、单 GPU、等效大 batch；两次重复 | 是 |
| slowlink | K=1/4，仅为子进程设置 `NCCL_P2P_DISABLE=1`；两次重复 | 是 |
| 阅读源码、查看 trace、解释结果、改写 notes | 下载后完成 | 否 |

`mini_ddp_accum.py` 的 **YOU WRITE** 部分已经有完整代码，可以直接运行。如果你要完成“自己实现”的练习，应先在本地改好再上传；以后改了实现，这次运行只能证明当时归档版本的行为。
`notes.md` 和 GUIDE 附录是 2026-11-01 的预跑数据，不是你这次的结果。建议先在 GUIDE §1.9 写下预测。

## 2. 本地 PowerShell：打包并上传代码

保留 Day01–03 的代码依赖，但不重新跑前三天。服务器可能只是文件上传副本，不依赖 `git pull`。
上传会更新对应项目源码文件；如果服务器上有尚未下载的源码修改，先备份。

```powershell
Set-Location 'D:\AI Infra Learning'
$VastHost = '替换为当前SSH主机'
$VastPort = 12345 # 替换为当前SSH端口
$RunId = 'a100_' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$RemoteProject = '/workspace/AI-Infra-Learning'
New-Item -ItemType Directory -Force '.\outputs' | Out-Null
$InputArchive = Join-Path $PWD 'outputs\day04_input.tar.gz'
tar --exclude='__pycache__' --exclude='*.pyc' -czf $InputArchive README.md requirements.txt day04 experiments/day01_single_gpu experiments/day02_data_parallel experiments/day03_ddp_hooks experiments/day04_grad_accum
if ($LASTEXITCODE -ne 0) { throw '代码打包失败' }
scp -P $VastPort $InputArchive "root@${VastHost}:/workspace/day04_input.tar.gz"
if ($LASTEXITCODE -ne 0) { throw '代码上传失败' }
ssh -p $VastPort "root@$VastHost" "mkdir -p '$RemoteProject' && tar -xzf /workspace/day04_input.tar.gz -C '$RemoteProject'"
if ($LASTEXITCODE -ne 0) { throw '服务器解压失败' }
Write-Host "本次RunId：$RunId"
ssh -p $VastPort "root@$VastHost"
```

打包包含 Day02 的 `ring_allreduce.py` 等传递依赖，避免只上传 Day04 后出现 import 错误。
不包含本地 outputs、虚拟环境或 SSH 密钥。

## 3. 服务器 Bash：一次运行完整实验

把下面 `RID` 换成上一步打印的 RunId。命令中的服务器路径沿用现有项目设置。
先确认现有环境存在；不要为这次实验升级 torch、重装 CUDA 或驱动。

```bash
cd /workspace/AI-Infra-Learning
test -f .venv/bin/activate || { echo '缺少现有虚拟环境，请先定位环境'; exit 1; }
RID=a100_替换为上一步的时间戳
# 避免 Windows CRLF 换行造成 Bash 报错。
sed -i 's/\r$//' experiments/day04_grad_accum/run_server_once.sh
bash experiments/day04_grad_accum/run_server_once.sh "$RID"
echo "脚本退出码：$?"
```

保持 SSH 会话连接直到完成。README 估计 parity 约 3 分钟、count 约 2 分钟、timing 约 25 分钟、slowlink 约 8 分钟；额外预留打包和下载时间，共享主机实际耗时可能不同。
脚本按顺序执行，每阶段输出实时显示并写入独立日志。`pipefail` 确保 `tee` 不会掩盖实验失败；parity 失败就停止，不继续计时错误实现。
预跑前会执行双进程 NCCL smoke test；正常链路清除三个历史 NCCL 覆盖项，slowlink 仅影响它自己的子进程。
阶段失败时也会尝试打包部分日志；只有验证成功才生成 `SUCCESS.txt`。

正常结束会显示：

```text
PARITY: ALL CHECKS PASSED
OUTPUT CHECKS PASSED: 103 runs, 14 traces, 3 summary tables, 5 detected controls
Archive: /workspace/AI-Infra-Learning/outputs/day04_exports/day04_<RunId>.tar.gz
Run exit code: 0 (0 = all stages and output checks passed)
```

103 个子实验来自 parity 16 + count 7 + timing 64 + slowlink 16；这是完整默认矩阵，预计三个汇总表分别为 7、32、8 行。
每个子实验都生成 `metrics.json` 和 `result.pt`；timing/count/slowlink 使用 `--no-tensors`，它们的 `result.pt` 主要保存指标，不应当作可恢复训练的 checkpoint。

## 4. 如果要逐阶段手动输入

这是上一节脚本的实验命令展开，**二选一，不要与脚本使用同一 RunId 重复跑**。
脚本还负责快照、源码保存、产物检查和打包；下面适合了解执行顺序或排查单个阶段。

```bash
cd /workspace/AI-Infra-Learning
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ALGO
set -eo pipefail
RID="a100_$(date -u +%Y%m%dT%H%M%SZ)"
LOG="outputs/day04_grad_accum/manual_$RID"
mkdir -p "$LOG"
python experiments/day02_data_parallel/gpu_probe.py --want 2 2>&1 | tee "$LOG/probe_before.log"
python experiments/day04_grad_accum/find_symbols.py 2>&1 | tee "$LOG/find_symbols.log"
python experiments/day04_grad_accum/accum_semantics.py --device cuda 2>&1 | tee "$LOG/semantics.log"
python experiments/day04_grad_accum/compare_accum.py --stage parity --condition shared --run-id "$RID" 2>&1 | tee "$LOG/parity.log"
python experiments/day04_grad_accum/compare_accum.py --stage count --condition shared --run-id "$RID" --no-smoke 2>&1 | tee "$LOG/count.log"
python experiments/day04_grad_accum/compare_accum.py --stage timing --condition shared --run-id "$RID" --no-smoke --repeats 2 2>&1 | tee "$LOG/timing.log"
python experiments/day04_grad_accum/compare_accum.py --stage slowlink --condition shared --run-id "$RID" --no-smoke --repeats 2 2>&1 | tee "$LOG/slowlink.log"
python experiments/day02_data_parallel/gpu_probe.py --want 2 2>&1 | tee "$LOG/probe_after.log"
```

`compare_accum.py` 用普通 `python` 启动，它内部会调用 torchrun。不要再给它套一层双进程 torchrun。
故意注入错误的 negative controls 出现小写 `fail` 是预期；看各自的 `DETECTED by check` 和最终 `ALL CHECKS PASSED`。
断点恢复不能直接重跑失败阶段：已存在的子实验目录会拒绝覆盖。最简单的是新 RunId 重跑；若只缺少后续阶段，可沿用旧 RunId 仅运行尚未开始的阶段，随后手工归档整个结果目录。

## 5. 本地 PowerShell：下载、校验、解压

以下对应第 3 节的运行脚本。回到原本的 PowerShell，保留 `$VastHost`、`$VastPort`、`$RunId`、`$RemoteProject`；如果换了终端，请先重新赋值。
下载到独立目录，避免覆盖已有的预跑或其他本地结果。

```powershell
Set-Location 'D:\AI Infra Learning'
$DownloadDir = Join-Path $PWD "outputs\day04_downloads\$RunId"
New-Item -ItemType Directory -Force $DownloadDir | Out-Null
$ArchiveName = "day04_$RunId.tar.gz"
scp -P $VastPort "root@${VastHost}:$RemoteProject/outputs/day04_exports/$ArchiveName" $DownloadDir
if ($LASTEXITCODE -ne 0) { throw '结果下载失败' }
scp -P $VastPort "root@${VastHost}:$RemoteProject/outputs/day04_exports/$ArchiveName.sha256" $DownloadDir
if ($LASTEXITCODE -ne 0) { throw '校验文件下载失败' }
$Expected = ((Get-Content (Join-Path $DownloadDir "$ArchiveName.sha256")) -split '\s+')[0]
$Actual = (Get-FileHash (Join-Path $DownloadDir $ArchiveName) -Algorithm SHA256).Hash
if ($Actual -ine $Expected) { throw 'SHA256不一致，请重新下载' }
tar -xzf (Join-Path $DownloadDir $ArchiveName) -C $DownloadDir
if ($LASTEXITCODE -ne 0) { throw '结果解压失败' }
Get-Content (Join-Path $DownloadDir "session_$RunId\status.txt")
Get-Content (Join-Path $DownloadDir "session_$RunId\SUCCESS.txt") -ErrorAction Stop
Get-ChildItem $DownloadDir -Filter '*summary*.json'
```

确认 `exit_code=0`、`SUCCESS.txt` 存在、三个汇总 JSON 存在。失败归档也可能下载和 SHA 校验成功，所以不能只看下载成功。
压缩包内还有逐文件 `SHA256SUMS`，路径相对于解压目录；在 Linux/WSL 可进入解压目录执行 `sha256sum -c session_<RunId>/SHA256SUMS`。

下载内容包括：

| 文件 | 关机后用途 |
| --- | --- |
| `session_<RunId>/*.log` | 环境、语义实验、A–E 检查、五个负对照、通信计数和性能表 |
| `session_<RunId>/project_snapshot/` | 本次源码和 Day04 文档，留存结果对应的实现 |
| `session_<RunId>/installed_torch_source/` | 实际安装的 DDP Python 源码；C++ 源码按实际 torch git revision 尝试下载 |
| `par_*_<RunId>/{result.pt,metrics.json}` | 梯度、更新前后参数等正确性证据；可在本地 CPU 读取 |
| `cnt_*_<RunId>/trace_rank{0,1}.json` | 用 Perfetto 查看，或本地运行 `count_collectives.py` 重数 NCCL |
| `time_*_<RunId>/`、`slow_*_<RunId>/` | 各个变体每次重复的指标与计时元数据 |
| `count_summary_*.json`、`timing_summary_*.json`、`slowlink_summary_*.json` | 生成笔记和对比表 |

源码下载失败不会使 GPU 实验失败，错误和对应 URL 在 `environment.log` 里，关机后可在本地补下载。没有必要为了读源码继续租 GPU。

## 6. 下载前三天未保存的产物，然后释放实例

本次归档只含当前 Day04 RunId。销毁前如果 Day01–03 的日志、checkpoint 或 trace 还只在服务器，应一起下载。
先查看体积，尤其是大量 profiler trace；空间和下载时间允许时可额外归档整个 outputs（排除 Day04 导出包，防止重复）。

服务器 Bash，可选：

```bash
cd /workspace/AI-Infra-Learning
du -sh outputs/*
tar --exclude='outputs/day04_exports' -czf /workspace/ai_infra_outputs_backup.tar.gz outputs
cd /workspace
sha256sum ai_infra_outputs_backup.tar.gz > ai_infra_outputs_backup.tar.gz.sha256
```

本地 PowerShell，可选：

```powershell
scp -P $VastPort "root@${VastHost}:/workspace/ai_infra_outputs_backup.tar.gz" $DownloadDir
if ($LASTEXITCODE -ne 0) { throw '历史产物下载失败' }
scp -P $VastPort "root@${VastHost}:/workspace/ai_infra_outputs_backup.tar.gz.sha256" $DownloadDir
if ($LASTEXITCODE -ne 0) { throw '历史产物校验文件下载失败' }
$BackupExpected = ((Get-Content (Join-Path $DownloadDir 'ai_infra_outputs_backup.tar.gz.sha256')) -split '\s+')[0]
$BackupActual = (Get-FileHash (Join-Path $DownloadDir 'ai_infra_outputs_backup.tar.gz') -Algorithm SHA256).Hash
if ($BackupActual -ine $BackupExpected) { throw '历史产物SHA256不一致' }
```

最后在 Vast.ai 控制台操作当前实例：

- **Stop**：停止 GPU 租用，保留数据，仍收存储费。
- **Destroy**：删除实例及容器数据；当你确认需要的文件均已下载校验、近期不再用这个环境时再选择。

不要把退出 SSH、关闭终端或容器里的 `shutdown` 当成完成云实例计费操作；以控制台实例状态为准。
依据：[Vast.ai 实例管理说明](https://github.com/vast-ai/docs/blob/main/guides/instances/manage-instances.mdx) 与 [存储说明](https://github.com/vast-ai/docs/blob/main/guides/instances/storage/types.mdx)，核对日期 2026-11-02。

## 7. 离线分析时抓住四个结论

1. 正确性先看 A–E；只看 loss 或 AdamW 首次参数更新，可能漏掉损失缩放错误。
2. K=4 时，预期 flat AllReduce 为 4→1，bucketed MiniDDP/DDP 为约 16→4；DDP 的 buffer Broadcast 约 4→1。确切 bucket 数以本次 trace 为准，`bwd_only_nosync` 应接近 every。
3. 比性能用相同 K、相同有效全局 batch、同一次重复的基线；`none` 只是无通信计时下限，两个 rank 会分叉，不是正确训练。NVLink 上的微小差异可能被共享主机噪声淹没；慢链路结果不可直接与另一次租赁比较。
4. 梯度累积的价值是用小 microbatch 实现所需 global batch，并让 `no_sync` 减少同步次数。大 batch 放得下时，直接大 batch 可能更快。`no_sync` 必须包住 forward 和 backward；它不减少 K 次计算、梯度存储或 optimizer 更新本身。
