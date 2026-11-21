# 目录结构说明

这个仓库除了上游的代码，还带一套本地跑起来的约定：**哪些目录要自己下载东西放进去、放什么、
是否入库**。换机器时照着这张表补文件即可，权重和大文件都不在 git 里。

```
InstantHMR/
├── demo.py                    入口：--image / --video / --camera
├── instanthmr/                库代码（detector.py 的 CUDA 快路径就在这里）
├── tools/                     export_onnx.py / prefetch_models.py / coreml/
├── training/                  训练、微调、mhr_model.pt（一般只读）
├── benchmark/                 评测脚本（只读）
├── datasets_pipeline/         数据集生成（只读）
├── assets/                    仓库自带素材 + 本地产物（.rrd / 截图 / 对比图）
├── models/                    ★ ONNX 权重（忽略，只保留 README.md）
├── pytorch/                   ★ .pth 检查点 + run_config.json
├── .venv/                     Python 环境（忽略）
├── .tmp_wheels/               离线 wheel 缓存（忽略）
├── run_image.bat              ◎ 图片
├── run_video.bat              ◎ 视频
├── run_camera.bat             ◎ 摄像头
├── setup_windows.bat          ◎ 一键配置
├── WINDOWS.md / WINDOWS.zh.md  使用指南（英 / 中）
├── RUN.md                     实测记录（性能、fit/focal、踩坑）
└── STRUCTURE.md               本文件
```

## 我要放 X，放哪里？

| 想放的东西 | 位置 | 入库？ | 说明 |
| --- | --- | --- | --- |
| `demo.py` 用的 ONNX 权重 | `models\` | ❌ `.onnx` 忽略 | 详见 [`models/README.md`](models/README.md)；也可以交给 HF 缓存自动下 |
| MHR body 图 `mhr-body.onnx` | `models\` | ❌ | 12.7 MB，和主模型放一起，demo 会自动找同目录的它 |
| PyTorch 检查点 `.pth` | `pytorch\<档位>\` | ❌ `.pth` 忽略 | 导出 ONNX 用，详见 [`pytorch/README.md`](pytorch/README.md) |
| `run_config.json` | 和 `.pth` 同目录 | ✅ **入库** | 1 KB，决定导出的图结构，必须跟 `.pth` 配对 |
| 训练用的 `mhr_model.pt` | `training\checkpoints\` | ❌ `checkpoints/` 忽略 | 只给 `--fit` 导出 / body 图转换 |
| 检测器权重 `rf-detr-*.pth` | `%USERPROFILE%\.roboflow\models\` | ❌ | rfdetr 自己管理，不在仓库里 |
| 离线 wheel（2.5 GB 那种） | `.tmp_wheels\` | ❌ | 只是给 `pip --no-index --find-links` 用 |
| 自己跑出来的回放 | `assets\*.rrd` | ❌ | 会长到 GB 级，别提交 |
| 对比图 / 截图 / 测试视频 | `assets\` | ❌ | 目前 `.gitignore` 忽略 `.mp4`，`.jpg/.png` 由本地 exclude 处理 |
| 新的文档、脚本 | 仓库根目录 | ✅ | 小的、可复现的东西尽量入库 |

## `.gitignore` 的关键几条

```
models/*              # 权重目录整目录忽略……
!models/README.md     # ……但保留说明文件
*.onnx  *.pth  *.pt   # 权重本身
*.rrd   *.mp4         # 回放和视频
.venv/  .tmp_wheels/  # 环境和 wheel 缓存
```

`pytorch/` **没有整目录忽略**：`.pth` 由 `*.pth` 规则挡住，`run_config.json` 则正常入库。

本机还有几条只在当前克隆生效的排除（`.git/info/exclude`，不会推给别人）：`assets` 下的
`test_zidane.jpg`、`root_track_fit0_vs_fit2.png`、`demo_user_video_overlay.jpg`。

## 常见操作

```bat
:: 从零准备权重（推荐，写到 HF 缓存）
setup_windows.bat
.\.venv\Scripts\python.exe tools\prefetch_models.py --model medium-commercial --fit 1,2

:: 手动放到 models\ 后指定路径
run_video.bat clip.mp4 --model models\instanthmr-medium-commercial-fit2.onnx --fit 2

:: 从 .pth 重新导出（改了代码或想换精度时）
.\.venv\Scripts\python.exe tools\export_onnx.py --ckpt pytorch\medium-commercial\instanthmr-medium-commercial.pth --fit 2 --output models\instanthmr-medium-commercial-fit2.onnx
```

## 加新目录时的约定

1. 目录里放一个 `README.md` 说明用途和文件来源；
2. 大文件在 `.gitignore` 里整类挡住（按后缀，别按目录），再用 `!xxx/README.md` 把说明文件放出来；
3. 目录结构有变化时同步更新本文件。
