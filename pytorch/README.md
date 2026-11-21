# `pytorch/` — PyTorch 权重与导出配置（**大文件不入库**）

这里放 `.pth` 检查点和它对应的 `run_config.json`，用途只有一个：用 `tools/export_onnx.py`
把它们导成 `models\` 里的 ONNX 图（`demo.py` 不直接读 `.pth`）。

## 目录约定

```
pytorch\
└── medium-commercial\
    ├── instanthmr-medium-commercial.pth    129,400,379 B   （HF 下载，不入库）
    └── run_config.json                       1,069 B        （HF 下载，入库）
```

`pytorch/*` 被忽略，例外是 `pytorch/*/run_config.json` —— 这个 JSON 很小（1 KB）且决定
`cliff_focal` / `backbone` / `image_size` / `cont_head` / `bound_scales`，跟着仓库走能保证
任何人导出的图一致。

## 从哪下

```
https://huggingface.co/momolesang/InstantHMR/resolve/main/instanthmr-medium-commercial.pth
https://huggingface.co/momolesang/InstantHMR/resolve/main/run_config.json
```

`.pth` 和 `run_config.json` 必须是**同名模型、同一目录**，`export_onnx.py` 会自己读旁边的 config。

## 导出

```bat
:: 网络本身
.\.venv\Scripts\python.exe tools\export_onnx.py ^
    --ckpt pytorch\medium-commercial\instanthmr-medium-commercial.pth ^
    --output models\instanthmr-medium-commercial.onnx

:: 带运动学拟合（fit 1 / 2），需要 training\checkpoints\mhr_model.pt（约 700 MB，MHR release assets）
.\.venv\Scripts\python.exe tools\export_onnx.py ^
    --ckpt pytorch\medium-commercial\instanthmr-medium-commercial.pth --fit 2 ^
    --output models\instanthmr-medium-commercial-fit2.onnx
```

自检：同一张图跑 `onnxruntime` 与 PyTorch，最大绝对误差应在 `1e-6` 量级（本机实测如此）。

## 其他权重放哪

| 文件 | 位置 | 说明 |
| --- | --- | --- |
| `rf-detr-nano.pth`（检测器） | `%USERPROFILE%\.roboflow\models\` | rfdetr 自己管理，可用 `--detector nano\|small\|medium` 切换 |
| `mhr_model.pt`（body 图） | `training\checkpoints\` | 只给 `--fit` 导出和 body 图转换用 |
| 自己微调的 `best_student_model*.pth` | 随意，建议 `pytorch\<档位>\` | 导出时 `--ckpt` 指过去即可 |
