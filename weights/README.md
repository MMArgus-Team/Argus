# weights/ — 本地模型权重

本目录存放**项目自带的本地模型权重**,不随 git 提交(体积大)。用一键脚本下载:

```bash
python download_weights.py
```

启动 dashboard 时,项目根的 `weights/` 会被**软链接**到 `<HERMES_HOME>/weights`
(见 `hermes_cli/config_sync.py`),所以 `config.yaml` 里可用**以 HERMES_HOME 为根的相对路径**
引用它们(相对路径是项目跨机器部署的既定约定)。

## 目录结构

```
weights/
├── README.md                       # 本文件 (唯一入 git 的内容)
└── qwen2.5-0.5b-instruct/          # Qwen2.5-0.5B-Instruct (HF snapshot: config/safetensors/tokenizer)
                                    #   用于 voice_intent_local (语音意图/分诊/语义EOU 的本地推理)
```

## config.yaml 引用(相对 HERMES_HOME,软链后可达)

```yaml
auxiliary:
  text:
    local_backend:
      local_path: "weights/qwen2.5-0.5b-instruct"
```

## 来源与获取方式

- **Qwen2.5-0.5B-Instruct** — 从 HF `Qwen/Qwen2.5-0.5B-Instruct`(Apache-2.0)**下载**:

  ```bash
  pip install huggingface_hub
  python download_weights.py            # 或 --hf-mirror 走国内镜像
  ```

## 关于 OCR(不在此目录)

OCR 用 RapidOCR,其 PP-OCR onnx **随 `rapidocr` wheel 自带**,`RapidOCR()` 无参构造即用
包内默认模型,**无需放进 weights/、也无需在 config 指定路径**。只要安装:

```bash
pip install -U rapidocr onnxruntime
```
