# RoleChat-7B 🤗

<p align="center">
  <img src="assets/sft_loss.png" width="45%" />
  <img src="assets/dpo_loss.png" width="45%" />
</p>

> 基于Qwen2.5-7B的多角色情感对话大模型
> SFT指令微调 + DPO偏好对齐 + 完整开源训练代码

[![GitHub Stars](https://img.shields.io/github/stars/chenzhikang123/RoleChat-7B?style=social)](https://github.com/chenzhikang123/RoleChat-7B)
[![ModelScope](https://img.shields.io/badge/ModelScope-RoleChat--7B-blue)](https://modelscope.cn/models/czk123123/RoleChat-7B)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)


## ✨ 项目亮点

- 🎭 支持**8种角色**（知心朋友/生活导师/心理咨询师/温暖姐姐等）
- 💬 情感陪伴专项训练，回答有温度、自然真实
- 🔧 **完整开源训练代码**，数据构建→训练→评估→部署全流程可复现
- 📊 SFT+DPO完整训练流程，Loss从1.62降至1.15

## 📊 效果对比

**用户：我今天失恋了，心情很差**

| 模型 | 回复 |
|------|------|
| 基础Qwen2.5-7B | 这样的感受是很正常的，每个人在面对感情挫折时都会有类似情绪反应...（说教式） |
| **RoleChat-7B** | **你愿意和我聊聊发生了什么吗？我在这里陪着你，不着急，你想说多少都可以。** |

## 🚀 快速开始

### 在线体验
👉 [ModelScope模型页](https://modelscope.cn/models/czk123123/RoleChat-7B)

### 本地使用

```python
from modelscope import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "czk123123/RoleChat-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="cuda"
)

messages = [
    {
        "role": "system",
        "content": "你是一个温柔体贴的知心朋友，善于倾听，给予情感支持。"
    },
    {
        "role": "user",
        "content": "我今天心情很差"
    }
]

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer([text], return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True
)
print(tokenizer.decode(
    outputs[0][len(inputs.input_ids[0]):],
    skip_special_tokens=True
))
```

## 🎭 支持的角色

```python
roles = [
    "你是一个温柔体贴的知心朋友，善于倾听，给予情感支持，说话温和有耐心。",
    "你是一个积极乐观的生活导师，善于鼓励他人，帮助用户走出低谷。",
    "你是一个理性温和的人生顾问，善于分析问题，提供务实建议。",
    "你是一个幽默风趣的好朋友，善于用轻松方式化解压力。",
    "你是一个经验丰富的心理咨询师，善于引导用户发现问题根源。",
    "你是一个温暖的姐姐，说话亲切自然，给予姐姐般的关爱。",
    "你是一个可靠的兄长，说话直接但充满关心。",
    "你是一个睿智的长者，善于用人生经验帮助用户看清问题。",
]
```

## 🔧 训练流程

### 数据构建
BELLE开源数据集（15601条）+ 自合成情感对话数据（1920条） = 17521条SFT训练数据 针对情感场景自构建DPO偏好数据集（800条）
### 训练阶段

| 阶段 | 方法 | 数据量 | 效果 |
|------|------|--------|------|
| SFT指令微调 | QLoRA(r=16) | 17521条 | Loss 1.62→1.15 |
| DPO偏好对齐 | LoRA(r=8) | 800条 | 回答更自然流畅 |

### 训练曲线

| SFT训练Loss | DPO训练Loss |
|------------|------------|
| ![SFT Loss](assets/sft_loss.png) | ![DPO Loss](assets/dpo_loss.png) |

> DPO训练中loss较低，分析为训练数据量不足导致轻微过拟合，
> 但实际对话效果经三阶段对比测试仍有明显提升。


## 🖥️ 环境配置

```bash
git clone https://github.com/chenzhikang123/RoleChat-7B.git
cd RoleChat-7B
conda create -n rolechat python=3.11 -y
conda activate rolechat
pip install -r requirements.txt
```

## 📦 模型下载

```python
from modelscope import snapshot_download
snapshot_download('czk123123/RoleChat-7B', cache_dir='./models')
```

## 📝 引用

如果本项目对你有帮助，欢迎Star ⭐

## 开源协议

Apache License 2.0