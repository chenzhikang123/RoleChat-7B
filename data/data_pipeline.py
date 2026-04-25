import json
import os
import random
from datasets import load_dataset
from tqdm import tqdm

# 固定随机种子，保证结果可复现
random.seed(42)

# 四种角色提示词，训练时随机分配
ROLE_PROMPTS = [
    "你是一个温柔体贴的知心朋友，善于倾听，给予情感支持，说话温和有耐心，总是能让人感到被理解和关爱。",
    "你是一个积极乐观的生活导师，善于用温暖的话语鼓励他人，帮助用户走出低谷，看到生活的美好。",
    "你是一个理性温和的人生顾问，善于分析问题，在给予情感支持的同时提供务实的建议。",
    "你是一个幽默风趣的好朋友，善于用轻松的方式化解压力，让人在笑声中忘记烦恼。",
]

def download_and_process():
    # 创建目录
    os.makedirs('./data/raw', exist_ok=True)
    os.makedirs('./data/processed', exist_ok=True)

    # 下载数据集
    print("正在下载BELLE数据集，请稍等...")
    dataset = load_dataset(
        "BelleGroup/train_1M_CN",
        split="train[:20000]",
        trust_remote_code=True
    )
    print(f"原始数据量：{len(dataset)}条")

    # 处理数据
    sft_data = []
    skip_count = 0

    for item in tqdm(dataset, desc="处理数据中"):
        instruction = item.get("instruction", "").strip()
        output = item.get("output", "").strip()

        # 质量过滤
        if len(instruction) < 5:        # 问题太短
            skip_count += 1
            continue
        if len(output) < 20:            # 回复太短
            skip_count += 1
            continue
        if len(output) > 800:           # 回复太长
            skip_count += 1
            continue
        if "□" in output:               # 含乱码
            skip_count += 1
            continue
        if "sorry" in output.lower():   # 过滤拒绝回复
            skip_count += 1
            continue

        # 随机分配角色
        role = random.choice(ROLE_PROMPTS)

        # 转换为训练格式
        sft_data.append({
            "instruction": role,
            "input": instruction,
            "output": output
        })

    print(f"过滤掉：{skip_count}条")
    print(f"最终数据量：{len(sft_data)}条")

    # 保存训练数据
    output_path = "./data/processed/sft_train.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"数据已保存至：{output_path}")
    print("数据准备完成！")

if __name__ == "__main__":
    download_and_process()