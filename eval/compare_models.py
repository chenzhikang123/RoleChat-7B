"""
三阶段模型对比测试
对比：基础模型 vs SFT微调后 vs DPO对齐后
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# ==========================================
# 测试配置
# ==========================================
ROLE = "你是一个温柔体贴的知心朋友，善于倾听，给予情感支持，说话温和有耐心。"

TEST_CASES = [
    "我今天失恋了，心情很差，感觉整个人都垮了",
    "我最近工作压力很大，感觉快撑不住了",
    "我觉得自己一事无成，很自卑",
    "我和好朋友吵架了，很难受",
    "我今天完成了一个重要目标，好开心",
]

# ==========================================
# 模型路径配置
# ==========================================
MODELS = {
    "基础Qwen2.5-7B": r"D:\llm\RoleChat-7B\models\Qwen\Qwen2___5-7B-Instruct",
    "SFT微调后": r"D:\llm\RoleChat-7B\output\merged_model",
    "DPO对齐后": r"D:\llm\RoleChat-7B\output\dpo",
}


def load_model(model_path, is_lora=False, base_path=None):
    """加载模型"""
    print(f"  加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_path if is_lora else model_path
    )

    print(f"  加载模型权重...")
    model = AutoModelForCausalLM.from_pretrained(
        base_path if is_lora else model_path,
        dtype=torch.bfloat16,
        device_map="cuda"
    )

    if is_lora:
        print(f"  加载LoRA权重...")
        model = PeftModel.from_pretrained(model, model_path)

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, role, user_input):
    """生成回复"""
    messages = [
        {"role": "system", "content": role},
        {"role": "user", "content": user_input}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    )
    return response.strip()


def run_comparison():
    print("=" * 70)
    print("RoleChat-7B 三阶段模型对比测试")
    print("=" * 70)

    # 存储所有结果
    all_results = {}

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*70}")
        print(f"加载：{model_name}")
        print(f"路径：{model_path}")
        print(f"{'='*70}")

        try:
            # DPO阶段用LoRA加载
            is_lora = model_name == "DPO对齐后"
            base_path = r"D:\llm\RoleChat-7B\output\merged_model" if is_lora else None

            model, tokenizer = load_model(model_path, is_lora, base_path)
            print(f"✅ {model_name} 加载完成")

            results = []
            for i, question in enumerate(TEST_CASES):
                print(f"\n  测试 [{i+1}/{len(TEST_CASES)}]: {question[:20]}...")
                response = generate_response(model, tokenizer, ROLE, question)
                results.append(response)
                print(f"  回复：{response[:50]}...")

            all_results[model_name] = results

            # 释放显存
            del model
            torch.cuda.empty_cache()
            print(f"\n✅ {model_name} 测试完成，已释放显存")

        except Exception as e:
            print(f"❌ {model_name} 加载失败：{e}")
            all_results[model_name] = ["加载失败"] * len(TEST_CASES)

    # ==========================================
    # 打印完整对比结果
    # ==========================================
    print("\n" + "=" * 70)
    print("完整对比结果")
    print("=" * 70)

    for i, question in enumerate(TEST_CASES):
        print(f"\n{'='*70}")
        print(f"问题 {i+1}：{question}")
        print(f"{'='*70}")

        for model_name, results in all_results.items():
            print(f"\n【{model_name}】")
            print(results[i])
            print("-" * 40)

    # ==========================================
    # 保存结果到文件
    # ==========================================
    import json
    output = {
        "role": ROLE,
        "test_cases": TEST_CASES,
        "results": all_results
    }
    with open(r"D:\llm\RoleChat-7B\eval\comparison_results.json",
              'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 对比结果已保存至：eval/comparison_results.json")
    print("=" * 70)


if __name__ == "__main__":
    run_comparison()