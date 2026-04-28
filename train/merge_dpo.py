"""
合并DPO LoRA权重到SFT模型
输入：merged_model（SFT合并后）+ DPO LoRA权重
输出：final_model（最终可部署模型）
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

# 路径配置
base_model_path = r"D:\llm\RoleChat-7B\output\merged_model"
dpo_lora_path = r"D:\llm\RoleChat-7B\output\dpo"
output_path = r"D:\llm\RoleChat-7B\output\final_model"

os.makedirs(output_path, exist_ok=True)

print("第一步：加载SFT合并后的模型...")
print("（用CPU加载，避免显存不足）")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.bfloat16,
    device_map="cpu"
)
print("✅ 基础模型加载完成")

print("\n第二步：加载DPO LoRA权重...")
model = PeftModel.from_pretrained(model, dpo_lora_path)
print("✅ DPO LoRA加载完成")

print("\n第三步：合并DPO权重...")
model = model.merge_and_unload()
print("✅ 合并完成")

print("\n第四步：保存最终模型...")
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print("=" * 50)
print("✅ 全部完成！")
print(f"最终模型保存至：{output_path}")
print("这就是你的RoleChat-7B最终版本")