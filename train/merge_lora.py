"""
合并LoRA权重到基础模型
作用：把训练好的LoRA补丁永久合并进基础模型
     得到完整的可部署模型
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

# 路径配置
base_model_path = r"D:\llm\RoleChat-7B\models\Qwen\Qwen2___5-7B-Instruct"
lora_path = r"D:\llm\RoleChat-7B\output\sft\checkpoint-3288"
output_path = r"D:\llm\RoleChat-7B\output\merged_model"

os.makedirs(output_path, exist_ok=True)

print("第一步：加载基础模型...")
print("用CPU加载，避免显存不足，需要几分钟...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.bfloat16,
    device_map="cpu"
)
print("✅ 基础模型加载完成！")

print("\n第二步：加载LoRA权重...")
model = PeftModel.from_pretrained(model, lora_path)
print("✅ LoRA权重加载完成！")

print("\n第三步：合并LoRA到基础模型...")
print("这一步需要几分钟，请耐心等待...")
model = model.merge_and_unload()
print("✅ 合并完成！")

print("\n第四步：保存完整模型...")
print("文件较大约14GB，需要几分钟...")
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print("="*50)
print("✅ 全部完成！")
print(f"完整模型保存至：{output_path}")