from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 路径配置
base_model_path = r"D:\llm\RoleChat-7B\models\Qwen\Qwen2___5-7B-Instruct"
lora_path = r"D:\llm\RoleChat-7B\output\sft\checkpoint-3288"

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
model = PeftModel.from_pretrained(model, lora_path)
model.eval()
print("模型加载完成！")


def chat(role, user_input):
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
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    )
    return response


# 测试
role = "你是一个温柔体贴的知心朋友，善于倾听，给予情感支持，说话温和有耐心。"

test_cases = [
    "我今天失恋了，心情很差",
    "我最近工作压力很大，感觉快撑不住了",
    "我觉得自己一事无成，很自卑",
]

print("\n" + "=" * 50)
print("微调模型测试")
print("=" * 50)

for user_input in test_cases:
    print(f"\n用户：{user_input}")
    response = chat(role, user_input)
    print(f"模型：{response}")
    print("-" * 30)