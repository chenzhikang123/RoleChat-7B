import json
import os
import time
import random
from openai import OpenAI

client = OpenAI(
    api_key="填入你自己的大模型apikey进行数据生成",
    base_url="https://api.siliconflow.cn/v1"
)

SCENARIOS = [
    # 情感低落类
    "用户刚刚失恋，情绪低落",
    "用户感到孤独，需要陪伴",
    "用户被人误解，感到委屈",
    "用户和朋友吵架了，感到委屈",
    "用户和家人发生矛盾，心情复杂",
    "用户失去了一个重要的机会，很后悔",
    "用户觉得自己不被人喜欢，很难过",
    "用户暗恋的人喜欢了别人，心里很痛",
    "用户和多年好友闹翻了，很伤心",
    "用户感觉被最好的朋友背叛了",
    # 压力焦虑类
    "用户工作压力很大，感到焦虑",
    "用户考试失利，很沮丧",
    "用户对未来感到迷茫",
    "用户最近失眠，状态很差",
    "用户最近经济压力很大，很焦虑",
    "用户找工作屡屡碰壁，感到绝望",
    "用户感觉自己一事无成，很自卑",
    "用户家人生病，感到担心",
    "用户面临重要考试，紧张焦虑",
    "用户毕业找不到工作，感到压力巨大",
    "用户被领导批评，感到很挫败",
    "用户拖延症严重，对自己很失望",
    "用户感觉自己比不上身边的人，很焦虑",
    "用户准备换工作，对新环境感到不安",
    "用户最近总是情绪不稳定，不知道为什么",
    # 积极开心类
    "用户今天过得很开心，想分享",
    "用户刚完成一个重要目标，很有成就感",
    "用户收到了好消息，兴奋不已",
    "用户交到了新朋友，心情很好",
    "用户今天被人夸奖了，很开心",
    # 人际关系类
    "用户不知道怎么拒绝别人，感到困扰",
    "用户觉得自己社交能力差，不敢主动交友",
    "用户在新环境中不知道如何融入",
    "用户不知道怎么和喜欢的人表白",
    "用户感觉朋友圈的人都过得比自己好",
    # 成长困惑类
    "用户不知道自己的人生目标是什么",
    "用户想改变自己但不知道从哪里开始",
    "用户觉得生活没有意思，很空虚",
    "用户想学一项新技能但总是坚持不下去",
    "用户对自己的选择感到后悔，想重新来过",
]

ROLE_PROMPTS = [
    "你是一个温柔体贴的知心朋友，善于倾听，给予情感支持，说话温和有耐心，总是能让人感到被理解和关爱。",
    "你是一个积极乐观的生活导师，善于用温暖的话语鼓励他人，帮助用户走出低谷，看到生活的美好。",
    "你是一个理性温和的人生顾问，善于分析问题，在给予情感支持的同时提供务实的建议。",
    "你是一个幽默风趣的好朋友，善于用轻松的方式化解压力，让人在笑声中忘记烦恼。",
    "你是一个经验丰富的心理咨询师，善于引导用户发现问题根源，帮助用户找到内心的平静。",
    "你是一个温暖的姐姐，说话亲切自然，善于站在用户角度理解问题，给予姐姐般的关爱。",
    "你是一个可靠的兄长，说话直接但充满关心，善于给出实际可行的建议和鼓励。",
    "你是一个睿智的长者，见过很多人生起伏，善于用人生经验和故事帮助用户看清问题。",
]

INVALID_KEYWORDS = [
    "用户说的一句话",
    "用户说的话",
    "角色回复",
]


def generate_one_sample(scenario, role):
    prompt = f"""根据场景生成情感对话，严格JSON格式输出。

场景：{scenario}
角色：{role}

输出格式（只输出JSON，不要其他内容）：
{{"input": "用户说的一句话", "output": "角色回复50~150字"}}"""

    print(f"  正在请求API...", end="", flush=True)
    start = time.time()

    response = client.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=300,
        extra_body={"enable_thinking": False}
    )

    elapsed = time.time() - start
    print(f" 耗时{elapsed:.1f}s", flush=True)

    content = response.choices[0].message.content.strip()

    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    start_idx = content.find("{")
    end_idx = content.rfind("}") + 1
    if start_idx != -1 and end_idx > start_idx:
        content = content[start_idx:end_idx]

    data = json.loads(content)
    return data


def is_valid_sample(sample):
    if "input" not in sample or "output" not in sample:
        return False, "格式错误"

    input_text = sample["input"].strip()
    output_text = sample["output"].strip()

    if len(input_text) < 5:
        return False, "input太短"
    if len(output_text) < 20:
        return False, "output太短"
    if len(output_text) > 500:
        return False, "output太长"

    for kw in INVALID_KEYWORDS:
        if kw in input_text:
            return False, f"包含占位符:{kw}"

    return True, "ok"


def generate_dataset(total=2000):
    os.makedirs('./data/processed', exist_ok=True)

    results = []
    failed = 0
    start_time = time.time()

    print(f"开始生成{total}条情感对话数据")
    print(f"模型：Qwen/Qwen3-8B")
    print(f"场景：{len(SCENARIOS)}个 | 角色：{len(ROLE_PROMPTS)}个 | 组合：{len(SCENARIOS)*len(ROLE_PROMPTS)}种")
    print("-" * 50)

    for i in range(total):
        scenario = random.choice(SCENARIOS)
        role = random.choice(ROLE_PROMPTS)

        if i > 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = avg_time * (total - i)
            remaining_min = int(remaining / 60)
            print(f"[{i+1}/{total}] 预计剩余{remaining_min}分钟 | 场景：{scenario[:12]}...")
        else:
            print(f"[{i+1}/{total}] 场景：{scenario[:12]}...")

        try:
            sample = generate_one_sample(scenario, role)

            valid, reason = is_valid_sample(sample)
            if not valid:
                print(f"  ❌ 不合格：{reason}")
                failed += 1
                continue

            results.append({
                "instruction": role,
                "input": sample["input"].strip(),
                "output": sample["output"].strip()
            })
            print(f"  ✅ 成功 | 累计：{len(results)}条 | 失败：{failed}条")

            # 每50条保存一次
            if len(results) % 50 == 0:
                tmp_path = "./data/processed/synthetic_train_tmp.json"
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"  💾 已临时保存{len(results)}条，请检查数据质量")
                print(f"  文件路径：./data/processed/synthetic_train_tmp.json")

            time.sleep(0.3)

        except json.JSONDecodeError as e:
            print(f"  ❌ JSON解析失败：{e}")
            failed += 1
            time.sleep(0.5)
            continue
        except Exception as e:
            print(f"  ❌ 请求失败：{e}")
            failed += 1
            time.sleep(2)
            continue

    # 最终保存
    output_path = "./data/processed/synthetic_train.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_time = int((time.time() - start_time) / 60)
    print("-" * 50)
    print(f"生成完成！总耗时：{total_time}分钟")
    print(f"成功：{len(results)}条 | 失败：{failed}条")
    print(f"成功率：{len(results)/total*100:.1f}%")
    print(f"保存至：{output_path}")


if __name__ == "__main__":
    generate_dataset(total=2000)