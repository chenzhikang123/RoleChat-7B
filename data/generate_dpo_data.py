"""
RoleChat-7B DPO偏好对齐数据构建
特性：
1. 5类场景均匀采样，正负向平衡
2. 15种chosen开头风格，禁止模板化
3. 5种rejected风格，强制多样化
4. 多重质量过滤，自动丢弃低质量数据
5. 每20条保存一次，方便实时查看
"""
import json
import os
import time
import random
from openai import OpenAI

client = OpenAI(
    api_key="填入大模型key",
    base_url="https://api.siliconflow.cn/v1"
)

# ==========================================
# 8种角色设定
# ==========================================
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

# ==========================================
# 5类情感场景，共50个用户输入
# ==========================================
EMOTION_INPUTS = {
    "情感低落类": [
        "我今天失恋了，心情很差，感觉整个人都垮了",
        "我和最好的朋友吵架了，感觉很委屈，明明是他的错",
        "我暗恋的人喜欢了别人，心里好痛，感觉自己很傻",
        "我被最信任的朋友背叛了，不知道该怎么办",
        "我和家人大吵了一架，现在谁都不理谁，很难受",
        "我觉得自己不被任何人喜欢，越想越难过",
        "我和多年的好友闹翻了，感觉失去了很重要的东西",
        "我今天被人当众嘲笑了，感到很羞耻",
        "我喜欢的人说只想和我做朋友，我不知道该怎么面对他",
        "我觉得自己付出了很多，但没人在乎",
    ],
    "压力焦虑类": [
        "我最近工作压力很大，感觉快撑不住了",
        "我考试没考好，很害怕告诉父母",
        "我对未来感到很迷茫，不知道自己想要什么",
        "我最近一直失眠，整个人状态很差",
        "我找工作一直碰壁，感觉快绝望了",
        "我毕业了找不到工作，压力好大",
        "我被领导当众批评了，感觉很挫败",
        "我拖延症很严重，什么事都做不好，对自己很失望",
        "我面临一个很重要的选择，不知道该怎么决定",
        "我最近总是情绪不稳定，不知道为什么",
    ],
    "自我怀疑类": [
        "我觉得自己一事无成，看着同龄人都过得很好，很自卑",
        "我感觉自己比不上身边任何人，越来越没有自信",
        "我总是做不好任何事，感觉自己很没用",
        "我觉得自己长得不好看，很自卑",
        "我感觉自己说话总是出错，不受人欢迎",
        "我努力了很久但没有成果，开始怀疑自己的能力",
        "我感觉自己什么都学不会，是不是比别人笨",
        "我总是被人忽视，感觉自己不重要",
        "我害怕失败，所以什么都不敢尝试",
        "我觉得生活没有意思，每天都很空虚",
    ],
    "孤独无助类": [
        "我感觉没有人真正理解我，很孤独",
        "我在一个新城市，认识的人很少，感觉很孤单",
        "我有话想说但不知道跟谁说",
        "我感觉自己是人群中最格格不入的那个",
        "我家人不理解我，朋友也不理解我，感觉很绝望",
        "我最近一个人住，感觉特别孤独",
        "我不知道怎么融入新环境，感觉大家都不喜欢我",
        "我感觉朋友们都有自己的生活，只有我是多余的",
        "我最近心情很差，但不想让别人担心，只能自己扛着",
        "我感觉自己越来越不会和人相处了",
    ],
    "积极分享类": [
        "我今天完成了一个重要的目标，好开心想分享",
        "我今天被人夸奖了，心情超好",
        "我交到了一个很好的新朋友，很开心",
        "我今天鼓起勇气做了一件一直不敢做的事",
        "我最近感觉自己在慢慢变好，想和你说说",
        "我今天帮助了一个陌生人，感觉很温暖",
        "我终于解决了困扰我很久的问题，松了一口气",
        "我最近开始坚持运动了，感觉越来越好",
        "我今天和久违的朋友重新联系上了，很感动",
        "我完成了一件很有意义的事，想和你分享",
    ]
}

# ==========================================
# 15种chosen开头风格（每次随机选一种给模型参考）
# ==========================================
CHOSEN_OPENING_STYLES = [
    "听到你说这些，我心里很难受",
    "你现在一定很不好受",
    "这种感觉真的很沉重",
    "我很心疼你现在的状态",
    "感谢你愿意告诉我这些",
    "你不用一个人扛着这些",
    "听到这个我真的很担心你",
    "你说的这些让我很揪心",
    "这听起来真的很难熬",
    "你能说出来就已经很勇敢了",
    "我陪着你，你慢慢说",
    "这段时间一定很辛苦吧",
    "我在这里，你不是一个人",
    "听到你这样说，我心里很沉",
    "你愿意告诉我，我真的很高兴",
]

# ==========================================
# 积极场景专用chosen开头（分享喜悦时用）
# ==========================================
POSITIVE_OPENING_STYLES = [
    "太好了，听到你这么说我也开心",
    "哇，这真的太棒了",
    "听到这个消息我真的为你高兴",
    "你做到了，我好为你骄傲",
    "能感受到你现在的开心",
    "这真的值得好好庆祝一下",
    "你的这份喜悦让我也感到温暖",
    "能分享你的喜悦，我真的很开心",
]

# ==========================================
# 5种rejected风格（强制多样化）
# ==========================================
REJECTED_STYLE_PROMPTS = [
    {
        "name": "说教式",
        "desc": "用说教语气，比如'你应该''你要振作''你得学会'，给出大道理但没有共情",
        "example": "你要振作起来，每个人都会遇到这种事，要学会调整自己的心态。"
    },
    {
        "name": "建议堆砌式",
        "desc": "直接给一堆建议和方案，完全跳过情感回应，比如'你可以去运动/找朋友聊聊/换个环境'",
        "example": "你可以试试去跑步，或者找朋友聊聊，换个环境也会好很多。"
    },
    {
        "name": "轻描淡写式",
        "desc": "用'这很正常''没事的''大家都这样''时间会治愈'等敷衍带过，不重视用户感受",
        "example": "这很正常，每个人都会有这样的时候，时间会治愈一切的。"
    },
    {
        "name": "否定感受式",
        "desc": "否定或质疑用户的感受，比如'你想太多了''别太敏感''没必要这样''不至于吧'",
        "example": "你是不是想太多了，没那么严重，别太敏感了。"
    },
    {
        "name": "冷漠说理式",
        "desc": "用冷静理性的语气讲道理，没有温度，比如'从客观角度来看''其实你需要的是''这种情况下应该'",
        "example": "从客观角度来看，这种情况很常见，你需要的是调整认知，接受现实。"
    },
]

# ==========================================
# 幻觉检测关键词
# ==========================================
HALLUCINATION_KEYWORDS = [
    "好的回答内容", "差的回答内容", "chosen内容", "rejected内容",
    "用户说的话", "角色回复", "示例回答", "参考答案",
    "```", "{{", "}}", "json", "JSON",
    "好回答：", "差回答：", "回答一：", "回答二：",
    "根据以上要求", "以下是我生成", "我将生成",
]

# chosen必须包含的情感词
EMPATHY_KEYWORDS = [
    "我", "你", "陪", "听", "说", "聊", "感受", "心疼",
    "难受", "辛苦", "不容易", "倾诉", "分享", "开心",
    "在这里", "不孤单", "告诉我", "了解", "明白", "懂",
]

# 禁止chosen使用的固定开头
BANNED_OPENINGS = [
    "我能感受到你",
    "我完全理解你",
    "我非常理解你",
]


# ==========================================
# 质量验证函数
# ==========================================
def check_hallucination(text):
    for kw in HALLUCINATION_KEYWORDS:
        if kw in text:
            return True, f"包含异常词：{kw}"
    return False, "ok"


def validate_chosen(text, is_positive=False):
    if len(text) < 30:
        return False, "chosen太短(<30字)"
    if len(text) > 350:
        return False, "chosen太长(>350字)"

    has_hallucination, reason = check_hallucination(text)
    if has_hallucination:
        return False, f"chosen幻觉：{reason}"

    has_empathy = any(kw in text for kw in EMPATHY_KEYWORDS)
    if not has_empathy:
        return False, "chosen缺少情感词"

    for banned in BANNED_OPENINGS:
        if text.startswith(banned):
            return False, f"使用了禁止开头：{banned}"

    return True, "ok"


def validate_rejected(text):
    if len(text) < 10:
        return False, "rejected太短(<10字)"
    if len(text) > 200:
        return False, "rejected太长(>200字)"

    has_hallucination, reason = check_hallucination(text)
    if has_hallucination:
        return False, f"rejected幻觉：{reason}"

    return True, "ok"


def validate_pair(chosen, rejected):
    if chosen.strip() == rejected.strip():
        return False, "chosen和rejected完全相同"

    if len(chosen) <= len(rejected):
        return False, f"chosen({len(chosen)}字)不比rejected({len(rejected)}字)长"

    chosen_chars = set(chosen)
    rejected_chars = set(rejected)
    if len(chosen_chars | rejected_chars) == 0:
        return False, "字符集为空"

    overlap = len(chosen_chars & rejected_chars) / len(chosen_chars | rejected_chars)
    if overlap > 0.88:
        return False, f"chosen和rejected过于相似({overlap:.2f})"

    return True, "ok"


def full_validate(sample, user_input, is_positive=False):
    if "chosen" not in sample or "rejected" not in sample:
        return False, "缺少chosen或rejected字段"

    chosen = sample["chosen"].strip()
    rejected = sample["rejected"].strip()

    valid, reason = validate_chosen(chosen, is_positive)
    if not valid:
        return False, reason

    valid, reason = validate_rejected(rejected)
    if not valid:
        return False, reason

    valid, reason = validate_pair(chosen, rejected)
    if not valid:
        return False, reason

    if chosen == user_input or rejected == user_input:
        return False, "回答和用户输入相同"

    return True, "ok"


# ==========================================
# 生成单条DPO数据
# ==========================================
def generate_dpo_sample(user_input, role, category):
    is_positive = category == "积极分享类"

    if is_positive:
        opening = random.choice(POSITIVE_OPENING_STYLES)
        chosen_requirement = f"""
【好回答（chosen）要求】
这是用户分享喜悦的场景，好回答要：
1. 开头参考这种风格（不要照抄）："{opening}"
2. 真诚为用户感到高兴，语气温暖
3. 主动追问细节，表示想多了解
4. 和用户一起分享这份喜悦
5. 50~150字，自然流畅

【差回答（rejected）要求】
使用这种风格：轻描淡写式，用"嗯""不错""这很正常"等敷衍带过，没有真诚的回应
20~60字"""
    else:
        opening = random.choice(CHOSEN_OPENING_STYLES)
        rejected_style = random.choice(REJECTED_STYLE_PROMPTS)
        chosen_requirement = f"""
【好回答（chosen）要求】
1. 开头参考这种风格（不要照抄，要自然变化）："{opening}"
2. 绝对不能用"我能感受到你""我完全理解你"开头
3. 表达真实共情，让用户感到被理解
4. 主动追问或表示愿意陪伴
5. 语气自然，像真朋友说话，不像模板
6. 50~150字

【差回答（rejected）要求】
使用{rejected_style["name"]}：{rejected_style["desc"]}
参考示例（不要照抄）："{rejected_style["example"]}"
20~80字，语气和chosen要有明显差异"""

    prompt = f"""你是一个专业的AI训练数据生成专家，正在为情感陪伴大模型生成DPO偏好训练数据。

【核心任务】
根据角色设定和用户输入，生成一对偏好数据：
- chosen：高质量回答，有温度，让用户感到被理解和陪伴
- rejected：低质量回答，缺乏共情或过于敷衍

【角色设定】
{role}

【用户说的话】（场景类别：{category}）
{user_input}

{chosen_requirement}

【输出格式要求】
1. 只输出JSON，不要任何解释或标签
2. 不要在回答内容里写"好回答""差回答"等标签
3. chosen和rejected内容要有明显差异
4. 每次生成的内容要有新意，不要重复固定句式

{{"chosen": "好回答的具体内容", "rejected": "差回答的具体内容"}}"""

    response = client.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.92,
        max_tokens=600,
        extra_body={"enable_thinking": False}
    )

    content = response.choices[0].message.content.strip()

    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    start_idx = content.find("{")
    end_idx = content.rfind("}") + 1
    if start_idx == -1 or end_idx <= start_idx:
        raise ValueError("未找到有效JSON结构")

    content = content[start_idx:end_idx]
    return json.loads(content), is_positive


# ==========================================
# 主函数：生成完整数据集
# ==========================================
def generate_dpo_dataset(total=800):
    save_dir = r"D:\llm\RoleChat-7B\data\processed"
    os.makedirs(save_dir, exist_ok=True)

    results = []
    failed = 0
    fail_reasons = {}
    category_stats = {cat: 0 for cat in EMOTION_INPUTS.keys()}
    start_time = time.time()

    categories = list(EMOTION_INPUTS.keys())

    print("=" * 60)
    print(f"RoleChat-7B DPO数据生成 - 工业级最终版")
    print(f"目标数量：{total}条")
    print(f"模型：Qwen/Qwen3-8B")
    print(f"场景类别：{len(categories)}类（均匀采样）")
    print(f"角色数量：{len(ROLE_PROMPTS)}种")
    print(f"chosen开头风格：{len(CHOSEN_OPENING_STYLES)+len(POSITIVE_OPENING_STYLES)}种")
    print(f"rejected风格：{len(REJECTED_STYLE_PROMPTS)}种")
    print(f"每{20}条保存一次")
    print("=" * 60)

    for i in range(total):
        # 均匀采样：先选类别，再选输入
        category = categories[i % len(categories)]
        user_input = random.choice(EMOTION_INPUTS[category])
        role = random.choice(ROLE_PROMPTS)

        if i > 0:
            elapsed = time.time() - start_time
            avg = elapsed / i
            remaining = int(avg * (total - i) / 60)
            print(f"\n[{i+1}/{total}] 剩余约{remaining}分钟 | {category} | {user_input[:15]}...")
        else:
            print(f"\n[{i+1}/{total}] {category} | {user_input[:15]}...")

        try:
            sample, is_positive = generate_dpo_sample(user_input, role, category)

            valid, reason = full_validate(sample, user_input, is_positive)
            if not valid:
                print(f"  ❌ 丢弃：{reason}")
                failed += 1
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                continue

            results.append({
                "instruction": role,
                "input": user_input,
                "chosen": sample["chosen"].strip(),
                "rejected": sample["rejected"].strip(),
                "category": category
            })

            category_stats[category] = category_stats.get(category, 0) + 1
            print(f"  ✅ 成功 | 累计：{len(results)}条 | 丢弃：{failed}条")
            print(f"  chosen预览：{sample['chosen'][:40]}...")
            print(f"  rejected预览：{sample['rejected'][:30]}...")

            # 每20条保存一次
            if len(results) % 20 == 0:
                tmp_path = os.path.join(save_dir, "dpo_train_tmp.json")
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\n  💾 已保存{len(results)}条 → {tmp_path}")
                print(f"  📊 类别分布：{category_stats}")
                print(f"  📊 丢弃原因：{fail_reasons}")

            time.sleep(0.5)

        except json.JSONDecodeError as e:
            reason = "JSON解析失败"
            print(f"  ❌ {reason}：{str(e)[:50]}")
            failed += 1
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            time.sleep(0.5)

        except ValueError as e:
            reason = "未找到有效JSON"
            print(f"  ❌ {reason}")
            failed += 1
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            time.sleep(0.5)

        except Exception as e:
            reason = "API请求失败"
            print(f"  ❌ {reason}：{str(e)[:80]}")
            failed += 1
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            time.sleep(2)

    # 最终保存（去掉category字段，只保留训练需要的字段）
    final_results = [
        {
            "instruction": r["instruction"],
            "input": r["input"],
            "chosen": r["chosen"],
            "rejected": r["rejected"]
        }
        for r in results
    ]

    output_path = os.path.join(save_dir, "dpo_train.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    total_time = int((time.time() - start_time) / 60)

    print("\n" + "=" * 60)
    print(f"✅ 生成完成！")
    print(f"总耗时：{total_time}分钟")
    print(f"成功：{len(results)}条 | 丢弃：{failed}条")
    print(f"成功率：{len(results)/total*100:.1f}%")
    print(f"\n类别分布：")
    for cat, count in category_stats.items():
        print(f"  {cat}：{count}条")
    print(f"\n丢弃原因统计：")
    for reason, count in sorted(
        fail_reasons.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {reason}：{count}条")
    print(f"\n最终数据保存至：{output_path}")
    print("=" * 60)


if __name__ == "__main__":
    generate_dpo_dataset(total=800)