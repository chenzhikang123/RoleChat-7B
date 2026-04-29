"""
eval_rag.py
===========
三组对比评测：
    A. 基座 Qwen2.5-7B-Instruct（未微调）
    B. RoleChat-7B SFT+DPO（无 RAG）
    C. RoleChat-7B SFT+DPO+RAG（完整版）

评测维度（各 1-5 分）：
    1. 角色一致性 —— 回复是否符合所选角色人设和说话风格
    2. 情绪共情度 —— 是否理解并回应用户情绪，让用户感到被看见
    3. 回复质量   —— 语言是否自然流畅，避免说教/重复/空话

评判模型：Qwen3-235B-A22B-Instruct-2507（via SiliconFlow）

运行方式（三步，原因见下方说明）：
    # 第一步：关掉 api.py，用本机加载基座模型推理（显存只够跑一个7B）
    python eval/eval_rag.py --step base

    # 第二步：启动 api.py，推理 SFT+DPO 两组（有无RAG）
    python deploy/api.py   # 另一个终端先启动
    python eval/eval_rag.py --step sft

    # 第三步：调用 SiliconFlow 对三组回复统一打分，生成报告
    python eval/eval_rag.py --step judge --api_key YOUR_KEY

输出文件：
    eval/results/base_responses.json    ← A组回复
    eval/results/sft_responses.json     ← B+C组回复
    eval/results/scores.json            ← 原始评分
    eval/results/summary.txt            ← 汇总表格（直接放进README）
"""

import os, sys, json, time, argparse, requests, re
from pathlib import Path
from datetime import datetime

ROOT_DIR   = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

CHAT_API    = "http://localhost:8000/chat"
JUDGE_URL   = "https://api.siliconflow.cn/v1/chat/completions"
JUDGE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
SILICONFLOW_API_KEY = ""   # ← 填入你的 SiliconFlow API Key

# 基座模型路径（未微调的原始 Qwen2.5-7B-Instruct）
# 如果本地没有，填 ModelScope 模型名会自动下载
BASE_MODEL_PATH = r"D:\llm\RoleChat-7B\models\Qwen\Qwen2___5-7B-Instruct"

# ============================================================
# 50 条测试问题，覆盖 8 种角色 × 多种情绪场景
# ============================================================
TEST_CASES = [
    # ── 知心朋友 ──────────────────────────────────────────
    {"role": "知心朋友", "query": "我今天失恋了，心情很差，不知道怎么办"},
    {"role": "知心朋友", "query": "我感觉最近很孤独，身边没有人真正理解我"},
    {"role": "知心朋友", "query": "我和好朋友吵架了，现在很难受"},
    {"role": "知心朋友", "query": "我今天被老板骂了，心里很委屈"},
    {"role": "知心朋友", "query": "我最近总是睡不着，脑子里乱七八糟的"},
    {"role": "知心朋友", "query": "感觉自己什么都做不好，很沮丧"},
    {"role": "知心朋友", "query": "我妈今天又说我了，我真的很烦"},

    # ── 生活导师 ──────────────────────────────────────────
    {"role": "生活导师", "query": "我已经很努力了，但还是没什么进步，想放弃了"},
    {"role": "生活导师", "query": "我感觉自己人生没有方向，很迷茫"},
    {"role": "生活导师", "query": "我每天都很懒，什么计划都坚持不下去"},
    {"role": "生活导师", "query": "我一直想改变，但就是行动不起来"},
    {"role": "生活导师", "query": "感觉身边的人都比我优秀，我不知道自己有什么用"},
    {"role": "生活导师", "query": "我最近状态很差，对什么事都提不起劲"},

    # ── 人生顾问 ──────────────────────────────────────────
    {"role": "人生顾问", "query": "我不知道要不要辞职，现在的工作很稳定但没发展"},
    {"role": "人生顾问", "query": "我和男朋友在要不要异地这件事上有分歧"},
    {"role": "人生顾问", "query": "我父母希望我回老家，但我想留在大城市发展"},
    {"role": "人生顾问", "query": "我在两份工作机会之间纠结，不知道怎么选"},
    {"role": "人生顾问", "query": "我感觉自己一直在原地踏步，不知道问题出在哪"},

    # ── 幽默朋友 ──────────────────────────────────────────
    {"role": "幽默朋友", "query": "今天开会被领导当众点名批评，尬死了"},
    {"role": "幽默朋友", "query": "我室友又没洗碗，我快被气死了"},
    {"role": "幽默朋友", "query": "上班好无聊，每天都想摸鱼"},
    {"role": "幽默朋友", "query": "我今天发朋友圈没人点赞，感觉被世界抛弃了"},
    {"role": "幽默朋友", "query": "又到月底了，卡里只剩三位数"},

    # ── 心理咨询师 ────────────────────────────────────────
    {"role": "心理咨询师", "query": "我总是控制不住地焦虑，不知道为什么"},
    {"role": "心理咨询师", "query": "我经常莫名其妙地想哭，但说不出是为什么"},
    {"role": "心理咨询师", "query": "我总是觉得自己不值得被爱"},
    {"role": "心理咨询师", "query": "我很害怕失败，所以很多事情干脆不去做"},
    {"role": "心理咨询师", "query": "我最近总是回想小时候被批评的事，很难受"},
    {"role": "心理咨询师", "query": "我感觉自己情绪波动很大，不太能控制"},

    # ── 温暖姐姐 ──────────────────────────────────────────
    {"role": "温暖姐姐", "query": "我今天一个人在宿舍，感觉很空"},
    {"role": "温暖姐姐", "query": "我失恋了，感觉对感情很绝望"},
    {"role": "温暖姐姐", "query": "最近压力好大，感觉快撑不住了"},
    {"role": "温暖姐姐", "query": "我今天没吃饭，也不想动"},
    {"role": "温暖姐姐", "query": "我感觉家里人不理解我，很委屈"},
    {"role": "温暖姐姐", "query": "我考试没考好，被爸妈说了很久"},

    # ── 可靠兄长 ──────────────────────────────────────────
    {"role": "可靠兄长", "query": "我现在好烦好难过，感觉什么都不顺"},
    {"role": "可靠兄长", "query": "我被朋友背刺了，很心寒"},
    {"role": "可靠兄长", "query": "我最近一直在扛，感觉快撑不住了"},
    {"role": "可靠兄长", "query": "我在外地出差，很想家"},
    {"role": "可靠兄长", "query": "我创业失败了，亏了很多钱"},
    {"role": "可靠兄长", "query": "我和父亲关系一直不好，最近又起了冲突"},

    # ── 睿智长者 ──────────────────────────────────────────
    {"role": "睿智长者", "query": "我不知道我这辈子的意义是什么"},
    {"role": "睿智长者", "query": "我总觉得时间过得太快，什么都还没做好就老了"},
    {"role": "睿智长者", "query": "我感觉人生就是不断重复，很空洞"},
    {"role": "睿智长者", "query": "我三十岁了，感觉和理想的自己差距越来越大"},
    {"role": "睿智长者", "query": "我害怕死亡，想到这件事就很恐惧"},
    {"role": "睿智长者", "query": "我总是后悔过去的选择，走不出来"},
    {"role": "睿智长者", "query": "我感觉自己活得很累，但又不知道累在哪"},
]


# ============================================================
# 工具函数
# ============================================================
def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[保存] {path}")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_chat_api(role, query, use_rag, timeout=120):
    """调用本地 api.py 获取模型回复"""
    try:
        resp = requests.post(
            CHAT_API,
            json={
                "character": role,
                "messages": [{"role": "user", "content": query}],
                "use_rag": use_rag
            },
            timeout=timeout
        )
        resp.raise_for_status()
        return resp.json()["reply"]
    except Exception as e:
        print(f"  [错误] {e}")
        return f"[ERROR: {str(e)}]"


def call_judge(prompt, api_key, retries=3):
    """调用 SiliconFlow 评判模型打分"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    for attempt in range(retries):
        try:
            resp = requests.post(
                JUDGE_URL,
                headers=headers,
                json={
                    "model": JUDGE_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 300
                },
                timeout=60
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  [评判重试 {attempt+1}/{retries}] {e}")
            time.sleep(2)
    return None


def parse_scores(text):
    """
    从评判模型的回复里解析三个维度的分数。
    期望格式：角色一致性: 4, 情绪共情度: 5, 回复质量: 4
    """
    scores = {}
    patterns = {
        "角色一致性": r"角色一致性[：:]\s*([1-5])",
        "情绪共情度": r"情绪共情度[：:]\s*([1-5])",
        "回复质量":   r"回复质量[：:]\s*([1-5])",
    }
    for dim, pat in patterns.items():
        m = re.search(pat, text)
        scores[dim] = int(m.group(1)) if m else None
    return scores


def build_judge_prompt(role, query, reply_a, reply_b, reply_c):
    """构建评判 prompt"""
    return f"""你是一个专业的对话质量评审员。请对以下三个模型对同一问题的回复进行评分。

【角色设定】{role}
【用户问题】{query}

【模型A回复（基座模型，无微调）】
{reply_a}

【模型B回复（SFT+DPO微调，无RAG）】
{reply_b}

【模型C回复（SFT+DPO微调+RAG知识增强）】
{reply_c}

请分别对三个模型的回复，从以下三个维度各打1-5分：
- 角色一致性：回复是否符合"{role}"的人设、说话风格和行为边界（1=完全不符合，5=完全符合）
- 情绪共情度：是否理解并回应用户的情绪，让用户感到被看见和被接纳（1=冷漠说教，5=温暖共情）
- 回复质量：语言是否自然流畅，是否避免了空话/重复/说教（1=很差，5=很好）

输出格式（严格按此格式，不要多余内容）：
模型A - 角色一致性: X, 情绪共情度: X, 回复质量: X
模型B - 角色一致性: X, 情绪共情度: X, 回复质量: X
模型C - 角色一致性: X, 情绪共情度: X, 回复质量: X"""


# ============================================================
# 第一步：基座模型推理
# ============================================================
def step_base():
    """
    加载基座 Qwen2.5-7B-Instruct（未微调），推理所有测试问题。
    运行前请确保 api.py 已关闭（显存不够同时跑两个模型）。
    """
    print("=" * 60)
    print("第一步：基座模型推理")
    print("=" * 60)
    print(f"加载模型：{BASE_MODEL_PATH}")
    print("注意：请确保 api.py 已关闭！\n")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    os.environ["HF_HUB_OFFLINE"] = "0"  # 基座模型可能需要下载

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    model.eval()
    print("✅ 基座模型加载完成\n")

    results = []
    total = len(TEST_CASES)

    for i, case in enumerate(TEST_CASES):
        role  = case["role"]
        query = case["query"]
        print(f"[{i+1:02d}/{total}] {role} | {query[:30]}...")

        system_prompt = f"你是{role}，请保持角色设定，用温暖真诚的方式与用户对话。"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query}
        ]

        import torch
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.75,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        new_tokens = outputs[0][len(inputs.input_ids[0]):]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"  → {reply[:60]}...")

        results.append({
            "id":    i,
            "role":  role,
            "query": query,
            "reply_base": reply
        })

    save_json(results, OUTPUT_DIR / "base_responses.json")
    print(f"\n✅ 基座模型推理完成，共 {total} 条，已保存")
    print("下一步：启动 api.py，然后运行 --step sft")


# ============================================================
# 第二步：SFT+DPO 推理（有无 RAG 两组）
# ============================================================
def step_sft():
    """
    调用已启动的 api.py，分别收集 use_rag=False 和 use_rag=True 的回复。
    运行前请确保 api.py 已启动。
    """
    print("=" * 60)
    print("第二步：SFT+DPO 推理（无RAG vs 有RAG）")
    print("=" * 60)

    # 检查 api.py 是否在运行
    try:
        r = requests.get("http://localhost:8000/health", timeout=5)
        if not r.json().get("model_loaded"):
            print("❌ api.py 模型还未加载完成，请等待")
            return
        print("✅ api.py 运行正常\n")
    except Exception:
        print("❌ 无法连接 api.py，请先启动：python deploy/api.py")
        return

    # 加载已有的 base 结果
    base_path = OUTPUT_DIR / "base_responses.json"
    if not base_path.exists():
        print("❌ 未找到 base_responses.json，请先运行 --step base")
        return
    base_results = load_json(base_path)

    results = []
    total = len(TEST_CASES)

    for i, case in enumerate(TEST_CASES):
        role  = case["role"]
        query = case["query"]
        print(f"[{i+1:02d}/{total}] {role} | {query[:30]}...")

        # B 组：SFT+DPO，无 RAG
        print("  B（无RAG）...", end=" ", flush=True)
        reply_b = call_chat_api(role, query, use_rag=False)
        print(f"{reply_b[:40]}...")

        # C 组：SFT+DPO+RAG
        print("  C（有RAG）...", end=" ", flush=True)
        reply_c = call_chat_api(role, query, use_rag=True)
        print(f"{reply_c[:40]}...")

        results.append({
            "id":       i,
            "role":     role,
            "query":    query,
            "reply_b":  reply_b,
            "reply_c":  reply_c,
            "reply_base": base_results[i]["reply_base"]  # 合并 A 组
        })

        time.sleep(0.5)  # 防止请求太快

    save_json(results, OUTPUT_DIR / "sft_responses.json")
    print(f"\n✅ SFT 推理完成，已保存")
    print("下一步：运行 --step judge --api_key YOUR_KEY")


# ============================================================
# 第三步：LLM-as-Judge 打分
# ============================================================
def step_judge(api_key):
    print("=" * 60)
    print("第三步：LLM-as-Judge 打分")
    print("=" * 60)

    sft_path = OUTPUT_DIR / "sft_responses.json"
    if not sft_path.exists():
        print("❌ 未找到 sft_responses.json，请先运行 --step sft")
        return

    all_data = load_json(sft_path)
    scored   = []
    total    = len(all_data)

    for i, item in enumerate(all_data):
        print(f"[{i+1:02d}/{total}] 评判 {item['role']} | {item['query'][:30]}...")

        prompt = build_judge_prompt(
            role    = item["role"],
            query   = item["query"],
            reply_a = item["reply_base"],
            reply_b = item["reply_b"],
            reply_c = item["reply_c"]
        )

        judge_text = call_judge(prompt, api_key)
        if not judge_text:
            print("  [跳过] 评判失败")
            continue

        # 解析三组分数
        scores_a = parse_scores(re.search(r"模型A.*", judge_text).group() if re.search(r"模型A.*", judge_text) else "")
        scores_b = parse_scores(re.search(r"模型B.*", judge_text).group() if re.search(r"模型B.*", judge_text) else "")
        scores_c = parse_scores(re.search(r"模型C.*", judge_text).group() if re.search(r"模型C.*", judge_text) else "")

        print(f"  A={scores_a}  B={scores_b}  C={scores_c}")

        scored.append({
            "id":       item["id"],
            "role":     item["role"],
            "query":    item["query"],
            "reply_a":  item["reply_base"],
            "reply_b":  item["reply_b"],
            "reply_c":  item["reply_c"],
            "scores_a": scores_a,
            "scores_b": scores_b,
            "scores_c": scores_c,
            "judge_raw": judge_text
        })

        time.sleep(1.5)  # 避免触发限流

    save_json(scored, OUTPUT_DIR / "scores.json")
    print(f"\n✅ 打分完成，共 {len(scored)} 条")

    # 生成汇总报告
    generate_summary(scored)


# ============================================================
# 汇总报告生成
# ============================================================
def generate_summary(scored):
    """计算各维度均分，生成 README 可用的 Markdown 表格"""

    dims = ["角色一致性", "情绪共情度", "回复质量"]
    totals = {
        "A": {d: [] for d in dims},
        "B": {d: [] for d in dims},
        "C": {d: [] for d in dims},
    }

    for item in scored:
        for d in dims:
            va = item["scores_a"].get(d)
            vb = item["scores_b"].get(d)
            vc = item["scores_c"].get(d)
            if va: totals["A"][d].append(va)
            if vb: totals["B"][d].append(vb)
            if vc: totals["C"][d].append(vc)

    def avg(lst):
        return round(sum(lst) / len(lst), 2) if lst else 0.0

    avgs = {
        g: {d: avg(totals[g][d]) for d in dims}
        for g in ["A", "B", "C"]
    }
    for g in avgs:
        avgs[g]["平均分"] = round(
            sum(avgs[g][d] for d in dims) / len(dims), 2
        )

    lines = []
    lines.append("# RoleChat-7B 评测报告")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"评测样本：{len(scored)} 条  |  评判模型：{JUDGE_MODEL}")
    lines.append("")
    lines.append("## 各维度均分对比（1-5分）")
    lines.append("")
    lines.append("| 模型 | 角色一致性 | 情绪共情度 | 回复质量 | 平均分 |")
    lines.append("|------|-----------|-----------|---------|------|")
    lines.append(f"| Qwen2.5-7B（基座） | {avgs['A']['角色一致性']} | {avgs['A']['情绪共情度']} | {avgs['A']['回复质量']} | {avgs['A']['平均分']} |")
    lines.append(f"| RoleChat-7B SFT+DPO | {avgs['B']['角色一致性']} | {avgs['B']['情绪共情度']} | {avgs['B']['回复质量']} | {avgs['B']['平均分']} |")
    lines.append(f"| **RoleChat-7B SFT+DPO+RAG** | **{avgs['C']['角色一致性']}** | **{avgs['C']['情绪共情度']}** | **{avgs['C']['回复质量']}** | **{avgs['C']['平均分']}** |")
    lines.append("")
    lines.append("## 各角色分项均分")
    lines.append("")

    # 按角色细分
    roles = list({item["role"] for item in scored})
    lines.append("| 角色 | 基座均分 | SFT+DPO均分 | +RAG均分 |")
    lines.append("|------|---------|------------|---------|")
    for role in sorted(roles):
        role_items = [x for x in scored if x["role"] == role]
        def role_avg(g):
            vals = []
            for item in role_items:
                s = item[f"scores_{g.lower()}"]
                vals.extend([v for v in s.values() if v])
            return round(sum(vals)/len(vals), 2) if vals else 0
        lines.append(f"| {role} | {role_avg('A')} | {role_avg('B')} | {role_avg('C')} |")

    report = "\n".join(lines)
    report_path = OUTPUT_DIR / "summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    print(f"\n✅ 报告已保存至 {report_path}")


# ============================================================
# 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="RoleChat-7B 三组对比评测")
    parser.add_argument(
        "--step",
        choices=["base", "sft", "judge", "summary"],
        required=True,
        help="运行阶段：base → sft → judge"
    )
    parser.add_argument(
        "--api_key",
        default=SILICONFLOW_API_KEY,
        help="SiliconFlow API Key（仅 judge 阶段需要）"
    )
    args = parser.parse_args()

    if args.step == "base":
        step_base()
    elif args.step == "sft":
        step_sft()
    elif args.step == "judge":
        if not args.api_key:
            print("❌ --step judge 需要提供 --api_key")
            sys.exit(1)
        step_judge(args.api_key)
    elif args.step == "summary":
        scores_path = OUTPUT_DIR / "scores.json"
        if not scores_path.exists():
            print("❌ 未找到 scores.json，请先运行 --step judge")
            sys.exit(1)
        generate_summary(load_json(scores_path))


if __name__ == "__main__":
    main()