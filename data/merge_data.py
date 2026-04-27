import json
import os
import random
import re

random.seed(42)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def fix_input(input_text):
    """修复input字段，将场景描述转换为用户视角"""
    input_text = input_text.strip()

    # 如果以"用户"开头，进行替换
    if input_text.startswith("用户"):
        # 常见替换规则
        replacements = [
            ("用户刚刚失恋，情绪低落", "我刚刚失恋了，心情很差"),
            ("用户感到孤独，需要陪伴", "我最近感到很孤独，没人陪我"),
            ("用户被人误解，感到委屈", "我被人误解了，真的很委屈"),
            ("用户和朋友吵架了，感到委屈", "我和朋友吵架了，心里很难受"),
            ("用户和家人发生矛盾，心情复杂", "我和家人闹矛盾了，心情很复杂"),
            ("用户失去了一个重要的机会，很后悔", "我错过了一个重要机会，好后悔"),
            ("用户觉得自己不被人喜欢，很难过", "我觉得没人喜欢我，很难过"),
            ("用户暗恋的人喜欢了别人，心里很痛", "我暗恋的人喜欢了别人，心里好痛"),
            ("用户和多年好友闹翻了，很伤心", "我和多年的好友闹翻了，很伤心"),
            ("用户感觉被最好的朋友背叛了", "我感觉被最好的朋友背叛了"),
            ("用户工作压力很大，感到焦虑", "我工作压力好大，感到很焦虑"),
            ("用户考试失利，很沮丧", "我考试没考好，很沮丧"),
            ("用户对未来感到迷茫", "我对未来感到很迷茫，不知道该怎么办"),
            ("用户最近失眠，状态很差", "我最近一直失眠，整个人状态很差"),
            ("用户最近经济压力很大，很焦虑", "我最近经济压力很大，每天都很焦虑"),
            ("用户找工作屡屡碰壁，感到绝望", "我找工作一直碰壁，感觉快绝望了"),
            ("用户感觉自己一事无成，很自卑", "我感觉自己一事无成，真的很自卑"),
            ("用户家人生病，感到担心", "我家人生病了，我很担心他"),
            ("用户面临重要考试，紧张焦虑", "我马上要考试了，紧张得不行"),
            ("用户毕业找不到工作，感到压力巨大", "我毕业了但找不到工作，压力好大"),
            ("用户被领导批评，感到很挫败", "我被领导批评了，感觉很挫败"),
            ("用户拖延症严重，对自己很失望", "我拖延症好严重，对自己很失望"),
            ("用户感觉自己比不上身边的人，很焦虑", "我感觉自己比不上身边的人，很焦虑"),
            ("用户准备换工作，对新环境感到不安", "我准备换工作了，对新环境感到不安"),
            ("用户最近总是情绪不稳定，不知道为什么", "我最近情绪总是不稳定，不知道为什么"),
            ("用户今天过得很开心，想分享", "我今天过得很开心，想跟你分享"),
            ("用户刚完成一个重要目标，很有成就感", "我刚完成了一个重要目标，好有成就感"),
            ("用户收到了好消息，兴奋不已", "我刚收到了一个好消息，好兴奋"),
            ("用户交到了新朋友，心情很好", "我交到了新朋友，心情超好"),
            ("用户今天被人夸奖了，很开心", "我今天被人夸奖了，好开心"),
            ("用户不知道怎么拒绝别人，感到困扰", "我不知道怎么拒绝别人，感到很困扰"),
            ("用户觉得自己社交能力差，不敢主动交友", "我觉得自己社交能力很差，不敢主动交朋友"),
            ("用户在新环境中不知道如何融入", "我在新环境里不知道怎么融入"),
            ("用户不知道怎么和喜欢的人表白", "我不知道怎么向喜欢的人表白"),
            ("用户感觉朋友圈的人都过得比自己好", "我看朋友圈感觉大家都过得比我好"),
            ("用户不知道自己的人生目标是什么", "我不知道自己的人生目标是什么"),
            ("用户想改变自己但不知道从哪里开始", "我想改变自己但不知道从哪里开始"),
            ("用户觉得生活没有意思，很空虚", "我觉得生活没什么意思，很空虚"),
            ("用户想学一项新技能但总是坚持不下去", "我想学一项新技能但总是坚持不下去"),
            ("用户对自己的选择感到后悔，想重新来过", "我对自己的选择很后悔，真想重新来过"),
        ]

        for old, new in replacements:
            if old in input_text:
                return new

        # 没有匹配到的，通用替换
        input_text = input_text.replace("用户", "我", 1)
        # 去掉常见的描述性词语
        input_text = re.sub(r"，感到\w+$", "", input_text)
        input_text = re.sub(r"，很\w+$", "", input_text)

    return input_text


def filter_and_fix_synthetic(data):
    """过滤并修复合成数据"""
    results = []
    fixed = 0
    removed = 0

    for item in data:
        input_text = item["input"].strip()
        output_text = item["output"].strip()

        # 过滤占位符
        if "用户说的一句话" in input_text:
            removed += 1
            continue
        # 过滤太短
        if len(input_text) < 5:
            removed += 1
            continue
        if len(output_text) < 20:
            removed += 1
            continue

        # 修复input
        new_input = fix_input(input_text)
        if new_input != input_text:
            fixed += 1

        results.append({
            "instruction": item["instruction"],
            "input": new_input,
            "output": output_text
        })

    print(f"合成数据原始：{len(data)}条")
    print(f"修复input：{fixed}条")
    print(f"过滤掉：{removed}条")
    print(f"最终保留：{len(results)}条")
    return results


def merge_all():
    os.makedirs('./data/processed', exist_ok=True)

    # 加载BELLE数据
    print("加载BELLE开源数据...")
    belle_data = load_json('./data/processed/sft_train.json')
    print(f"BELLE数据：{len(belle_data)}条")

    # 加载合成数据
    print("加载合成数据...")
    synthetic_data = load_json('./data/processed/synthetic_train.json')

    # 过滤并修复合成数据
    print("过滤并修复合成数据...")
    synthetic_clean = filter_and_fix_synthetic(synthetic_data)

    # 合并
    all_data = belle_data + synthetic_clean

    # 打乱顺序
    random.shuffle(all_data)

    # 保存
    output_path = './data/processed/final_train.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print("-" * 50)
    print(f"BELLE数据：{len(belle_data)}条")
    print(f"合成数据（修复后）：{len(synthetic_clean)}条")
    print(f"合并总计：{len(all_data)}条")
    print(f"最终训练数据：{output_path}")


if __name__ == "__main__":
    merge_all()