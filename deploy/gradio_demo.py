"""
gradio_demo.py  （修复版 v3）
修复：history 格式兼容（旧 tuple 格式 → 新 messages 格式）
"""

import sys
import os
import argparse
import requests

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import gradio as gr

API_BASE_URL = "http://localhost:8000"

SUPPORTED_ROLES = [
    "知心朋友", "生活导师", "人生顾问", "幽默朋友",
    "心理咨询师", "温暖姐姐", "可靠兄长", "睿智长者",
]

ROLE_DESC = {
    "知心朋友":  "🫂 温柔体贴，善于倾听，给你情感支持",
    "生活导师":  "🌟 积极乐观，帮你走出低谷，找回动力",
    "人生顾问":  "🧩 理性温和，分析问题，提供务实建议",
    "幽默朋友":  "😄 幽默风趣，用笑声化解你的压力",
    "心理咨询师":"🧠 专业引导，帮你探索情绪背后的根源",
    "温暖姐姐":  "💗 亲切自然，像家人一样关心你",
    "可靠兄长":  "💪 直接靠谱，关键时刻陪你一起扛",
    "睿智长者":  "🌿 沉稳从容，用人生经验帮你看清方向",
}


# ——————————————————————————————————————————
# 核心修复：格式转换函数
# 大白话：不管历史记录是什么格式，统一转成新格式
# ——————————————————————————————————————————
def normalize_history(history: list) -> list:
    """
    把任意格式的 history 统一转成 messages 格式。

    兼容三种情况：
        旧 tuple 格式：[["你好", "我也好"], ...]
        旧 list 格式： [["你好", "我也好"], ...]
        新 dict 格式： [{"role": "user", "content": "你好"}, ...]

    全部统一转成：
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    """
    if not history:
        return []

    normalized = []
    for item in history:
        # 已经是新格式（字典）
        if isinstance(item, dict) and "role" in item and "content" in item:
            normalized.append(item)
        # 旧格式（列表或元组）：[user_msg, assistant_msg]
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, assistant_msg = item
            if user_msg:
                normalized.append({"role": "user", "content": str(user_msg)})
            if assistant_msg:
                normalized.append({"role": "assistant", "content": str(assistant_msg)})
        # 其他未知格式：跳过
        else:
            continue

    return normalized


def call_api(character: str, history: list, use_rag: bool) -> tuple[str, str]:
    """调用 api.py，history 已经是标准 messages 格式"""
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("content") is not None
    ]

    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"character": character, "messages": messages, "use_rag": use_rag},
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        return data["reply"], data["system_prompt"]

    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到后端服务，请先启动 api.py", ""
    except requests.exceptions.Timeout:
        return "❌ 请求超时，模型生成时间过长", ""
    except requests.exceptions.HTTPError as e:
        # 把 api 返回的具体错误信息显示出来，方便排查
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return f"❌ API 错误 {e.response.status_code}：{detail}", ""
    except Exception as e:
        return f"❌ 未知错误：{str(e)}", ""


def user_submit(user_input, history):
    if not user_input.strip():
        return history, ""
    # 先做格式标准化，再追加新消息
    history = normalize_history(history)
    history = history + [{"role": "user", "content": user_input}]
    return history, ""


def bot_reply(history, character, use_rag):
    history = normalize_history(history)
    if not history or history[-1]["role"] != "user":
        return history, ""

    reply, system_prompt = call_api(character, history, use_rag)
    history = history + [{"role": "assistant", "content": reply}]
    return history, system_prompt


def clear_history():
    return [], ""


def update_role_desc(character):
    return ROLE_DESC.get(character, "")


def build_interface():
    with gr.Blocks(
        title="RoleChat-7B 多角色情感对话",
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="pink",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Noto Sans SC")
        ),
        css="""
        .gradio-container { max-width: 960px !important; margin: 0 auto; }
        .title-area { text-align: center; padding: 24px 0 8px 0; }
        .title-area h1 {
            font-size: 2rem; font-weight: 700;
            background: linear-gradient(135deg, #7c3aed, #db2777);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        }
        .title-area p { color: #64748b; font-size: 0.95rem; }
        .role-desc {
            font-size: 0.9rem; color: #7c3aed; padding: 6px 12px;
            background: #f5f3ff; border-radius: 8px; border-left: 3px solid #7c3aed;
        }
        #chatbot { height: 480px; border-radius: 12px; }
        .system-prompt-box textarea {
            font-size: 0.8rem !important; color: #475569 !important;
            font-family: monospace !important;
        }
        """
    ) as demo:

        gr.HTML("""
        <div class="title-area">
            <h1>🎭 RoleChat-7B</h1>
            <p>基于 Qwen2.5-7B 的多角色情感对话大模型 · SFT + DPO + RAG</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### 🎭 选择角色")
                character = gr.Dropdown(
                    choices=SUPPORTED_ROLES,
                    value="知心朋友",
                    label="当前角色",
                    interactive=True
                )
                role_desc_box = gr.Markdown(
                    value=ROLE_DESC["知心朋友"],
                    elem_classes="role-desc"
                )
                gr.Markdown("### ⚙️ 设置")
                use_rag = gr.Checkbox(
                    value=True,
                    label="启用 RAG 知识增强",
                    info="开启后模型会检索角色知识库，回复更精准"
                )
                clear_btn = gr.Button("🗑️ 清空对话", variant="secondary", size="sm")
                gr.Markdown("""
                ---
                **模型信息**
                - 基座：Qwen2.5-7B-Instruct
                - 微调：QLoRA SFT + DPO
                - 增强：RAG 知识库
                """)

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="对话",
                    elem_id="chatbot",
                    type="messages",
                    show_label=False,
                    avatar_images=(
                        None,
                        "https://api.dicebear.com/7.x/bottts/svg?seed=rolechat"
                    ),
                    height=480,
                )
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="说点什么吧... （Enter 发送，Shift+Enter 换行）",
                        show_label=False,
                        lines=2,
                        scale=5
                    )
                    send_btn = gr.Button("发送 ➤", variant="primary", scale=1, min_width=80)

                with gr.Accordion("🔍 查看 System Prompt（调试）", open=False):
                    system_prompt_display = gr.Textbox(
                        label="当前使用的 System Prompt（含 RAG 检索结果）",
                        lines=8,
                        interactive=False,
                        elem_classes="system-prompt-box"
                    )

        gr.Markdown("""
        > 💡 **提示**：切换角色会保留对话历史，点击「清空对话」开始新的对话。
        > 如遇历史记录格式错误，点「清空对话」即可恢复正常。
        """)

        character.change(fn=update_role_desc, inputs=character, outputs=role_desc_box)

        user_input.submit(
            fn=user_submit,
            inputs=[user_input, chatbot],
            outputs=[chatbot, user_input]
        ).then(
            fn=bot_reply,
            inputs=[chatbot, character, use_rag],
            outputs=[chatbot, system_prompt_display]
        )

        send_btn.click(
            fn=user_submit,
            inputs=[user_input, chatbot],
            outputs=[chatbot, user_input]
        ).then(
            fn=bot_reply,
            inputs=[chatbot, character, use_rag],
            outputs=[chatbot, system_prompt_display]
        )

        clear_btn.click(fn=clear_history, outputs=[chatbot, system_prompt_display])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if resp.json().get("model_loaded"):
            print("[✅] 检测到 API 服务正常运行，模型已加载")
        else:
            print("[⚠️] API 服务在运行但模型未加载，请稍等")
    except Exception:
        print(f"[⚠️] 未检测到 API 服务（{API_BASE_URL}），请先启动 deploy/api.py")

    print(f"\n[启动] Gradio 界面：http://localhost:{args.port}")
    demo = build_interface()
    demo.launch(server_port=args.port, share=args.share, inbrowser=True)