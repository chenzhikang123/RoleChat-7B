"""
api.py  （修复版 v3）
加入完整错误追踪：
    - 所有异常打印完整 traceback
    - 每次请求打印入参、出参、耗时
    - RAG 检索结果也打印出来方便排查
"""

import sys
import os
import time
import traceback
import logging

# ⚠️ 必须在所有 huggingface/transformers 相关 import 之前设置
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from rag.retriever import build_system_prompt

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("rolechat")

# ============================================================
# 配置区
# ============================================================
MODEL_PATH = os.path.join(ROOT_DIR, "output", "merged_model")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(ROOT_DIR, "output", "final_model")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "czk123123/RoleChat-7B"

MAX_NEW_TOKENS     = 512
TEMPERATURE        = 0.75
TOP_P              = 0.9
REPETITION_PENALTY = 1.1
PORT               = 8000

SUPPORTED_ROLES = [
    "知心朋友", "生活导师", "人生顾问", "幽默朋友",
    "心理咨询师", "温暖姐姐", "可靠兄长", "睿智长者",
]
# ============================================================

tokenizer = None
model     = None


def load_model():
    global tokenizer, model
    logger.info(f"模型路径：{MODEL_PATH}")
    logger.info("正在加载 tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        logger.info("tokenizer 加载完成")
    except Exception:
        logger.error("tokenizer 加载失败：\n" + traceback.format_exc())
        raise

    logger.info("正在加载模型权重（可能需要几十秒）...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True
        )
        model.eval()
        logger.info("✅ 模型加载完成！")
    except Exception:
        logger.error("模型加载失败：\n" + traceback.format_exc())
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    logger.info("服务已停止")


app = FastAPI(
    title="RoleChat-7B API",
    description="多角色情感对话大模型推理服务，集成 RAG 知识增强",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ——————————————————————————————————————————
# 全局异常捕获中间件
# 大白话：任何地方抛异常，都会在这里被捕获并打印完整堆栈
# ——————————————————————————————————————————
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录每个请求的路径、耗时，以及任何未捕获的异常"""
    start = time.time()
    logger.info(f"→ {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        elapsed = time.time() - start
        logger.info(f"← {request.method} {request.url.path} "
                    f"[{response.status_code}] {elapsed:.2f}s")
        return response
    except Exception:
        elapsed = time.time() - start
        logger.error(
            f"← {request.method} {request.url.path} [500] {elapsed:.2f}s\n"
            + traceback.format_exc()
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "服务器内部错误，详见终端日志"}
        )


# ——————————————————————————————————————————
# 数据结构
# ——————————————————————————————————————————
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    character: str
    messages: list[Message]
    use_rag: bool = True


class ChatResponse(BaseModel):
    reply: str
    character: str
    system_prompt: str
    use_rag: bool


# ——————————————————————————————————————————
# 核心推理
# ——————————————————————————————————————————
def generate_reply(
    character: str,
    messages: list[dict],
    use_rag: bool = True
) -> tuple[str, str]:

    # Step 1: 取用户最后一句话
    user_last_msg = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            user_last_msg = msg["content"]
            break
    logger.info(f"用户最后一句：{user_last_msg[:50]}...")

    # Step 2: RAG 检索
    if use_rag and user_last_msg:
        try:
            system_prompt = build_system_prompt(character, user_last_msg)
            logger.info(f"RAG system prompt 长度：{len(system_prompt)} 字符")
            logger.debug(f"System Prompt 内容：\n{system_prompt}")
        except Exception:
            logger.error("RAG 检索失败，降级为基础 prompt：\n" + traceback.format_exc())
            system_prompt = f"你是{character}，请保持角色设定，用温暖真诚的方式与用户对话。"
    else:
        system_prompt = f"你是{character}，请保持角色设定，用温暖真诚的方式与用户对话。"
        logger.info("RAG 已关闭，使用基础 prompt")

    # Step 3: 构建对话
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    logger.info(f"对话轮数：{len(messages)}，total messages：{len(full_messages)}")

    # Step 4: tokenize
    try:
        text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]
        logger.info(f"输入 token 数：{input_len}")
    except Exception:
        logger.error("Tokenize 失败：\n" + traceback.format_exc())
        raise

    # Step 5: 推理
    try:
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_time = time.time() - t0
        new_tokens = outputs[0][input_len:]
        output_len = len(new_tokens)
        logger.info(f"生成 token 数：{output_len}，耗时：{gen_time:.2f}s，"
                    f"速度：{output_len/gen_time:.1f} tokens/s")
    except Exception:
        logger.error("模型推理失败：\n" + traceback.format_exc())
        raise

    # Step 6: 解码
    try:
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        logger.info(f"模型回复（前100字）：{reply[:100]}")
    except Exception:
        logger.error("解码失败：\n" + traceback.format_exc())
        raise

    return reply, system_prompt


# ——————————————————————————————————————————
# 路由
# ——————————————————————————————————————————
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "device": str(next(model.parameters()).device) if model else "N/A"
    }


@app.get("/roles")
def get_roles():
    return {"roles": SUPPORTED_ROLES}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    logger.info(f"收到请求 → 角色：{request.character}，"
                f"消息数：{len(request.messages)}，RAG：{request.use_rag}")

    if model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")

    if request.character not in SUPPORTED_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的角色：{request.character}，可选：{SUPPORTED_ROLES}"
        )

    if not request.messages:
        raise HTTPException(status_code=400, detail="messages 不能为空")

    try:
        messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
        reply, system_prompt = generate_reply(
            character=request.character,
            messages=messages_dict,
            use_rag=request.use_rag
        )
    except HTTPException:
        raise
    except Exception:
        # 打印完整堆栈到终端，让用户能看到具体哪一行出错
        err_detail = traceback.format_exc()
        logger.error(f"推理过程异常：\n{err_detail}")
        raise HTTPException(
            status_code=500,
            detail=f"推理出错，请查看终端日志获取详细信息。错误摘要：{err_detail.splitlines()[-1]}"
        )

    logger.info(f"请求处理完成 ✓")
    return ChatResponse(
        reply=reply,
        character=request.character,
        system_prompt=system_prompt,
        use_rag=request.use_rag
    )


if __name__ == "__main__":
    logger.info("启动 RoleChat-7B API 服务")
    logger.info(f"接口文档：http://localhost:{PORT}/docs")
    uvicorn.run(app, host="0.0.0.0", port=PORT)