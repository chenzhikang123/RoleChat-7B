"""
retriever.py
============
作用：接收用户输入的文本，从 ChromaDB 里检索最相关的知识片段，
      拼成字符串返回，用于注入 system prompt。

大白话：
    用户说"我今天失恋了"
    → 把这句话变成向量
    → 在数据库里找数字最接近的几条知识
    → 把找到的知识拼成一段话
    → 塞进 system prompt，让模型"知道该怎么回答"

不需要手动运行，被 api.py / gradio_demo.py 调用。
"""

import os

# 强制离线模式：bge 模型已在本地缓存，不需要联网验证
# 必须在 import sentence_transformers / chromadb 之前设置
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import chromadb
from chromadb.utils import embedding_functions

# ============================================================
# 配置项（和 build_index.py 保持一致）
# ============================================================
CHROMA_PATH = r"D:\llm\RoleChat-7B\rag\chroma_db"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
# ============================================================

# 全局单例，避免每次检索都重新加载模型（很慢）
_client = None
_collection = None
_emb_fn = None


def _get_collection():
    """
    懒加载：第一次调用时初始化，之后复用。
    大白话：只在第一次用到的时候才加载模型，以后直接用，不重复加载。
    """
    global _client, _collection, _emb_fn

    if _collection is not None:
        return _collection

    print("[RAG] 初始化检索器...")
    _emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    _client = chromadb.PersistentClient(path=CHROMA_PATH)
    _collection = _client.get_collection(
        name="role_knowledge",
        embedding_function=_emb_fn
    )
    print(f"[RAG] 检索器就绪，数据库共 {_collection.count()} 条片段")
    return _collection


def retrieve(
    query: str,
    character: str = None,
    top_k: int = 4,
    type_filter: list[str] = None
) -> str:
    """
    检索与 query 最相关的知识片段。

    参数：
        query      : 用户输入的文本（用来做语义匹配）
        character  : 角色名（如"知心朋友"），只检索该角色的知识；None 则全局检索
        top_k      : 返回最相关的前 K 条，默认 4
        type_filter: 只检索特定类型，如 ["emotional_strategy", "knowledge"]
                     None 则不过滤类型

    返回：
        拼接好的字符串，格式如下（直接可以塞进 system prompt）：

        【参考知识】
        - 共情的核心是让对方感到"被看见"...
        - 当用户说"我不知道为什么"时...
    """
    collection = _get_collection()

    # ----------------------------------------------------------
    # 构建过滤条件
    # 大白话：如果指定了角色，就只在这个角色的数据里找
    # ----------------------------------------------------------
    where = None
    if character and type_filter:
        where = {
            "$and": [
                {"character": {"$eq": character}},
                {"type": {"$in": type_filter}}
            ]
        }
    elif character:
        where = {"character": {"$eq": character}}
    elif type_filter:
        where = {"type": {"$in": type_filter}}

    # ----------------------------------------------------------
    # 执行检索
    # 大白话：把 query 变成向量，在数据库里找最近的 top_k 条
    # ----------------------------------------------------------
    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        print(f"[RAG] 检索出错：{e}")
        return ""

    docs = results["documents"][0]       # 文本列表
    distances = results["distances"][0]  # 距离列表（越小越相关）

    if not docs:
        return ""

    # ----------------------------------------------------------
    # 过滤掉相关性太低的片段
    # 大白话：余弦距离 > 0.6 说明这条知识和用户说的话关系不大，丢掉
    # ----------------------------------------------------------
    filtered = [
        doc for doc, dist in zip(docs, distances)
        if dist < 0.6
    ]

    if not filtered:
        return ""

    # 去掉前缀标签（【知识】、【情绪应对】等），让文本更干净
    cleaned = []
    for doc in filtered:
        for prefix in ["【人设】", "【背景】", "【说话风格】", "【价值观】",
                        "【情绪应对】", "【避免行为】", "【知识】"]:
            doc = doc.replace(prefix, "")
        cleaned.append(doc.strip())

    context = "\n".join(f"- {d}" for d in cleaned)
    return f"\n\n【参考知识】\n{context}"


def retrieve_persona(character: str) -> str:
    """
    专门检索某个角色的人设 + 说话风格，
    用于构建 system prompt 的基础部分。

    大白话：每次对话开始时，先把这个角色"是谁、怎么说话"查出来。
    """
    collection = _get_collection()

    results = collection.query(
        query_texts=[character],
        n_results=3,
        where={
            "$and": [
                {"character": {"$eq": character}},
                {"type": {"$in": ["persona", "speaking_style", "background"]}}
            ]
        },
        include=["documents"]
    )

    docs = results["documents"][0]
    if not docs:
        return f"你是{character}，请保持角色设定进行对话。"

    combined = " ".join(doc for doc in docs)
    for prefix in ["【人设】", "【背景】", "【说话风格】"]:
        combined = combined.replace(prefix, "")

    return combined.strip()


def build_system_prompt(character: str, user_query: str) -> str:
    """
    组合：角色人设 + 检索到的相关知识 → 完整 system prompt

    大白话：
        1. 先查这个角色的基础人设（固定部分）
        2. 再根据用户这句话，检索最相关的知识（动态部分）
        3. 拼在一起给模型

    这是最终暴露给 api.py / gradio_demo.py 使用的主函数。
    """
    # 基础人设（固定）
    persona = retrieve_persona(character)

    # 动态检索：优先找情绪应对策略和知识
    dynamic_context = retrieve(
        query=user_query,
        character=character,
        top_k=4,
        type_filter=["emotional_strategy", "knowledge", "avoid_behavior"]
    )

    system_prompt = f"{persona}{dynamic_context}"
    return system_prompt


# ============================================================
# 简单测试（直接运行此文件时执行）
# ============================================================
if __name__ == "__main__":
    test_cases = [
        ("知心朋友", "我今天失恋了，心情很差"),
        ("心理咨询师", "我总是控制不住地焦虑，不知道为什么"),
        ("生活导师", "我感觉自己什么都做不好，想放弃了"),
        ("幽默朋友", "上班好无聊，老板又烦我"),
        ("睿智长者", "我不知道我的人生意义是什么"),
    ]

    print("=" * 60)
    print("RAG 检索测试")
    print("=" * 60)

    for character, query in test_cases:
        print(f"\n角色：{character}")
        print(f"用户：{query}")
        prompt = build_system_prompt(character, query)
        print(f"System Prompt：\n{prompt}")
        print("-" * 60)