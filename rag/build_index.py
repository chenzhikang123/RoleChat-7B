"""
build_index.py
==============
作用：读取 data/raw/characters/ 下所有角色 json，
      把每条知识片段向量化，存入 ChromaDB 本地数据库。

只需要运行一次。角色 json 有更新时重新运行即可。

运行方式：
    python rag/build_index.py
"""

import json
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# ============================================================
# 配置项
# ============================================================
# ChromaDB 存储路径（会自动创建）
CHROMA_PATH = r"D:\llm\RoleChat-7B\rag\chroma_db"

# 角色 json 所在目录
CHAR_DIR = r"D:\llm\RoleChat-7B\data\raw\characters"

# 向量模型（中文小模型，本地运行，首次会自动下载约 100MB）
# 也可以换成 "BAAI/bge-large-zh-v1.5" 效果更好但更慢
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


# ============================================================


def load_character(file_path: Path) -> dict:
    """读取单个角色 json 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def char_to_chunks(char: dict) -> list[dict]:
    """
    把一个角色 json 拆成多个独立的文本片段（chunk）。

    大白话：ChromaDB 存的是一条一条的文本，
    我们要把 json 里的每个字段都拆成单独的条目存进去，
    这样检索时才能精准命中对应的片段，而不是整个 json 一锅端。

    返回格式：[{"text": "...", "type": "...", "character": "..."}, ...]
    """
    name = char.get("name", "未知角色")
    chunks = []

    # 1. 人设（persona）
    if char.get("persona"):
        chunks.append({
            "text": f"【人设】{char['persona']}",
            "type": "persona",
            "character": name
        })

    # 2. 背景故事（background）
    if char.get("background"):
        chunks.append({
            "text": f"【背景】{char['background']}",
            "type": "background",
            "character": name
        })

    # 3. 说话风格（speaking_style）
    if char.get("speaking_style"):
        chunks.append({
            "text": f"【说话风格】{char['speaking_style']}",
            "type": "speaking_style",
            "character": name
        })

    # 4. 核心价值观（core_values，列表）
    for i, val in enumerate(char.get("core_values", [])):
        chunks.append({
            "text": f"【价值观】{val}",
            "type": "core_value",
            "character": name
        })

    # 5. 情绪应对策略（emotional_response_strategies，列表）—— 最重要
    for i, strategy in enumerate(char.get("emotional_response_strategies", [])):
        chunks.append({
            "text": f"【情绪应对】{strategy}",
            "type": "emotional_strategy",
            "character": name
        })

    # 6. 应该避免的行为（avoid_behaviors，列表）
    for i, avoid in enumerate(char.get("avoid_behaviors", [])):
        chunks.append({
            "text": f"【避免行为】{avoid}",
            "type": "avoid_behavior",
            "character": name
        })

    # 7. 知识库（knowledge，列表）—— 最多
    for i, knowledge in enumerate(char.get("knowledge", [])):
        chunks.append({
            "text": f"【知识】{knowledge}",
            "type": "knowledge",
            "character": name
        })

    return chunks


def build_index():
    """主函数：构建并保存向量索引"""

    # ----------------------------------------------------------
    # 1. 初始化向量模型
    # 大白话：这个模型负责把文本变成数字向量
    # ----------------------------------------------------------
    print(f"[1/4] 加载向量模型：{EMBED_MODEL}")
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    # ----------------------------------------------------------
    # 2. 初始化 ChromaDB
    # 大白话：ChromaDB 是本地向量数据库，数据会存在 CHROMA_PATH 文件夹里
    # 每次重新 build 会先删掉旧的 collection，再重建，保证数据干净
    # ----------------------------------------------------------
    print(f"[2/4] 初始化 ChromaDB，存储路径：{CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # 如果 collection 已存在就删掉重建（保证数据是最新的）
    try:
        client.delete_collection("role_knowledge")
        print("       已删除旧索引，重新构建")
    except Exception:
        pass

    collection = client.create_collection(
        name="role_knowledge",
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}  # 用余弦相似度做检索
    )

    # ----------------------------------------------------------
    # 3. 读取所有角色 json，拆成 chunks
    # ----------------------------------------------------------
    print(f"[3/4] 读取角色文件：{CHAR_DIR}")
    char_dir = Path(CHAR_DIR)
    if not char_dir.exists():
        raise FileNotFoundError(f"角色目录不存在：{CHAR_DIR}")

    all_docs = []  # 文本内容
    all_metas = []  # 元数据（角色名、类型）
    all_ids = []  # 唯一 ID

    json_files = list(char_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"在 {CHAR_DIR} 下没有找到任何 json 文件")

    for file in json_files:
        char = load_character(file)
        chunks = char_to_chunks(char)
        name = char.get("name", file.stem)
        print(f"       ✓ {name}：{len(chunks)} 条片段")

        for i, chunk in enumerate(chunks):
            all_docs.append(chunk["text"])
            all_metas.append({
                "character": chunk["character"],
                "type": chunk["type"],
                "source_file": file.stem
            })
            all_ids.append(f"{file.stem}_{i:04d}")

    # ----------------------------------------------------------
    # 4. 批量写入 ChromaDB
    # 大白话：把所有文本交给向量模型，转成数字，存进数据库
    # 这一步会花几秒到几十秒，视片段数量和机器性能而定
    # ----------------------------------------------------------
    print(f"[4/4] 向量化并写入数据库，共 {len(all_docs)} 条片段...")

    # 分批写入，避免一次太多报错（每批 100 条）
    batch_size = 100
    for i in range(0, len(all_docs), batch_size):
        collection.add(
            documents=all_docs[i:i + batch_size],
            metadatas=all_metas[i:i + batch_size],
            ids=all_ids[i:i + batch_size]
        )

    print(f"\n✅ 索引构建完成！共写入 {len(all_docs)} 条片段")
    print(f"   数据库位置：{os.path.abspath(CHROMA_PATH)}")
    print(f"\n各角色片段统计：")

    # 打印统计
    from collections import Counter
    char_counter = Counter(m["character"] for m in all_metas)
    for char_name, count in sorted(char_counter.items()):
        print(f"   {char_name}：{count} 条")


if __name__ == "__main__":
    build_index()