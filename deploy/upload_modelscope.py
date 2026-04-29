from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility
import os

# ========== 配置区 ==========
MODELSCOPE_TOKEN = ""          # 填入你的 ModelScope API Token
MODEL_ID = "chenzhikang123/RoleChat-7B"   # 你的模型 ID（用户名/模型名）
LOCAL_MODEL_DIR = "./RoleChat-7B"         # 本地模型文件夹路径
COMMIT_MESSAGE = "upload RoleChat-7B weights"
# ============================

api = HubApi()
api.login(MODELSCOPE_TOKEN)

# 如果模型仓库不存在则先创建
try:
    api.create_model(
        model_id=MODEL_ID,
        visibility=ModelVisibility.PUBLIC,
        license=Licenses.APACHE_V2,
        chinese_name="RoleChat-7B 角色对话大模型",
    )
    print(f"[INFO] 模型仓库 {MODEL_ID} 创建成功")
except Exception as e:
    print(f"[INFO] 仓库已存在或创建跳过: {e}")

# 上传整个文件夹
api.push_model(
    model_id=MODEL_ID,
    model_dir=LOCAL_MODEL_DIR,
    commit_message=COMMIT_MESSAGE,
)

print(f"[DONE] 上传完成 → https://modelscope.cn/models/{MODEL_ID}")