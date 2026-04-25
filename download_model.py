from modelscope import snapshot_download
import os

os.makedirs('./models', exist_ok=True)

print("开始下载Qwen2.5-7B-Instruct，约14GB，请耐心等待...")
snapshot_download(
    'Qwen/Qwen2.5-7B-Instruct',
    cache_dir='./models'
)
print("模型下载完成！")