"""
OpenAI 兼容 Chat Completions API 统一封装

默认指向本地 Ollama 的 /v1 兼容端点（http://localhost:11434/v1，只需 ollama serve）
可通过环境变量切换为任意 OpenAI 兼容服务：
    OPENAI_API_BASE（默认派生自 OLLAMA_BASE_URL + "/v1"）
    OPENAI_API_KEY（本地 Ollama 忽略内容，默认占位 "ollama"）
"""
import base64
import functools
import logging
from openai import OpenAI

from config import OPENAI_API_BASE, OPENAI_API_KEY

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """返回 OpenAI 客户端单例（base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY）"""
    return OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)


def encode_image(image_path: str) -> str:
    """读取本地图片并返回 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def chat_completion(model: str, prompt: str, temperature: float = 0.7) -> str:
    """文本补全：POST /v1/chat/completions，返回 choices[0].message.content"""
    response = get_openai_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


def chat_completion_with_image(model: str, text: str, image_path: str,
                               temperature: float = 0.7) -> str:
    """带图补全：图片 base64 编码为 data URI 放入 content[].image_url（Ollama /v1 支持）"""
    image_data = encode_image(image_path)
    data_uri = f"data:image/jpeg;base64,{image_data}"
    response = get_openai_client().chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }],
        temperature=temperature,
    )
    return response.choices[0].message.content
