"""
电影智能推荐 Gradio 演示界面（改编自 practice02/main.py）
使用 BGE-small-zh + FAISS 做语义相似度匹配，输入偏好关键词推荐电影
"""
import os
import random
import gradio as gr
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, BertModel

from config import BGE_MODEL_PATH, MOVIES_DATA_DIR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 3


def create_movie_recommend_app() -> gr.Blocks:
    """创建电影推荐 Gradio 应用"""

    # 加载电影数据
    movies = []
    data_dir = str(MOVIES_DATA_DIR)
    if os.path.isdir(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                    content = f.read().strip().split("\n", 1)
                    if len(content) == 2:
                        movies.append({"title": content[0], "description": content[1]})

    if not movies:
        def _no_data(x=None):
            return "错误：未找到电影数据文件，请确认 practice_data/movies/ 目录下有 .txt 文件"
        return gr.Interface(
            fn=_no_data,
            inputs=gr.Textbox(label="输入偏好（可选）"),
            outputs=gr.Textbox(label="推荐结果"),
            title="电影智能推荐系统",
        )

    # 加载 BGE 模型
    tokenizer = AutoTokenizer.from_pretrained(str(BGE_MODEL_PATH))
    model = BertModel.from_pretrained(str(BGE_MODEL_PATH)).to(DEVICE)

    def get_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        return torch.nn.functional.normalize(
            outputs.last_hidden_state[:, 0], p=2, dim=1
        ).cpu().numpy()

    # 构建 FAISS 索引
    movie_descriptions = [m["description"] for m in movies]
    embeddings = np.vstack([get_embedding(desc) for desc in movie_descriptions])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    def recommend(query=None):
        if query and query.strip():
            query_embedding = get_embedding(query)
        else:
            random_movie = random.choice(movie_descriptions)
            query_embedding = get_embedding(random_movie)
        distances, indices = index.search(query_embedding, TOP_K)
        return "\n\n".join([
            f"{movies[i]['title']}\n{movies[i]['description'][:100]}..."
            for i in indices[0]
        ])

    return gr.Interface(
        fn=lambda x: recommend(x) if x else recommend(),
        inputs=gr.Textbox(label="输入偏好（可选）", placeholder="例如：科幻太空冒险..."),
        outputs=gr.Textbox(label="推荐结果"),
        title="电影智能推荐系统",
        description="输入偏好关键词或直接点击提交获取推荐",
        allow_flagging="never",
        examples=[
            ["科幻时间旅行"],
            ["浪漫爱情故事"],
            ["神话传说故事"],
        ],
    )
