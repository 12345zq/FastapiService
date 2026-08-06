"""
智能医疗问答 Gradio 演示界面（改编自 practice01/main.py）
使用 BGE-small-zh + FAISS 做本地语义检索，通过 Ollama deepseek-r1 生成回答
"""
import requests
import gradio as gr
def gradio_ask(question: str) -> str:
    try:
        resp = requests.post(
            "http://localhost:8000/api/qa/ask",
            json={"question": question, "top_k": 3},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        return f"HTTP 错误：{e.response.status_code}"
    except Exception as e:
        return f"请求失败：{str(e)}"

def create_medical_qa_app() -> gr.Blocks:
    """创建医疗问答 Gradio 应用"""
    return gr.Interface(
        fn=gradio_ask,
        inputs=gr.Textbox(label="输入问题", placeholder="请输入关于医疗有关的问题..."),
        outputs=gr.Textbox(label="答案", placeholder="答案将在这里显示"),
        title="智能问答系统（医疗）",
        flagging_mode="never",
    )
