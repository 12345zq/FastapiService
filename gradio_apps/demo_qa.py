"""
智能问答 Gradio 演示界面（改编自 practice01/main.py）
使用 BGE-small-zh + FAISS 做本地语义检索，通过 Ollama deepseek-r1 生成回答
"""
import requests
import gradio as gr
from services.qa_service import qa_service
from services.creative_service import creative_service

def create_demo_app() -> gr.Blocks:
    with gr.Blocks(title="智能问答 & 创作系统") as demo:
        gr.Markdown("## 智能问答 / 创作系统")

        mode = gr.Dropdown(
            choices=["问答","创作"],
            value="创作",
            label="功能选择"
        )
        qa_input = gr.Textbox(label="输入问题", placeholder="请输入关于问答有关的问题...",visible=False)
        genre_input = gr.Dropdown(
            choices=["小说", "剧本", "诗歌", "文案"],
            value="小说",
            label="创作类型",
            visible=True
        )
        req_input = gr.Textbox(label="创作要求", placeholder="请输入创作要求...",visible=True)

        output = gr.Textbox(label="结果", placeholder="结果将在这里显示")

        def switch_mode(mode):
            if mode == "问答":
                return [
                    gr.update(visible=True),   # qa_input
                    gr.update(visible=False),  # genre_input
                    gr.update(visible=False),  # req_input
                ]
            else:
                return [
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                ]
        
        mode.change(
            fn=switch_mode,
            inputs=mode,
            outputs=[qa_input, genre_input, req_input]
        )
        def handle_query(mode: str, qa_question: str, genre: str, requirement: str) -> str:
            """根据下拉框模式分发到对应 service"""
            if mode == "问答":
                response = qa_service.ask(qa_question.strip())
                if not response["success"]:
                    return f"问答失败：{response['error']}"
                return response["answer"]
            else:
                response = creative_service.generate(genre.strip(), requirement.strip())
            if not response["success"]:
                return f"生成失败：{response['error']}"
            return response["result"]
        gr.Button("提交").click(
            fn = handle_query,
            inputs =[mode,qa_input,genre_input,req_input],
            outputs = output
        )
    return demo