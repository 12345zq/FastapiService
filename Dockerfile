# RAG 统一服务平台 - Docker 镜像
# 基础镜像：Python 3.14.6（slim 精简版）
FROM python:3.14.6-slim

WORKDIR /app

# 安装依赖（先于代码复制，充分利用构建缓存）
COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码（data/、db/ 不写入镜像，由 docker-compose 从宿主机挂载）
COPY config.py main.py ./
COPY routers/ ./routers/
COPY schemas/ ./schemas/
COPY services/ ./services/
COPY gradio_apps/ ./gradio_apps/

# 上传文件目录（多模态/知识库接口使用）
RUN mkdir -p uploads

EXPOSE 8000

# 纯 Python uvicorn（不用 [standard]，避免 uvloop 等编译依赖在 3.14 上出问题）
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
