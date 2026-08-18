# RAG 统一服务平台

基于 **FastAPI + Ollama + LangChain + ChromaDB + Gradio** 的多模块 RAG（检索增强生成）服务平台，提供知识库问答、内容创作、多模态图像分析、动态知识库四大能力，并内置 Gradio 演示界面。

---

## 一、方案介绍

### 1. 技术架构

```
用户请求
   │
   ▼
FastAPI 应用 (main.py)
   │── /api/qa          知识库问答（本地向量库 RAG）
   │── /api/creative    内容创作（创作技巧知识库 + 生成）
   │── /api/multimodal  多模态（图片分析 + 知识库综合报告）
   │── /api/knowledge   动态知识库（网页抓取 + 增量更新）
   │── /gradio/query    Gradio 演示界面（问答/创作切换）
   │
   ▼
Service 层 (services/)
   ├── qa_service        问答：Chroma 检索 → Ollama 生成
   ├── creative_service  创作：Chroma 检索 → Ollama 生成
   ├── multimodal_service 多模态：图片描述 + 向量检索 + 综合报告
   └── knowledge_service 动态库：网页抓取 → 增量入库 → 问答
   │
   ▼
Ollama (本地大模型服务, http://localhost:11434)
   ├── MFDoom/deepseek-r1-tool-calling:7b   （问答/创作/知识库）
   ├── bge-m3                                （问答/创作 向量模型）
   ├── nomic-embed-text                      （多模态/知识库 向量模型）
   └── llava:7b                              （多模态 图像理解）
```

### 2. 核心设计

- **向量库持久化**：四个模块各自独立的 Chroma 向量库（`db/demo01~04`），启动时检测到已有库则**直接加载**，不重复构建
- **数据与代码分离**：知识库源文档（`data/`）与向量库（`db/`）通过 Docker 卷从宿主机挂载，容器重建数据不丢失
- **Ollama 地址可配置**：通过环境变量 `OLLAMA_BASE_URL` 覆盖（默认 `localhost:11434`），本地与 Docker 部署切换无需改代码

### 3. 功能模块

| 模块 | API 路径 | 数据源 | 向量库 | 说明 |
|------|----------|--------|--------|------|
| 📚 知识库问答 | `/api/qa` | `data/text01`（大美安徽） | `db/demo01` | 基于本地知识库的 RAG 问答 |
| ✍️ 内容创作 | `/api/creative` | `data/text02`（创作技巧） | `db/demo02` | 按类型/要求生成内容 |
| 🖼️ 多模态分析 | `/api/multimodal` | `data/text03`（含图片） | `db/demo03` | 上传图片 → 描述 + 知识库综合分析 |
| 🔄 动态知识库 | `/api/knowledge` | 网页抓取 | `db/demo04` | 抓取网页 → 增量入库 → 问答 |
| 🖥️ 演示界面 | `/gradio/query` | — | — | Gradio 页面，下拉框切换问答/创作 |

## 二、目录结构

```
FastapiService/
├── main.py                  # FastAPI 入口 + Gradio 挂载 + 服务初始化
├── config.py                # 全局配置（模型、路径、Ollama 地址）
├── requirements.txt         # 本地开发依赖（含 practice 遗留依赖）
├── requirements-docker.txt  # Docker 部署依赖（精简锁定版）
├── Dockerfile               # 镜像构建（Python 3.14.6-slim）
├── docker-compose.yml       # 容器编排（data/db 本地挂载）
├── .dockerignore
├── data/                    # 知识库源文档（text01/02/03）
├── db/                      # 已有 Chroma 向量库（demo01/02/03）
├── routers/                 # API 路由层
├── schemas/                 # Pydantic 请求/响应模型
├── services/                # 业务服务层（RAG 核心）
├── gradio_apps/             # Gradio 演示应用
└── uploads/                 # 上传文件目录（运行时创建）
```

## 三、前置要求

- **Ollama**：本地安装并启动 `ollama serve`，并已拉取以下模型：

```bash
ollama pull MFDoom/deepseek-r1-tool-calling:7b
ollama pull bge-m3
ollama pull nomic-embed-text
ollama pull llava:7b
```

- **Python 3.14+**（本地运行）或 **Docker / Docker Compose**（容器部署）

## 四、本地运行

```bash
pip install -r requirements.txt

# 方式一：直接运行
python main.py

# 方式二：uvicorn 启动（推荐开发）
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问：

| 入口 | URL |
|------|-----|
| Swagger API 文档 | http://localhost:8000/docs |
| ReDoc 文档 | http://localhost:8000/redoc |
| 服务状态 | http://localhost:8000/ |
| Gradio 演示 | http://localhost:8000/gradio/query |

> 首次启动若 `db/` 为空会自动构建向量库；已有 `db/` 时直接加载。

## 五、Docker 部署

### 1. 部署方案说明

| 项目 | 方案 |
|------|------|
| 基础镜像 | `python:3.14.6-slim`（纯 Python uvicorn，避免编译依赖） |
| 依赖 | `requirements-docker.txt`，**锁定与本地一致版本**（chromadb 1.5.9 / gradio 6.22.0 等），保证已有向量库格式兼容 |
| 数据挂载 | `./data` → `/app/data`、`./db` → `/app/db`，**双向实时同步**，容器销毁数据不丢失 |
| Ollama 访问 | 环境变量 `OLLAMA_BASE_URL`，容器内经 `host.docker.internal:11434` 访问宿主机 Ollama |
| 端口 | `8000` |

### 2. Dockerfile

```dockerfile
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
```

### 3. docker-compose.yml

```yaml
services:
  rag:
    build:
      context: .
      dockerfile: Dockerfile
    image: fastapi-rag:latest
    container_name: fastapi-rag
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-http://host.docker.internal:11434}
    volumes:
      - ./data:/app/data
      - ./db:/app/db
    restart: unless-stopped
```

### 4. 执行命令

```bash
cd d:/source/FastapiService

# 构建并启动（首次构建约 5-10 分钟）
docker compose up -d --build

# 查看初始化日志（确认 4 个服务 ✅ 就绪）
docker logs -f fastapi-rag

# 停止
docker compose down
```

不使用 Compose 时，等效的 `docker run` 命令（PowerShell）：

```powershell
docker build -t fastapi-rag .
docker run -d --name fastapi-rag `
  -p 8000:8000 `
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 `
  -v d:/source/FastapiService/data:/app/data `
  -v d:/source/FastapiService/db:/app/db `
  fastapi-rag
```

Git Bash 执行需加 `MSYS_NO_PATHCONV=1` 前缀，防止路径被自动转换。

### 5. 验证

```bash
# 日志应依次出现
# ✅ 知识库问答服务已就绪 / 内容创作 / 多模态 / 动态知识库
# 出现"加载已有向量库"而非"首次构建" = 本地 db 加载成功

curl http://localhost:8000/          # 服务状态 JSON
# 浏览器打开 http://localhost:8000/gradio/query
```

### 6. 常见问题

| 现象 | 处理 |
|------|------|
| 启动日志提示无法连接 Ollama | 宿主机先执行 `ollama serve`，且已 pull 上述 4 个模型 |
| 向量库加载失败（版本不兼容） | 确认 `requirements-docker.txt` 中 chromadb 版本与构建 db 的环境一致 |
| Python 3.14 上某依赖无 wheel 构建失败 | 将 Dockerfile 基础镜像改为 `python:3.12-slim` 重新构建 |
| 通过知识库 API 新增内容不生效 | 确认 `./db` 挂载成功（`docker compose config` 查看 volumes） |

## 六、常用 API 示例

```bash
# 知识库问答
curl -X POST http://localhost:8000/api/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "黄山有哪些著名景点？", "top_k": 3}'

# 内容创作
curl -X POST http://localhost:8000/api/creative/generate \
  -H "Content-Type: application/json" \
  -d '{"genre": "诗歌", "requirements": "写一首赞美黄山的诗"}'

# 知识库问答重载（增量更新后刷新向量库）
curl -X POST http://localhost:8000/api/qa/reload

# 动态知识库：网页抓取
curl -X POST http://localhost:8000/api/knowledge/fetch \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

详细请求/响应字段见 Swagger：`http://localhost:8000/docs`。

## 七、文档与入口 URL 汇总

| 用途 | URL |
|------|-----|
| Swagger UI（交互式调试） | http://localhost:8000/docs |
| ReDoc（只读文档） | http://localhost:8000/redoc |
| 服务状态（JSON） | http://localhost:8000/ |
| Gradio 演示界面 | http://localhost:8000/gradio/query |
| 知识库问答 API | POST http://localhost:8000/api/qa/ask |
| 内容创作 API | POST http://localhost:8000/api/creative/generate |
| 多模态分析 API | POST http://localhost:8000/api/multimodal/analyze |
| 动态知识库 API | POST http://localhost:8000/api/knowledge/fetch |
