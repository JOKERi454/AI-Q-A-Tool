# 📄 PDF 问答工具 — 基于 RAG 的文档智能助手

基于检索增强生成（RAG）的 PDF 问答工具，支持上传课件、论文、教材等 PDF 文档，并针对内容自由提问。使用 **LangChain** + **DeepSeek** + **Chroma** 构建，具备多轮对话和原文引用能力。

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.x-green)
![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ 功能特性

- **📤 PDF 上传** — 支持拖拽上传单个或多个 PDF 文件（课件、论文、教材等）
- **💬 多轮对话** — 支持追问，系统自动结合对话历史理解上下文
- **📚 原文引用** — 每个回答附带 `[第N页]` 格式的出处标记，可展开查看原文片段
- **🔍 范围筛选** — 可按指定文档检索，也可跨所有已上传 PDF 搜索
- **🧠 历史感知检索** — "刚才那个概念再详细解释一下" 这类追问会自动结合对话上下文改写为独立查询
- **🌐 国内网络友好** — 内置 HuggingFace 镜像支持，无需科学上网即可下载模型

## 🏗️ 架构设计

```
┌──────────┐     ┌──────────────┐     ┌────────────────┐
│  PDF(s)  │────▶│  文本切块    │────▶│  向量嵌入      │
│  上传    │     │  (1K 字符)  │     │  (Chroma DB)   │
└──────────┘     └──────────────┘     └───────┬────────┘
                                              │
┌──────────┐     ┌──────────────┐             │  检索
│  回答    │◀────│  DeepSeek    │◀────────────┘  (Top-K)
│  + 引用  │     │  大模型      │
└──────────┘     └──────────────┘
       ▲                ▲
       │                │
  ┌────┴────────────────┴────┐
  │  历史感知检索器 + 记忆   │
  └──────────────────────────┘
```

**处理流程**：上传 PDF → 提取文本 → 切块（1000 字符/块，200 字符重叠）→ 生成向量嵌入（`all-MiniLM-L6-v2`）→ 存入 Chroma → 检索 Top-4 相关块 → DeepSeek 结合上下文生成回答并标注 `[第N页]` 引用 → 存入对话记忆供后续追问

## 🛠️ 技术栈

| 层级 | 技术选型 |
|---|---|
| **大模型** | DeepSeek（v4-flash / v4-pro），通过 OpenAI 兼容接口调用 |
| **嵌入模型** | `sentence-transformers/all-MiniLM-L6-v2`（384维，本地运行） |
| **向量数据库** | Chroma + LangChain-Chroma |
| **开发框架** | LangChain 1.x（链式调用使用 `langchain-classic`） |
| **PDF 解析** | `pypdf` |
| **界面** | Streamlit |
| **对话记忆** | 自定义滑动窗口 MemoryManager（保留最近 10 轮对话） |

## 🚀 快速开始

### 环境要求

- Python 3.10+
- [DeepSeek API Key](https://platform.deepseek.com/)（需注册并充值）

### 1. 克隆项目 & 安装依赖

```bash
git clone https://github.com/your-username/pdf-qa-tool.git
cd pdf-qa-tool

# 推荐创建虚拟环境
python -m venv venv

# Windows 激活
venv\Scripts\activate
# macOS / Linux 激活
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 DeepSeek API Key
# DEEPSEEK_API_KEY=sk-你的真实key
```

### 3. 启动

```bash
streamlit run app.py
```

浏览器访问 **http://localhost:8501**，上传 PDF，点击 **Process PDFs** 即可开始提问！

## ⚙️ 配置说明

所有配置项在 `.env` 文件中设置（`core/config.py` 提供默认值）：

```env
# DeepSeek API（必填）
DEEPSEEK_API_KEY=sk-你的key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# LLM 模型选择
LLM_MODEL=deepseek-v4-flash    # 快速且够用（默认）
# LLM_MODEL=deepseek-v4-pro    # 更强能力，稍慢

# 嵌入模型（默认本地运行）
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# HuggingFace 镜像（国内用户必设）
HF_ENDPOINT=https://hf-mirror.com
```

### 可调参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `CHUNK_SIZE` | 1000 | 每个文本块的字符数 |
| `CHUNK_OVERLAP` | 200 | 相邻文本块的重叠字符数 |
| `RETRIEVAL_K` | 4 | 每次检索返回的文本块数量 |
| `SEARCH_TYPE` | `similarity` | 检索策略（`similarity` = 相似度 / `mmr` = 最大边际相关性） |
| `LLM_TEMPERATURE` | 0.3 | 大模型输出随机性（0=确定, 1=创造性） |
| `LLM_MAX_TOKENS` | 2048 | 回答最大 token 数 |
| `MAX_HISTORY` | 10 | 保留的对话轮数 |

## 📁 项目结构

```
pdf-qa-tool/
├── app.py                      # Streamlit 前端界面
├── core/
│   ├── config.py               # 集中配置 & 环境变量校验
│   ├── pdf_loader.py           # 基于 pypdf 的文本提取
│   ├── chunker.py              # 文本递归切块
│   ├── embeddings_store.py     # Chroma 向量库 + 嵌入管理
│   ├── rag_chain.py            # LangChain RAG 流程编排
│   ├── citation_tracker.py     # 引用提取 & 格式化
│   └── memory_manager.py       # 滑动窗口对话记忆
├── data/                       # 运行时自动创建
│   ├── uploads/                # 上传的 PDF 临时文件
│   └── chroma_db/              # 持久化向量库
├── requirements.txt
├── .env.example
└── README.md
```

## 💡 使用技巧

- **入门提问**：「请总结这篇文档的核心观点」、「这篇文章主要讲了什么？」
- **精准定位**：「第 3.2 节关于实验方法说了什么？」
- **追问深入**：「刚才那个概念能展开讲讲吗？」——系统会自动结合上文理解
- **跨文档对比**：上传多篇论文后问「这两篇文章的方法有什么异同？」
- **范围筛选**：在侧边栏下拉菜单中选择特定文档，仅在该文档范围内搜索

## 🔧 国内用户注意事项

1. **HuggingFace 镜像**：`.env` 中默认已配置 `HF_ENDPOINT=https://hf-mirror.com`，嵌入模型首次运行时会自动从镜像下载（约 80 MB，仅需一次）
2. **DeepSeek API**：`api.deepseek.com` 国内直连，无需代理
3. 如需更换其他 HuggingFace 镜像，修改 `HF_ENDPOINT` 即可

## 📄 开源协议

MIT
