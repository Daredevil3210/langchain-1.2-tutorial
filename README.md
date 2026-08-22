# LangChain 1.2 学习教程

基于 **LangChain 1.x** 的大模型应用开发入门教程，通过 Jupyter Notebook 逐步演示大模型的初始化、调用、流式输出与批量处理。

## 环境要求

- Python 3.13+
- conda（可选，推荐：`conda create -n langchain1.2 python=3.13`）
- 依赖安装：

```bash
pip install -r requirements.txt
```

## 项目结构

```
langchain1.2_tutorial/
├── README.md                     # 本文件
├── requirements.txt              # 依赖清单
├── .env.example                  # 环境变量模板（复制为 .env 并填入你的 Key）
├── .gitignore
├── LICENSE                       # MIT License
├── chapter01_summary/            # 第 1 章：LangChain 概览
│   └── test.py                   # 环境自检脚本
└── chapter02_model/              # 第 2 章：模型调用
    ├── 01-model-init-online.ipynb    # 在线模型初始化（DeepSeek / 智谱 / OpenRouter / init_chat_model）
    ├── 02-model-init-params.ipynb    # 模型初始化参数详解
    ├── 03-model-init-ollama.ipynb    # 本地模型（Ollama）
    ├── 04-model-invoke.ipynb         # 模型调用（invoke）
    └── 05-model-stream-batch.ipynb   # 流式调用与批量调用
```

## 快速开始

1. 复制环境变量模板并填入自己的 API Key：

```bash
cp .env.example .env   # Windows: copy .env.example .env
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 启动 Jupyter 并按顺序运行 `chapter02_model/` 下的 notebook：

```bash
jupyter notebook
```

## API Key 配置

本项目同时演示了多个平台的模型接入，请在 `.env` 中配置（⚠️ 不要提交真实 Key，`.env` 已被 `.gitignore` 忽略）：

| 变量 | 平台 | 说明 |
|------|------|------|
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` | DeepSeek 官网 | `https://api.deepseek.com` |
| `ZHIPU_API_KEY` / `ZHIPU_BASE_URL` | 智谱 AI | `https://open.bigmodel.cn/api/paas/v4` |
| `OPENROUTER_API_KEY` / `OPENROUTER_API_BASE` | OpenRouter | 中转平台 |

## 章节内容

- **chapter01_summary**：LangChain 环境与版本自检
- **chapter02_model**
  - `01` 在线模型初始化：DeepSeek 官网、智谱、OpenRouter 三种接入方式，以及 LangChain 1.x 统一的 `init_chat_model()`
  - `02` 模型初始化参数：`max_tokens`、`temperature` 等参数的设置与影响
  - `03` 本地模型：通过 Ollama 运行本地大模型（如 `deepseek-r1:1.5b`）
  - `04` 模型调用：`invoke()` 同步调用与响应结构解析
  - `05` 流式与批量：`stream()` 流式输出、批量调用

## 许可证

[MIT](LICENSE) © 2026 崔学成