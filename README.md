# LangChain 1.2 学习教程

基于 **LangChain 1.x** 的大模型应用开发入门教程，通过 Jupyter Notebook 逐步演示大模型应用的完整开发链路：模型接入与调用、链路追踪、消息与提示词、工具调用、结构化输出、Agent 智能体与中间件。

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
├── README.md                        # 本文件
├── requirements.txt                 # 依赖清单
├── .env.example                     # 环境变量模板（复制为 .env 并填入你的 Key）
├── .gitignore                       # 已忽略 .env 等敏感文件
├── LICENSE                          # MIT License
├── chapter01_summary/               # 第 1 章：LangChain 概览
│   └── test.py                      # 环境自检脚本
├── chapter02_model/                 # 第 2 章：模型调用
│   ├── 01-model-init-online.ipynb       # 在线模型初始化（DeepSeek / 智谱 / OpenRouter / init_chat_model）
│   ├── 02-model-init-params.ipynb       # 模型初始化参数详解
│   ├── 03-model-init-ollama.ipynb       # 本地模型（Ollama）
│   ├── 04-model-invoke.ipynb            # 模型调用（invoke）
│   ├── 05-model-stream-batch.ipynb      # 流式调用与批量调用
│   └── 07-profile-initparams-config.ipynb  # profile 初始化参数配置
├── chapter03_langsmith/             # 第 3 章：LangSmith 可观测性
│   └── 01-langsmith-test.ipynb      # LangSmith 链路追踪与调试
├── chapter04_messages_prompt/       # 第 4 章：消息与提示词
│   ├── 01-messages的使用.ipynb          # 消息类型（System / Human / AI）与滑动窗口
│   ├── 03-ChatPromptTemplate的使用.ipynb    # ChatPromptTemplate 基础
│   └── 04-ChatPromptTemplate高级特性.ipynb  # 提示词高级特性
├── chapter05-tools/                 # 第 5 章：工具调用
│   ├── 01-tool使用的概述.ipynb          # 工具概述
│   ├── 02-tool_invoke_test.py           # 工具调用测试脚本
│   ├── 02-不使用@tool的方式定义工具.ipynb   # 普通函数定义工具
│   ├── 03-使用@tool装饰器定义工具.ipynb     # @tool 装饰器
│   └── 04-工具的应用案例.ipynb           # 工具综合案例
├── chapter06-structured_output/     # 第 6 章：结构化输出
│   ├── 01-Pydantic格式的使用.ipynb      # Pydantic 结构化输出
│   ├── 02-TypeDict格式的使用.ipynb      # TypedDict 结构化输出
│   └── 03-获取结构化结果方式.ipynb       # 结构化结果的获取方式
├── chapter07-Agents/                # 第 7 章：Agent 智能体
│   ├── 01-Agent的基本用法.ipynb         # Agent 基本用法
│   ├── 02-Agent的高级用法.ipynb         # Agent 高级用法
│   ├── 03-Agent的高级用法-ToolStrategy.ipynb    # 工具调用策略
│   ├── 04-Agent的高级用法-错误处理机制.ipynb     # 错误处理机制
│   ├── 05-Agent的高级用法-流式输出.ipynb        # 流式输出
│   └── 06-实战：多功能智能助手.ipynb     # 实战：多功能智能助手
└── chapter08-Middleware/            # 第 8 章：中间件
    ├── 01- SummarizationMiddleware中间件.ipynb    # 对话摘要中间件
    └── 02-HumanInTheLoopMiddleware中间件.ipynb    # 人工审批（人在回路）
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

3. 启动 Jupyter 并按顺序运行各章 notebook：

```bash
jupyter notebook
```

## API Key 配置

本项目同时演示了多个平台的模型接入，请在 `.env` 中配置（⚠️ 不要提交真实 Key，`.env` 已被 `.gitignore` 忽略，远程仓库中的 `.env.example` 仅为占位模板）：

| 变量 | 平台 | 说明 |
|------|------|------|
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` | DeepSeek 官网 | `https://api.deepseek.com` |
| `ZHIPU_API_KEY` / `ZHIPU_BASE_URL` | 智谱 AI | `https://open.bigmodel.cn/api/paas/v4` |
| `OPENROUTER_API_KEY` / `OPENROUTER_API_BASE` | OpenRouter | 中转平台，Base URL 需为 `https://openrouter.ai/api/v1` |

> 第 3 章 LangSmith 追踪另需在 `.env` 中配置 `LANGSMITH_TRACING=true`、`LANGSMITH_API_KEY`、`LANGSMITH_PROJECT`、`LANGSMITH_ENDPOINT`。

## 章节内容

| 章节 | 主题 | 要点 |
|------|------|------|
| **chapter01_summary** | LangChain 概览 | 环境与版本自检 |
| **chapter02_model** | 模型调用 | 在线模型初始化（DeepSeek/智谱/OpenRouter/`init_chat_model()`）、初始化参数（`max_tokens`/`temperature`/profile）、本地模型（Ollama）、`invoke()` 同步调用、`stream()` 流式与批量调用 |
| **chapter03_langsmith** | 可观测性 | LangSmith 链路追踪：记录每次模型调用、工具调用，在线调试与复现 |
| **chapter04_messages_prompt** | 消息与提示词 | 消息类型与滑动窗口（`keep_recent_messages`）、`ChatPromptTemplate` 基础（模板与拼接的区别）、提示词高级特性 |
| **chapter05-tools** | 工具调用 | 工具概述、普通函数与 `@tool` 装饰器两种定义方式、工具综合案例 |
| **chapter06-structured_output** | 结构化输出 | Pydantic / TypedDict 定义输出结构、`with_structured_output` 获取结构化结果 |
| **chapter07-Agents** | Agent 智能体 | 基本用法、高级用法（ToolStrategy 工具策略、错误处理机制、流式输出）、实战：多功能智能助手 |
| **chapter08-Middleware** | 中间件 | `SummarizationMiddleware` 对话摘要、`HumanInTheLoopMiddleware` 人工审批（中断工具调用 → 人工 approve/reject/edit → `Command(resume)` 恢复） |

## 许可证

[MIT](LICENSE) © 2026 崔学成