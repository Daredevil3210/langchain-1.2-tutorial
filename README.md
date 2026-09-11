# LangChain 1.2 学习教程

基于 **LangChain 1.x** 的大模型应用开发入门教程，通过 Jupyter Notebook 逐步演示大模型应用的完整开发链路：模型接入与调用、链路追踪、消息与提示词、工具调用、结构化输出、Agent 智能体、中间件、记忆，以及 RAG 文档加载、切分、嵌入、Milvus 向量检索与知识库问答。

## 最新更新

[v0.8.0：Milvus 与客服知识库实战](https://github.com/Daredevil3210/langchain-1.2-tutorial/releases/tag/v0.8.0)

- 新增 Milvus 基础教程：数据库与集合管理、向量写入、主键查询和相似度检索。
- 新增 Atguigu Assistant 客服知识库案例：文档切分、向量入库、上下文检索与 Agent 回答生成。
- 补充 Milvus 运行前提与案例配置，修正案例中的检索参数和无用导入；详细记录见 [发布说明](releases/v0.8.0.md)。

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
├── requirements_full.txt             # RAG/记忆等完整依赖清单
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
├── chapter08-Middleware/            # 第 8 章：中间件
│   ├── 01- SummarizationMiddleware中间件.ipynb    # 对话摘要中间件
│   ├── 02-HumanInTheLoopMiddleware中间件.ipynb    # 人工审批（人在回路）
│   ├── 03-PIIMiddleware中间件.ipynb                # 个人信息识别与脱敏
│   ├── 04-TodoListMiddleware中间件.ipynb           # 待办事项中间件
│   ├── 05-其它内置中间件.ipynb                     # 其它内置中间件示例
│   ├── 06-自定义中间件-Node-style hooks.ipynb      # 节点式自定义中间件
│   ├── 07-自定义中间件-Wrap-style hooks.ipynb      # 包装式自定义中间件
│   ├── 08-hook函数的执行顺序.ipynb                  # hook 函数的执行顺序
│   └── fake_deepseek_server.py                     # 本地模拟 DeepSeek 服务
├── chapter09-memory/                 # 第 9 章：记忆
│   ├── 01-agent的记忆测试.ipynb           # Agent 记忆测试
│   ├── 02-短期记忆.ipynb                  # 短期记忆
│   ├── 03-记忆治理策略.ipynb              # 记忆治理策略（消息删除 / 摘要）
│   ├── 04-长期记忆-基础API的使用.ipynb     # 长期记忆：基础 API（含 PostgreSQL 存储）
│   └── 05-长期记忆-agent.ipynb            # 长期记忆：Agent 集成
├── chapter10-RAG/                    # 第 10 章：RAG 与 Milvus 知识库实战
│   ├── 01-文档加载器.ipynb                 # TXT/CSV/JSON/PDF/Word/Markdown/HTML/目录加载
│   ├── 02-文档切分器.ipynb                 # 字符/递归/Token/语义/HTML/代码/Markdown 切分
│   ├── 03-文档嵌入模型.ipynb               # 模型初始化、单句/批量文本/CSV 向量化
│   ├── 04-Milvus的基本使用.ipynb           # 数据库/集合管理、向量写入与检索
│   └── 05-案例：Atguigu Assistant客服知识库.ipynb # 客服知识库 RAG 问答
├── knowledge.txt                    # 客服知识库案例使用的文本素材
├── releases/                        # 版本发布说明
│   ├── v0.7.0.md
│   └── v0.8.0.md
├── asset/                            # Notebook 使用的示例素材
│   └── load/                         # 文本、结构化数据及文档样例
├── todo_workspace/                  # 待办工具示例与测试
│   ├── my_add.py
│   └── test_my_add.py
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

> 第 9 章 PostgreSQL 长期记忆示例需要额外配置 `POSTGRES_DB_URL`；仅在运行对应示例时使用。

### 第 10 章 RAG 配置

建议依次学习文档加载器、文档切分器、文档嵌入模型、Milvus 基础和客服知识库案例。Notebook 中的素材路径相对于 `chapter10-RAG/`，请将内核工作目录设为该目录。

`requirements.txt` 中部分 RAG 依赖已注释；需要运行文档解析、Token 切分等示例时，可安装完整依赖：

```bash
pip install -r requirements_full.txt
```

也可按示例安装所需依赖：Token 切分需要 `tiktoken`，PDF 加载需要 `pypdf`，Word / Markdown / HTML 的 Unstructured 加载器需要 `unstructured` 及对应格式依赖。文档加载器当前保存的 Word 示例输出包含缺少 `unstructured` 的报错，安装相关依赖后重新运行该单元格。

将下列变量按需添加到本地 `.env`，填入自己的凭据及服务地址：

```dotenv
CLOSEAI_API_KEY=your_closeai_api_key
CLOSEAI_BASE_URL=your_closeai_openai_compatible_base_url
SILICONFLOW_API_KEY=your_siliconflow_api_key
SILICONFLOW_BASE_URL=your_siliconflow_openai_compatible_base_url
MINERU_API_TOKEN=your_mineru_api_token
```

语义切分示例默认使用 CLOSEAI 的 `text-embedding-3-large`；嵌入模型教程还演示硅基流动的 `Pro/BAAI/bge-m3`，通过 `init_embeddings()` 或 `OpenAIEmbeddings` 初始化，再调用 `embed_query()` / `embed_documents()`。这些在线示例需要有效凭据及网络连接，并可能产生服务费用。`MINERU_API_TOKEN` 仅用于文档加载器中的 MinerU 在线解析示例。

### Milvus 与客服知识库案例

- 先启动支持数据库操作的 Milvus 服务；两个示例默认连接 `http://localhost:19530`，请按实际部署修改连接地址。
- `requirements_full.txt` 已包含 `pymilvus` 与 `langchain-milvus`；本次两个示例直接使用 `pymilvus.MilvusClient`。
- 嵌入模型使用硅基流动的 `Pro/BAAI/bge-m3`，集合配置为 1024 维、`COSINE` 相似度，需要配置 `SILICONFLOW_API_KEY` / `SILICONFLOW_BASE_URL`。
- 客服案例读取根目录 `knowledge.txt`，写入 `rag_tutorial` 数据库的 `docs` 集合；回答生成使用 `ChatDeepSeek`，需要配置 `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL`，并按账号支持情况调整 Notebook 中的模型名称。
- Milvus 基础教程包含删除数据库和集合的独立演示，运行删除数据库单元格后，应重新创建数据库再继续后续示例。客服案例初始化时会删除并重建同名集合，请使用专用的练习数据库和集合。

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
| **chapter08-Middleware** | 中间件 | `SummarizationMiddleware` 对话摘要、`HumanInTheLoopMiddleware` 人工审批、`PIIMiddleware` 个人信息脱敏、`TodoListMiddleware` 待办管理、其它内置中间件，以及 Node-style / Wrap-style 自定义中间件与 hook 执行顺序 |
| **chapter09-memory** | 记忆 | Agent 记忆测试、短期记忆、记忆治理策略（消息删除/摘要）、长期记忆（基础 API 与 Agent 集成，含 PostgreSQL 配置示例） |
| **chapter10-RAG** | RAG 与 Milvus 知识库实战 | 多格式文档加载与切分；嵌入模型与文档向量化；Milvus 数据库/集合管理、向量写入与检索；客服知识库 Agent 问答 |
| **todo_workspace** | 测试示例 | 简单工具函数及其测试代码 |

## 许可证

[MIT](LICENSE) © 2026 崔学成
