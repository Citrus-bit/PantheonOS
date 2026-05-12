# PantheonOS 代码库导览

## 项目总览

PantheonOS 是一个面向多智能体协作、可演化工作流和分布式工具调用的 Python 框架；这份文档关注“仓库怎么读”，不是替代现有 [`docs/source/architecture.rst`](/Users/tampouseng/Desktop/PantheonOS-main/docs/source/architecture.rst) 的系统级架构说明。

如果你第一次进入这个仓库，建议按下面的顺序读：

1. [`README.md`](/Users/tampouseng/Desktop/PantheonOS-main/README.md)：先看产品定位、安装方式和主要使用入口。
2. [`pantheon/agent.py`](/Users/tampouseng/Desktop/PantheonOS-main/pantheon/agent.py)：理解单个 Agent 的运行模型、消息流和模型调用。
3. [`pantheon/team/pantheon.py`](/Users/tampouseng/Desktop/PantheonOS-main/pantheon/team/pantheon.py)：理解多 Agent 编排、委派和插件装配。
4. [`pantheon/toolset.py`](/Users/tampouseng/Desktop/PantheonOS-main/pantheon/toolset.py)：理解工具如何被声明、注入上下文并暴露给 Agent。
5. [`pantheon/chatroom/start.py`](/Users/tampouseng/Desktop/PantheonOS-main/pantheon/chatroom/start.py) 与 [`pantheon/endpoint/core.py`](/Users/tampouseng/Desktop/PantheonOS-main/pantheon/endpoint/core.py)：理解本地 UI/服务启动和远程工具端点。

## 阅读约定

- 树里的每一行都采用 `path  # 简短说明`。
- 目录行说明“这一组文件解决什么问题”，文件行说明“职责 / 入口 / 被谁调用”。
- 同级节点按字母序排列，便于你直接在编辑器里对照跳转。
- `pantheon/` 是主体代码区，原则上展开到 tracked 文件；`pantheon/factory/templates/` 只保留关键层级，不把所有模板全文铺开。
- `tests/`、`examples/`、`docs/` 只做目录级摘要，避免把这份导览写成近千行的文件清单。
- 明确排除：`.git/`、`.venv/`、`.pantheon/`、`.executor/`、`jetstream_storage/`、`pantheon_agents.egg-info/` 以及其他本地缓存和运行产物。

## 核心仓库树

```text
./                                                                            # 仓库根目录；这里只展示 tracked 的核心结构与主要入口
├── .agents/                                                                  # 给协作代理的背景资料、仓库约定和补充说明
│   ├── .agents/README.md                                                     # 代理协作说明的入口文档
│   ├── .agents/conventions.md                                                # 面向代理的编码与协作约定
│   ├── .agents/memory-learning-systems.md                                    # 记忆与学习系统的背景说明
│   └── .agents/overview.md                                                   # 仓库全局概览与高层脉络
├── .claude/                                                                  # Claude/Codex 一类代理的本地行为配置
│   └── .claude/settings.local.json                                           # 本地代理偏好与工具行为设置
├── .dockerignore                                                             # 仓库根目录的 Docker 构建忽略规则
├── .env.test                                                                 # 测试环境变量样例
├── .github/                                                                  # CI/CD 工作流定义
│   └── .github/workflows/                                                    # GitHub Actions 工作流目录
│       ├── .github/workflows/docker-build.yml                                # Docker 镜像构建流程
│       ├── .github/workflows/publish_pypi.yml                                # PyPI 发布流程
│       └── .github/workflows/test.yml                                        # 主测试流水线
├── .gitignore                                                                # Git 忽略规则
├── .python-version                                                           # 本地 Python 版本提示
├── .readthedocs.yaml                                                         # Read the Docs 构建配置
├── CLAUDE.md                                                                 # 面向 Claude 类代理的仓库级说明
├── LICENSE                                                                   # 开源许可证文本
├── MANIFEST.in                                                               # Python 打包时的附带文件清单
├── README.md                                                                 # 项目总入口：定位、安装、使用方式与外部链接
├── build_backend.spec                                                        # 桌面/后端打包规格文件
├── docker/                                                                   # 容器化运行与双模式启动支持
│   ├── docker/.dockerignore                                                  # Docker 子目录的额外忽略规则
│   ├── docker/Dockerfile                                                     # Pantheon 容器镜像定义
│   ├── docker/README.md                                                      # Docker 使用与部署说明
│   ├── docker/docker-compose.yml                                             # 本地容器编排示例
│   ├── docker/docker-entrypoint-dual-mode.sh                                 # 同时拉起服务组件的容器入口脚本
│   └── docker/nats-ws.conf                                                   # 容器内 NATS WebSocket 配置
├── nats-ws.conf                                                              # 仓库根目录下的 NATS WebSocket 配置样例
├── pantheon/                                                                 # 核心 Python 包：Agent、Team、ToolSet、Chatroom、Endpoint 都在这里
│   ├── pantheon/__init__.py                                                  # 包版本与顶层导出
│   ├── pantheon/__main__.py                                                  # 统一 CLI 入口，分发到不同子命令
│   ├── pantheon/agent.py                                                     # Agent 核心实现：消息、模型、工具调用与上下文管理
│   ├── pantheon/background.py                                                # 后台任务支持，处理长耗时工具调用的异步反馈
│   ├── pantheon/chatroom/                                                    # 多智能体会话服务层，对接 UI、会话线程与流式输出
│   │   ├── pantheon/chatroom/__init__.py                                     # chatroom 包命名空间
│   │   ├── pantheon/chatroom/__main__.py                                     # chatroom 子命令与 OAuth 相关入口
│   │   ├── pantheon/chatroom/export.py                                       # 聊天记录导出与导入工具
│   │   ├── pantheon/chatroom/nats-ws.conf                                    # chatroom 内置使用的 NATS WebSocket 配置
│   │   ├── pantheon/chatroom/nats_manager.py                                 # 本地 NATS 服务的自动启动与管理
│   │   ├── pantheon/chatroom/projects.py                                     # 项目目录注册表与项目切换逻辑
│   │   ├── pantheon/chatroom/room.py                                         # ChatRoom 主服务，实现会话编排和前后端桥接
│   │   ├── pantheon/chatroom/special_agents.py                               # 摘要、建议问题、会话命名等特殊用途 Agent
│   │   ├── pantheon/chatroom/start.py                                        # 本地启动 chatroom、endpoint 与浏览器入口的总控脚本
│   │   ├── pantheon/chatroom/stream.py                                       # NATS 流式消息适配器
│   │   └── pantheon/chatroom/thread.py                                       # 单条会话线程的数据模型与状态容器
│   ├── pantheon/claw/                                                        # PantheonClaw 多渠道网关，把 chatroom 暴露到聊天平台
│   │   ├── pantheon/claw/__init__.py                                         # claw 包命名空间
│   │   ├── pantheon/claw/__main__.py                                         # claw 命令行入口，负责网关模式切换
│   │   ├── pantheon/claw/bridge.py                                           # chatroom 与外部聊天渠道之间的桥接层
│   │   ├── pantheon/claw/channels/                                           # 各聊天平台的适配实现
│   │   │   ├── pantheon/claw/channels/__init__.py                            # 渠道模块的延迟导入入口
│   │   │   ├── pantheon/claw/channels/discord.py                             # Discord 渠道适配器
│   │   │   ├── pantheon/claw/channels/feishu.py                              # 飞书渠道适配器
│   │   │   ├── pantheon/claw/channels/imessage.py                            # iMessage 渠道适配器
│   │   │   ├── pantheon/claw/channels/qq.py                                  # QQ 渠道适配器
│   │   │   ├── pantheon/claw/channels/slack.py                               # Slack 渠道适配器
│   │   │   ├── pantheon/claw/channels/telegram.py                            # Telegram 渠道适配器
│   │   │   └── pantheon/claw/channels/wechat.py                              # 微信渠道适配器
│   │   ├── pantheon/claw/config.py                                           # claw 配置的默认值、读写与脱敏逻辑
│   │   ├── pantheon/claw/manager.py                                          # 多渠道网关的统一生命周期管理
│   │   ├── pantheon/claw/registry.py                                         # 渠道路由注册表，记录会话与目标平台的映射
│   │   └── pantheon/claw/runtime.py                                          # 渠道共享运行时，处理去重、格式转换和进度文本
│   ├── pantheon/constant.py                                                  # 跨模块共享常量
│   ├── pantheon/endpoint/                                                    # 远程工具端点：把 ToolSet 与 MCP 服务暴露成后端能力
│   │   ├── pantheon/endpoint/__init__.py                                     # endpoint 包命名空间
│   │   ├── pantheon/endpoint/__main__.py                                     # endpoint 命令行入口与配置生成器
│   │   ├── pantheon/endpoint/core.py                                         # Endpoint 主类，装配工具、文件传输与 MCP 管理
│   │   ├── pantheon/endpoint/gateway.py                                      # 统一 MCP 网关，把多个 MCP server 聚合成单入口
│   │   ├── pantheon/endpoint/hub.py                                          # 多 endpoint 实例的聚合与发现层
│   │   ├── pantheon/endpoint/mcp.py                                          # MCP 进程池与 server 生命周期管理
│   │   ├── pantheon/endpoint/toolset_proxy.py                                # 远程 ToolSet 代理，供 Agent 透明调用
│   │   └── pantheon/endpoint/toolsets.py                                     # ToolSet 生命周期管理器
│   ├── pantheon/evolution/                                                   # 代码演化框架，用多 Agent 和评估器迭代改进程序
│   │   ├── pantheon/evolution/__init__.py                                    # evolution 包命名空间
│   │   ├── pantheon/evolution/__main__.py                                    # evolution CLI 入口
│   │   ├── pantheon/evolution/config.py                                      # 演化流程的配置对象与预设策略
│   │   ├── pantheon/evolution/database.py                                    # 演化结果数据库与 MAP-Elites 存储
│   │   ├── pantheon/evolution/evaluator.py                                   # 候选程序的混合评估系统
│   │   ├── pantheon/evolution/program.py                                     # 演化中的程序与代码快照数据结构
│   │   ├── pantheon/evolution/prompt_builder.py                              # 生成变异提示词与评估上下文
│   │   ├── pantheon/evolution/result.py                                      # 演化过程结果对象
│   │   ├── pantheon/evolution/team.py                                        # 协调多 Agent 执行代码演化的 Team
│   │   ├── pantheon/evolution/utils/                                         # 演化流程的底层辅助函数
│   │   │   ├── pantheon/evolution/utils/__init__.py                          # utils 子包命名空间
│   │   │   ├── pantheon/evolution/utils/diff.py                              # diff 解析、补丁应用与搜索替换工具
│   │   │   └── pantheon/evolution/utils/metrics.py                           # 代码复杂度、多样性等度量计算
│   │   └── pantheon/evolution/visualizer.py                                  # 演化结果可视化与 HTML 报告生成
│   ├── pantheon/factory/                                                     # 模板工厂：把 markdown/json 模板装配成 Agent、Team 与技能包
│   │   ├── pantheon/factory/__init__.py                                      # 暴露按模板创建 Agent/Team 的便捷函数
│   │   ├── pantheon/factory/models.py                                        # 模板配置的数据模型
│   │   ├── pantheon/factory/template_io.py                                   # 模板解析、prompt 解析与文件型模板管理
│   │   ├── pantheon/factory/template_manager.py                              # 模板发现、加载与缓存入口
│   │   └── pantheon/factory/templates/                                       # 预置模板仓库；这里只保留关键层级，不展开所有模板文件
│   │       ├── pantheon/factory/templates/.env.example                       # 模板环境变量样例
│   │       ├── pantheon/factory/templates/agents/                            # 预置 Agent 模板目录，覆盖 graph maker、single-cell、rare-disease 等场景
│   │       │   ├── pantheon/factory/templates/agents/graph_maker/            # 生成图表团队的角色模板目录
│   │       │   ├── pantheon/factory/templates/agents/paper_write/            # 论文写作团队的角色模板目录
│   │       │   ├── pantheon/factory/templates/agents/rare_disease/           # 罕见病分析团队的角色模板目录
│   │       │   └── pantheon/factory/templates/agents/single_cell/            # 单细胞分析团队的角色模板目录
│   │       ├── pantheon/factory/templates/mcp.json                           # 默认 MCP 模板配置
│   │       ├── pantheon/factory/templates/prompts/                           # 通用 prompt 模板目录，供 agent/team 组装系统提示词
│   │       ├── pantheon/factory/templates/settings.json                      # 工厂默认设置与模板注册表
│   │       ├── pantheon/factory/templates/skills/                            # 预置技能模板目录
│   │       │   ├── pantheon/factory/templates/skills/SKILLS.md               # 技能目录索引
│   │       │   ├── pantheon/factory/templates/skills/bio_image_processing/   # 生物图像处理技能族
│   │       │   ├── pantheon/factory/templates/skills/figure_styling/         # 论文图风格化技能族
│   │       │   ├── pantheon/factory/templates/skills/omics/                  # omics / single-cell / spatial 技能族
│   │       │   ├── pantheon/factory/templates/skills/paper_writing/          # 论文写作技能族
│   │       │   ├── pantheon/factory/templates/skills/presentation/           # 幻灯片与演示文稿技能族
│   │       │   └── pantheon/factory/templates/skills/rare_disease/           # 罕见病知识与本体技能族
│   │       └── pantheon/factory/templates/teams/                             # 预置 Team 模板目录，包含 default、evolution、single_cell 等组合
│   ├── pantheon/internal/                                                    # 内部插件与共享运行时：压缩、记忆、学习、任务系统都挂在这里
│   │   ├── pantheon/internal/__init__.py                                     # internal 包命名空间
│   │   ├── pantheon/internal/background_agent.py                             # 轻量后台 Agent 工厂，用于多轮后台推理
│   │   ├── pantheon/internal/compression/                                    # 上下文压缩子系统
│   │   │   ├── pantheon/internal/compression/__init__.py                     # compression 子包命名空间
│   │   │   ├── pantheon/internal/compression/compressor.py                   # 对话历史压缩器与状态模型
│   │   │   ├── pantheon/internal/compression/plugin.py                       # 把压缩器接入 PantheonTeam 的插件
│   │   │   └── pantheon/internal/compression/prompts.py                      # 压缩相关 prompt 模板
│   │   ├── pantheon/internal/learning_system/                                # 技能学习系统，从对话中提炼技能并注入 prompt
│   │   │   ├── pantheon/internal/learning_system/__init__.py                 # learning_system 子包命名空间
│   │   │   ├── pantheon/internal/learning_system/config.py                   # 学习系统配置解析
│   │   │   ├── pantheon/internal/learning_system/extractor.py                # 从对话中抽取技能的后台提取器
│   │   │   ├── pantheon/internal/learning_system/injector.py                 # 把技能索引注入系统提示词
│   │   │   ├── pantheon/internal/learning_system/plugin.py                   # 学习系统的 Team 插件适配层
│   │   │   ├── pantheon/internal/learning_system/prompts.py                  # 学习系统 prompt 模板
│   │   │   ├── pantheon/internal/learning_system/runtime.py                  # 学习系统共享运行时
│   │   │   ├── pantheon/internal/learning_system/store.py                    # 技能文件的持久化与原子写入
│   │   │   ├── pantheon/internal/learning_system/toolset.py                  # 面向 Agent 的技能管理 ToolSet
│   │   │   └── pantheon/internal/learning_system/types.py                    # 技能 frontmatter 与校验类型定义
│   │   ├── pantheon/internal/memory/                                         # 旧版/基础内存抽象与存储后端
│   │   │   ├── pantheon/internal/memory/__init__.py                          # memory 子包命名空间
│   │   │   ├── pantheon/internal/memory/memory.py                            # Memory 与 MemoryManager 实现
│   │   │   └── pantheon/internal/memory/storage.py                           # JSON / JSONL 存储后端
│   │   ├── pantheon/internal/memory_system/                                  # 统一长期记忆系统
│   │   │   ├── pantheon/internal/memory_system/__init__.py                   # memory_system 子包命名空间
│   │   │   ├── pantheon/internal/memory_system/chatroom.py                   # ChatRoom 与共享记忆运行时的适配层
│   │   │   ├── pantheon/internal/memory_system/config.py                     # 长期记忆系统配置解析
│   │   │   ├── pantheon/internal/memory_system/dream.py                      # dream/consolidation 记忆整合逻辑
│   │   │   ├── pantheon/internal/memory_system/extract_memories.py           # 每轮自动抽取长期记忆
│   │   │   ├── pantheon/internal/memory_system/flush.py                      # 压缩前刷新重要信息到长期记忆
│   │   │   ├── pantheon/internal/memory_system/freshness.py                  # 记忆新鲜度与过期提示
│   │   │   ├── pantheon/internal/memory_system/plugin.py                     # 记忆系统接入 PantheonTeam 的插件
│   │   │   ├── pantheon/internal/memory_system/prompts.py                    # 记忆系统 prompt 模板
│   │   │   ├── pantheon/internal/memory_system/retrieval.py                  # 基于 LLM 的记忆检索器
│   │   │   ├── pantheon/internal/memory_system/runtime.py                    # 统一记忆运行时的核心协调器
│   │   │   ├── pantheon/internal/memory_system/session_log.py                # 会话日志管理
│   │   │   ├── pantheon/internal/memory_system/session_note.py               # 持续更新的会话摘要/速记抽取器
│   │   │   ├── pantheon/internal/memory_system/store.py                      # 基于文件的长期记忆存储
│   │   │   └── pantheon/internal/memory_system/types.py                      # 记忆 frontmatter、类型和序列化约定
│   │   ├── pantheon/internal/message/                                        # 消息附件检测与处理流水线
│   │   │   ├── pantheon/internal/message/attachment_detection.py             # 自动识别图片、路径、链接等附件
│   │   │   └── pantheon/internal/message/attachment_pipeline.py              # 附件处理总流水线与消息预处理器
│   │   ├── pantheon/internal/package_runtime/                                # 运行时 package 上下文导出、发现和代理
│   │   │   ├── pantheon/internal/package_runtime/__init__.py                 # package runtime 包入口与管理器装配
│   │   │   ├── pantheon/internal/package_runtime/context.py                  # 导出/加载包上下文到子进程或解释器
│   │   │   ├── pantheon/internal/package_runtime/manager.py                  # 运行时 package 发现与调用管理
│   │   │   └── pantheon/internal/package_runtime/runtime.py                  # 在解释器里暴露给用户代码的 package runtime
│   │   ├── pantheon/internal/system_prompt.py                                # 系统提示词渲染与上下文块拼装
│   │   ├── pantheon/internal/task_system/                                    # 任务状态机与任务工具接入层
│   │   │   ├── pantheon/internal/task_system/__init__.py                     # task_system 子包入口
│   │   │   └── pantheon/internal/task_system/plugin.py                       # 把任务系统挂入 PantheonTeam 的插件
│   │   └── pantheon/internal/think_plugin.py                                 # leader-only think 工具与相关 prompt 注入
│   ├── pantheon/packages.py                                                  # 面向用户的 package runtime 兼容入口
│   ├── pantheon/providers.py                                                 # Tool provider 抽象，连接本地、MCP 与 ToolSet provider
│   ├── pantheon/remote/                                                      # 分布式远程后端抽象；当前以 NATS 为主
│   │   ├── pantheon/remote/__init__.py                                       # remote 包命名空间
│   │   ├── pantheon/remote/backend/                                          # 远程通信后端实现
│   │   │   ├── pantheon/remote/backend/__init__.py                           # backend 子包命名空间
│   │   │   ├── pantheon/remote/backend/base.py                               # 远程 backend、service、worker 的基础抽象
│   │   │   ├── pantheon/remote/backend/nats.py                               # NATS 后端实现与远程 worker 协议
│   │   │   └── pantheon/remote/backend/registry.py                           # 可用 backend 的注册表
│   │   ├── pantheon/remote/factory.py                                        # 远程 backend 配置解析与连接工厂
│   │   └── pantheon/remote/remote.py                                         # 远程连接的轻量便捷入口
│   ├── pantheon/repl/                                                        # 交互式命令行界面与文本 UI
│   │   ├── pantheon/repl/__init__.py                                         # repl 包命名空间
│   │   ├── pantheon/repl/__main__.py                                         # REPL 命令行启动入口
│   │   ├── pantheon/repl/conversationRecovery.py                             # 会话恢复与中断检测逻辑
│   │   ├── pantheon/repl/core.py                                             # REPL 主循环，连接 Agent/Team 与文本交互
│   │   ├── pantheon/repl/handlers/                                           # REPL 命令处理器
│   │   │   ├── pantheon/repl/handlers/__init__.py                            # handlers 命名空间
│   │   │   ├── pantheon/repl/handlers/base.py                                # 命令处理器基类
│   │   │   ├── pantheon/repl/handlers/builtin/                               # 内置斜杠命令处理器
│   │   │   │   ├── pantheon/repl/handlers/builtin/__init__.py                # builtin handlers 命名空间
│   │   │   │   ├── pantheon/repl/handlers/builtin/bash.py                    # `/bash` 命令处理器
│   │   │   │   ├── pantheon/repl/handlers/builtin/edit.py                    # `/edit` 命令处理器，调用外部编辑器
│   │   │   │   ├── pantheon/repl/handlers/builtin/mcp.py                     # MCP 管理命令处理器
│   │   │   │   ├── pantheon/repl/handlers/builtin/revert.py                  # 回滚/恢复类命令处理器
│   │   │   │   └── pantheon/repl/handlers/builtin/view.py                    # 文件查看命令处理器
│   │   │   └── pantheon/repl/handlers/template_handler.py                    # 基于模板的通用命令处理器
│   │   ├── pantheon/repl/prompt_app.py                                       # prompt_toolkit 驱动的输入框、补全和按键绑定
│   │   ├── pantheon/repl/renderers.py                                        # 工具调用与工具结果的终端渲染器
│   │   ├── pantheon/repl/sessionRestore.py                                   # 从日志和 worktree 恢复旧会话
│   │   ├── pantheon/repl/sessionStorage.py                                   # REPL 会话元数据与恢复状态持久化
│   │   ├── pantheon/repl/setup_wizard.py                                     # 首次使用时的 provider/API key 配置向导
│   │   ├── pantheon/repl/task_renderers.py                                   # 任务型 UI 的渲染器
│   │   ├── pantheon/repl/ui.py                                               # REPL 文本界面外观与输出封装
│   │   ├── pantheon/repl/user_response.py                                    # 面向用户的统一响应格式化
│   │   ├── pantheon/repl/utils.py                                            # REPL 共享辅助函数与样式片段
│   │   └── pantheon/repl/viewers/                                            # 全屏查看器与交互对话框
│   │       ├── pantheon/repl/viewers/__init__.py                             # viewers 命名空间
│   │       ├── pantheon/repl/viewers/file_viewer_ptk.py                      # 基于 prompt_toolkit 的文件查看器
│   │       ├── pantheon/repl/viewers/notify_dialog.py                        # 需要用户确认时的交互对话框
│   │       └── pantheon/repl/viewers/unified_dialog.py                       # 统一的文件审阅/提问对话框
│   ├── pantheon/settings.py                                                  # 全局配置加载、JSONC 解析与 Settings 单例
│   ├── pantheon/slack/                                                       # Slack 集成入口
│   │   ├── pantheon/slack/__init__.py                                        # slack 包命名空间
│   │   ├── pantheon/slack/__main__.py                                        # Slack 子命令入口
│   │   └── pantheon/slack/app.py                                             # Slack 应用启动器
│   ├── pantheon/smart_func.py                                                # 把普通函数包装成更易被 Agent 调用的 smart_func
│   ├── pantheon/store/                                                       # Pantheon Store 的 CLI、安装、发布与批量播种
│   │   ├── pantheon/store/__init__.py                                        # store 包命名空间
│   │   ├── pantheon/store/auth.py                                            # Store JWT 鉴权与本地 token 存储
│   │   ├── pantheon/store/cli.py                                             # Store 命令行子命令集合
│   │   ├── pantheon/store/client.py                                          # Store HTTP API 客户端
│   │   ├── pantheon/store/installer.py                                       # 从 Store 下载并安装 package
│   │   ├── pantheon/store/publisher.py                                       # 收集并发布 agent/team/skill 到 Store
│   │   └── pantheon/store/seed.py                                            # 批量向 Store 预灌模板与外部技能
│   ├── pantheon/team/                                                        # 多 Agent 编排模式：顺序、群聊、MoA、Agent-as-Tool 等
│   │   ├── pantheon/team/__init__.py                                         # team 包导出
│   │   ├── pantheon/team/aat.py                                              # Agent-as-Tool 团队模式
│   │   ├── pantheon/team/base.py                                             # Team 抽象基类
│   │   ├── pantheon/team/moa.py                                              # Mixture-of-Agents 团队模式
│   │   ├── pantheon/team/pantheon.py                                         # PantheonTeam 主实现，负责委派、插件与上下文继承
│   │   ├── pantheon/team/plugin.py                                           # Team 插件协议与紧凑提示接口
│   │   ├── pantheon/team/plugin_registry.py                                  # 插件注册与装配中心
│   │   ├── pantheon/team/sequential.py                                       # 顺序执行型团队模式
│   │   └── pantheon/team/swarm.py                                            # Swarm 与 SwarmCenter 团队模式
│   ├── pantheon/toolset.py                                                   # ToolSet 基础设施：`@tool` 装饰器、上下文注入与执行编排
│   ├── pantheon/toolsets/                                                    # 内置工具箱集合
│   │   ├── pantheon/toolsets/__init__.py                                     # ToolSet 模块的懒加载入口
│   │   ├── pantheon/toolsets/__main__.py                                     # 列出、检查和启动 ToolSet 的 CLI 入口
│   │   ├── pantheon/toolsets/code/                                           # 静态代码浏览与语法树分析工具
│   │   │   ├── pantheon/toolsets/code/__init__.py                            # code toolset 包入口
│   │   │   ├── pantheon/toolsets/code/code_toolset.py                        # 代码导航 ToolSet
│   │   │   └── pantheon/toolsets/code/tree_sitter_parser.py                  # 基于 tree-sitter 的多语言代码解析器
│   │   ├── pantheon/toolsets/database_api/                                   # 生物数据库查询 ToolSet
│   │   │   ├── pantheon/toolsets/database_api/README.md                      # 数据库查询工具说明
│   │   │   ├── pantheon/toolsets/database_api/__init__.py                    # database_api 包入口
│   │   │   ├── pantheon/toolsets/database_api/database_api_query.py          # 数据库查询 ToolSet 主实现
│   │   │   ├── pantheon/toolsets/database_api/schema_manager.py              # 数据库 schema 加载与管理
│   │   │   └── pantheon/toolsets/database_api/schemas/                       # 各数据库 API schema 定义
│   │   │       ├── pantheon/toolsets/database_api/schemas/cbioportal.json    # cBioPortal 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/clinvar.json       # ClinVar 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/dbsnp.json         # dbSNP 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/emdb.json          # EMDB 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/ensembl.json       # Ensembl 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/geo.json           # GEO 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/gnomad.json        # gnomAD 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/gtopdb.json        # Guide to Pharmacology 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/gwas_catalog.json  # GWAS Catalog 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/interpro.json      # InterPro 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/iucn.json          # IUCN 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/jaspar.json        # JASPAR 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/kegg.json          # KEGG 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/monarch.json       # Monarch Initiative 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/mpd.json           # Mouse Phenome Database 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/openfda.json       # OpenFDA 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/opentarget.json    # Open Targets 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/paleobiology.json  # Paleobiology Database 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/pdb.json           # Protein Data Bank 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/pride.json         # PRIDE 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/reactome.json      # Reactome 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/remap.json         # ReMap 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/stringdb.json      # STRING DB 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/ucsc.json          # UCSC Genome Browser 查询 schema
│   │   │       ├── pantheon/toolsets/database_api/schemas/uniprot.json       # UniProt 查询 schema
│   │   │       └── pantheon/toolsets/database_api/schemas/worms.json         # WoRMS 查询 schema
│   │   ├── pantheon/toolsets/evolution/                                      # 把演化能力暴露给 Agent 的工具箱
│   │   │   ├── pantheon/toolsets/evolution/__init__.py                       # evolution toolset 包入口
│   │   │   ├── pantheon/toolsets/evolution/evaluator_toolset.py              # 代码评估工具
│   │   │   └── pantheon/toolsets/evolution/evolution_toolset.py              # 演化会话与控制工具
│   │   ├── pantheon/toolsets/file/                                           # 文件编辑与搜索工具
│   │   │   ├── pantheon/toolsets/file/__init__.py                            # file toolset 包入口
│   │   │   ├── pantheon/toolsets/file/apply_patch.py                         # patch 解析与应用工具
│   │   │   ├── pantheon/toolsets/file/file_manager.py                        # 读写、替换、图像识别等文件管理工具
│   │   │   └── pantheon/toolsets/file/grep_glob.py                           # glob / grep 搜索实现
│   │   ├── pantheon/toolsets/file_transfer/                                  # 文件传输服务
│   │   │   ├── pantheon/toolsets/file_transfer/__init__.py                   # file_transfer 包入口
│   │   │   ├── pantheon/toolsets/file_transfer/client.py                     # 文件传输客户端
│   │   │   └── pantheon/toolsets/file_transfer/worker.py                     # 文件传输 ToolSet 与服务端逻辑
│   │   ├── pantheon/toolsets/image/                                          # 图像生成工具
│   │   │   ├── pantheon/toolsets/image/__init__.py                           # image toolset 包入口
│   │   │   └── pantheon/toolsets/image/image_gen.py                          # 图像生成 ToolSet
│   │   ├── pantheon/toolsets/julia/                                          # Julia 解释器工具
│   │   │   ├── pantheon/toolsets/julia/__init__.py                           # julia toolset 包入口
│   │   │   ├── pantheon/toolsets/julia/_julia.py                             # Julia 子进程解释器实现
│   │   │   └── pantheon/toolsets/julia/julia_interpreter.py                  # JuliaInterpreter ToolSet
│   │   ├── pantheon/toolsets/knowledge/                                      # 向量知识库与 RAG 配置
│   │   │   ├── pantheon/toolsets/knowledge/__init__.py                       # knowledge toolset 包入口
│   │   │   ├── pantheon/toolsets/knowledge/config.py                         # 知识库配置解析
│   │   │   ├── pantheon/toolsets/knowledge/config.yaml                       # 知识库默认配置文件
│   │   │   ├── pantheon/toolsets/knowledge/knowledge_manager.py              # 知识库管理 ToolSet
│   │   │   ├── pantheon/toolsets/knowledge/models.py                         # 知识库数据模型
│   │   │   └── pantheon/toolsets/knowledge/vector_store.py                   # 向量存储后端抽象
│   │   ├── pantheon/toolsets/notebook/                                       # Jupyter/Notebook 集成工具
│   │   │   ├── pantheon/toolsets/notebook/__init__.py                        # notebook toolset 包入口
│   │   │   ├── pantheon/toolsets/notebook/handlers.py                        # Jupyter IOPub 事件处理器
│   │   │   ├── pantheon/toolsets/notebook/integrated_notebook.py             # 集成式 Notebook ToolSet
│   │   │   ├── pantheon/toolsets/notebook/jedi_integration.py                # Jedi 代码补全与智能提示集成
│   │   │   ├── pantheon/toolsets/notebook/jupyter_kernel.py                  # Jupyter kernel 控制 ToolSet
│   │   │   └── pantheon/toolsets/notebook/notebook_contents.py               # Notebook 文件内容管理 ToolSet
│   │   ├── pantheon/toolsets/package.py                                      # package 安装、发现与调用的统一 ToolSet
│   │   ├── pantheon/toolsets/python/                                         # Python 解释器工具
│   │   │   ├── pantheon/toolsets/python/__init__.py                          # python toolset 包入口
│   │   │   └── pantheon/toolsets/python/python_interpreter.py                # PythonInterpreter ToolSet
│   │   ├── pantheon/toolsets/r/                                              # R 解释器工具
│   │   │   ├── pantheon/toolsets/r/__init__.py                               # r toolset 包入口
│   │   │   ├── pantheon/toolsets/r/_rinter.py                                # R 子进程解释器实现
│   │   │   └── pantheon/toolsets/r/r_interpreter.py                          # RInterpreter ToolSet
│   │   ├── pantheon/toolsets/rag/                                            # 旧版/通用向量 RAG 工具链
│   │   │   ├── pantheon/toolsets/rag/__init__.py                             # rag toolset 包入口
│   │   │   ├── pantheon/toolsets/rag/__main__.py                             # rag 子命令入口
│   │   │   ├── pantheon/toolsets/rag/build.py                                # 构建向量库与抓取文档
│   │   │   ├── pantheon/toolsets/rag/text.py                                 # 文本切分辅助函数
│   │   │   ├── pantheon/toolsets/rag/toolset.py                              # VectorRAG ToolSet
│   │   │   ├── pantheon/toolsets/rag/vectordb.py                             # 向量数据库封装
│   │   │   └── pantheon/toolsets/rag/wrap.py                                 # RAG 调用包装器
│   │   ├── pantheon/toolsets/scfm/                                           # single-cell foundation model 路由与适配层
│   │   │   ├── pantheon/toolsets/scfm/__init__.py                            # scfm toolset 包入口
│   │   │   ├── pantheon/toolsets/scfm/_conda_runner.py                       # 在独立 conda 环境里执行 SCFM 适配器
│   │   │   ├── pantheon/toolsets/scfm/adapters/                              # 各单细胞基础模型的适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/__init__.py               # adapters 命名空间
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/aidocell.py               # AIDO.Cell 模型适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/atacformer.py             # Atacformer 模型适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/base.py                   # SCFM 适配器基类
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/cell2sentence.py          # Cell2Sentence 模型适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/cellfm.py                 # CellFM 模型适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/cellplm.py                # CellPLM 模型适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/chatcell.py               # CHATCELL 模型适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/genecompass.py            # GeneCompass 模型适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/geneformer.py             # Geneformer 模型适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/genept.py                 # GenePT 模型适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/langcell.py               # LangCell 模型适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/nicheformer.py            # Nicheformer 模型适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/pulsar.py                 # PULSAR 模型适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/scbert.py                 # scBERT 模型适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/sccello.py                # scCello 模型适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/scfoundation.py           # scFoundation / xTrimoGene 适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/scgpt.py                  # scGPT 模型适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/scmulan.py                # scMulan 模型适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/scplantllm.py             # scPlantLLM 模型适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/scprint.py                # scPRINT 模型适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/tabula.py                 # Tabula 模型适配器
│   │   │   │   ├── pantheon/toolsets/scfm/adapters/tgpt.py                   # tGPT 模型适配器
│   │   │   │   └── pantheon/toolsets/scfm/adapters/uce.py                    # UCE 模型适配器
│   │   │   ├── pantheon/toolsets/scfm/registry.py                            # SCFM 模型注册表与能力元数据
│   │   │   ├── pantheon/toolsets/scfm/router.py                              # 基于 LLM 的 SCFM 路由器
│   │   │   └── pantheon/toolsets/scfm/toolset.py                             # SCFM ToolSet 主入口
│   │   ├── pantheon/toolsets/scraper.py                                      # 通用网页抓取 ToolSet
│   │   ├── pantheon/toolsets/shell/                                          # Shell 命令执行工具
│   │   │   ├── pantheon/toolsets/shell/__init__.py                           # shell toolset 包入口
│   │   │   ├── pantheon/toolsets/shell/_shell.py                             # shell 子进程执行核心
│   │   │   └── pantheon/toolsets/shell/shell.py                              # ShellToolSet 封装
│   │   ├── pantheon/toolsets/task/                                           # 任务状态与工作流工具
│   │   │   ├── pantheon/toolsets/task/__init__.py                            # task toolset 包入口
│   │   │   ├── pantheon/toolsets/task/ephemeral.py                           # 生成临时状态消息
│   │   │   ├── pantheon/toolsets/task/task_state.py                          # 任务状态、角色与会话状态模型
│   │   │   └── pantheon/toolsets/task/task_toolset.py                        # TaskToolSet 主实现
│   │   ├── pantheon/toolsets/web.py                                          # 面向 Agent 的网页搜索/抓取 ToolSet
│   │   └── pantheon/toolsets/web_urllib.py                                   # 基于 DDGS/urllib 的轻量网页 ToolSet 备选实现
│   └── pantheon/utils/                                                       # 跨模块共享的 LLM、日志、模板、视觉与工具辅助函数
│       ├── pantheon/utils/__init__.py                                        # utils 包命名空间
│       ├── pantheon/utils/adapters/                                          # 各 LLM provider 的统一适配层
│       │   ├── pantheon/utils/adapters/__init__.py                           # adapter 工厂入口
│       │   ├── pantheon/utils/adapters/anthropic_adapter.py                  # Anthropic / Claude 适配器
│       │   ├── pantheon/utils/adapters/base.py                               # provider adapter 基类与统一异常
│       │   ├── pantheon/utils/adapters/codex_adapter.py                      # 基于 OAuth 的 Codex / ChatGPT backend 适配器
│       │   ├── pantheon/utils/adapters/gemini_adapter.py                     # Gemini REST API 适配器
│       │   ├── pantheon/utils/adapters/gemini_cli_adapter.py                 # Gemini CLI 风格 REST 适配器
│       │   ├── pantheon/utils/adapters/image_blocks.py                       # 多模态消息里的图片块转换与缩放
│       │   └── pantheon/utils/adapters/openai_adapter.py                     # OpenAI 及 OpenAI-compatible provider 适配器
│       ├── pantheon/utils/display.py                                         # 富文本终端输出与工具结果打印辅助
│       ├── pantheon/utils/image_detection.py                                 # 运行前后快照比对，识别新生成的图片文件
│       ├── pantheon/utils/llm.py                                             # LLM 调用封装与 OpenAI Responses/Chat Completions 兼容层
│       ├── pantheon/utils/llm_catalog.json                                   # 模型目录与 provider 元数据
│       ├── pantheon/utils/llm_providers.py                                   # provider 识别、Responses API 策略与基础配置工具
│       ├── pantheon/utils/log.py                                             # 日志配置与临时日志级别控制
│       ├── pantheon/utils/memory_compress.py                                 # 旧版记忆压缩辅助函数
│       ├── pantheon/utils/message_formatter.py                               # 会话消息转文本的统一格式化器
│       ├── pantheon/utils/misc.py                                            # 常用杂项工具：调用、端口、描述解析等
│       ├── pantheon/utils/model_discovery.py                                 # 动态探测 provider 可用模型列表
│       ├── pantheon/utils/model_selector.py                                  # 默认模型选择、provider 排序与 Ollama 状态缓存
│       ├── pantheon/utils/oauth/                                             # provider OAuth 登录支持
│       │   ├── pantheon/utils/oauth/__init__.py                              # OAuth 包入口
│       │   ├── pantheon/utils/oauth/codex.py                                 # OpenAI Codex / ChatGPT OAuth 流程
│       │   └── pantheon/utils/oauth/gemini.py                                # Gemini CLI OAuth 流程
│       ├── pantheon/utils/process.py                                         # 跨平台子进程生命周期辅助函数
│       ├── pantheon/utils/provider_registry.py                               # 从模型目录加载 provider 元数据并提供查询接口
│       ├── pantheon/utils/template.py                                        # 模板项解析与模板加载工具
│       ├── pantheon/utils/token_optimization.py                              # 对话上下文折叠与 token 优化逻辑
│       ├── pantheon/utils/tool_pairing.py                                    # 工具调用消息与结果消息配对修复
│       ├── pantheon/utils/truncate.py                                        # 大体积工具输出截断与预览辅助函数
│       ├── pantheon/utils/vision.py                                          # 多模态图片输入封装与引用展开
│       └── pantheon/utils/vision_capability.py                               # 检测模型是否支持视觉/图片结果能力
├── pyproject.toml                                                            # Python 包元数据、依赖、入口点和 pytest 配置
├── runtime_hook_tiktoken.py                                                  # 打包环境里为 tiktoken 注入运行时 hook
├── scripts/                                                                  # 维护、基准测试与批量导入脚本
│   ├── scripts/benchmark_multi_provider.py                                   # 多 provider 响应/成本基准脚本
│   ├── scripts/benchmark_prompt_cache.py                                     # prompt cache 基准脚本
│   ├── scripts/benchmark_token_optimization_live.py                          # 在线 token 优化基准脚本
│   ├── scripts/benchmark_token_optimizations.py                              # token 优化离线比较脚本
│   ├── scripts/seed_omicclaw.py                                              # 为 OmicClaw / Store 批量播种数据
│   └── scripts/test_two_phase_live.py                                        # 双阶段流程的在线测试脚本
└── uv.lock                                                                   # uv 解析出的锁文件，固定依赖版本
```

## 支撑目录摘要

- `tests/`：自动化测试总入口。这里重点覆盖 `agent/team` 编排、`chatroom` 与 NATS 流、`memory_system` 与 `learning_system`、`toolsets`、provider/OAuth/model selection、REPL 恢复与 UI 渲染等核心行为；其中 `tests/test_learning_system/`、`tests/test_memory_system/`、`tests/test_task_system/` 是按子系统分组的测试目录。
- `examples/`：示例与实验工作流。主要分成演化类示例（`evolution_batch_correction`、`evolution_gene_panel`、`evolution_topact`）、报告生成示例（`paper_reporter`、`paper_reporter_v2`）、上游流程/技能样例（`fastq_processing`）以及单细胞空间分析完整示例（`single_cell_spatial_analysis`）。
- `docs/`：用户文档与开发者文档。`docs/source/` 是 Sphinx 主文档树，`docs/source/architecture.rst` 讲系统级架构，`docs/scfm/` 收纳单细胞基础模型专题材料，`docs/images/` 存放文档图片资源，其余脚本和配置用于本地构建与文档测试。
