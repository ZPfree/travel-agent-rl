# 出行规划Agent RL训练系统

基于强化学习的出行规划Agent，支持最高50次工具调用，在RTX 4090单卡上完成从SFT冷启动到PPO/DPO训练的完整流程。

## 核心特性

- **Long Horizon Agent**：支持最高50步工具调用，含上下文压缩、死循环检测、步骤预算管理
- **8个模拟工具**：高德地图（POI/路线/地理编码）、Google/Tavily搜索、天气、日历
- **自定义PPO训练**：不依赖TRL封装，步级奖励 + GAE计算advantage
- **课程学习**：4阶段渐进训练（10→20→35→50步）
- **ReAct格式**：Thought/Action/Observation 结构化输出

## 代码结构

```
travel-agent-rl/
├── src/
│   ├── tools/                        # 工具层
│   │   ├── base.py                   # ToolResult（含置信度）、BaseTool、ToolRetryPolicy
│   │   ├── amap_tools.py             # 高德4个工具：POI搜索、周边搜索、路线规划、地理编码
│   │   ├── search_tools.py           # Google搜索、Tavily搜索
│   │   ├── weather_tools.py          # 天气查询
│   │   ├── calendar_tools.py         # 日历查询
│   │   └── registry.py               # ToolRegistry（注册/执行/重试）+ ToolCallValidator（参数校验）
│   │
│   ├── agent/                        # Agent层
│   │   ├── state.py                  # TravelState、Constraint、ToolCall、ProgressTracker
│   │   ├── prompt.py                 # ReAct格式prompt模板 + build_system_prompt/build_step_prompt
│   │   ├── parser.py                 # parse_react_output：解析Thought/Action/Final Answer
│   │   ├── context.py                # ContextCompressor：规则上下文压缩（20K token限制）
│   │   ├── loop_detector.py          # LoopDetector：死循环检测 + 打破策略
│   │   ├── graph.py                  # LangGraph状态图（7节点，含compress_context/finalize）
│   │   └── runner.py                 # AgentRunner：episode收集器，供PPO训练使用
│   │
│   ├── training/                     # 训练层
│   │   ├── config.py                 # TrainingConfig数据类
│   │   ├── reward.py                 # StepRewardModel（步级奖励）+ compute_gae（GAE）
│   │   ├── ppo_trainer.py            # AgentPPOTrainer：自定义PPO训练循环
│   │   ├── dpo_trainer.py            # DPOTrainer（max_length=32768）
│   │   └── curriculum.py             # CurriculumManager：4阶段课程学习
│   │
│   └── data/
│       └── mock_data.py              # 模拟数据：北京POI、路线、天气、日历、搜索结果
│
├── scripts/
│   ├── train.py                      # 训练入口（PPO循环 + 课程学习）
│   ├── generate_data.py              # 生成训练episode数据（JSONL格式）
│   └── evaluate.py                   # 评估入口（占位）
│
├── configs/
│   ├── training_config.yaml          # 训练超参：PPO/DPO/Agent/课程学习配置
│   └── tool_config.yaml              # 工具注册表配置
│
├── tests/                            # 196个测试，全部通过
│   ├── test_tools/                   # 工具层测试（基类、8个工具、注册表）
│   ├── test_agent/                   # Agent层测试（状态、解析器、压缩器、循环检测、图、Runner）
│   ├── test_training/                # 训练层测试（奖励、PPO、DPO、课程学习）
│   └── test_integration.py           # 端到端集成测试
│
└── docs/
    ├── design/                       # 设计文档（含审查记录）
    └── superpowers/plans/            # 实现计划
```

## 关键设计决策

### 为什么自定义PPO而不是用TRL？

TRL的PPO假设单轮文本生成（prompt→completion→reward），但本项目需要多步环境交互（model→tool call→env result→model→...）。自定义训练循环支持每步收集logprobs和reward，用GAE计算advantage。

### 上下文如何控制在20K tokens以内？

50步工具调用约35K tokens，通过以下策略压缩：
- 每步工具结果限制200 tokens
- ProgressTracker独立追踪关键信息，序列化注入prompt
- 规则压缩器保留最近5步完整，旧步骤压缩为摘要

### 如何防止死循环？

LoopDetector检测两种模式：连续3次相同工具+相同参数、窗口内重复调用过多。检测到循环后注入提示建议切换备用工具。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest tests/ -v

# 生成训练数据
python scripts/generate_data.py --num_episodes 100 --max_steps 10

# 开始训练
python scripts/train.py --config configs/training_config.yaml
```

## 技术栈

| 组件 | 选择 |
|------|------|
| 基础模型 | Qwen3-0.6B |
| Agent框架 | LangGraph >= 1.0 |
| RL框架 | 自定义PPO（不使用TRL） |
| 训练方式 | PPO + DPO |
| 训练环境 | RTX 4090 单卡 |
