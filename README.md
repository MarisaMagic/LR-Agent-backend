# LR-Agent Backend

FastAPI 后端服务，为 LR-Agent Electron 客户端提供用户认证、Assist 对话编排与标注 LLM 能力。

## 架构说明

- **Electron 本地**：Agent 会话/消息、LLM Provider 配置（SQLite）
- **后端云端**：
  - 用户认证与资料（auth / users）
  - Assist 工具模式 SSE（`/agent/chat/stream`）
  - 标注/分析/质量报告 LLM 微服务（请求体直传 `api_key/base_url/model`）

## 技术栈

- FastAPI + Uvicorn
- SQLAlchemy 2.0 (async) + Alembic
- PostgreSQL 17 + Redis 7 + MinIO
- JWT (Access + Refresh) + Argon2
- LangChain（Assist 与标注 LLM 编排）

## 快速开始

### 1. Conda 环境

```bash
conda create -n lr-agent-backend python=3.12 -y
conda activate lr-agent-backend
pip install -r requirements.txt
```

### 2. 环境变量

```bash
cp .env.example .env
# 编辑 .env，至少设置 SECRET_KEY、POSTGRES_PASSWORD、REDIS_PASSWORD
```

Docker 基础设施使用 `docker/.env`（已提供开发默认值，与根目录 `.env` 密码保持一致）。

### 3. 启动基础设施

```bash
cd docker
docker compose up -d postgres redis minio
```

### 4. 数据库迁移

```bash
alembic upgrade head
```

### 5. 启动 API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- 健康检查：http://localhost:8000/health
- OpenAPI 文档：http://localhost:8000/docs

### 6. 一键启动（含 API 容器）

```bash
cd docker
docker compose up -d --build
```

API 容器启动时会自动执行 `alembic upgrade head`，并在 lifespan 中初始化 MinIO bucket。

## 项目结构

```
app/
├── main.py              # FastAPI 入口
├── core/                # 配置、安全、依赖注入
├── db/                  # 数据库会话、Redis 连接
├── models/              # SQLAlchemy 模型（users）
├── schemas/             # Pydantic DTO
├── agent/               # Assist / 标注 / 分析 LLM 逻辑
├── api/v1/              # 路由
├── services/            # 业务逻辑
└── middleware/          # 限流等中间件
```

## 已实现能力

### 用户与账号

- 用户注册 / 登录 / 刷新 / 登出
- 邮箱验证、忘记/重置密码
- `GET/PATCH /users/me`
- MinIO 头像上传
- Redis refresh token 轮换与 reuse 检测

### Agent Assist

- `POST /api/v1/agent/chat/stream` — 无状态 SSE（前端直传 Provider 配置）
- `POST /api/v1/agent/chat/cancel` — 取消生成任务
- Assist 工具循环（工作区搜索、文件读取、MCP 等）

### 标注 / 分析 / 质量 LLM

- `POST /api/v1/agent/annotation/*` — 批量准备、标签映射、评判等
- `POST /api/v1/agent/analysis/prepare` — 数据分析脚本生成
- `POST /api/v1/agent/analysis/summarize/stream` — 分析结果解读
- `POST /api/v1/agent/annotation-quality/report/compose/stream` — 质量报告撰写

## API 概览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/auth/register` | 注册 |
| POST | `/api/v1/auth/login` | 登录 |
| POST | `/api/v1/auth/refresh` | 刷新 token |
| POST | `/api/v1/auth/logout` | 登出 |
| POST | `/api/v1/auth/revoke-all-sessions` | 撤销所有设备 |
| POST | `/api/v1/auth/verify-email` | 验证邮箱 |
| POST | `/api/v1/auth/resend-verification-email` | 重发验证邮件（需登录） |
| POST | `/api/v1/auth/forgot-password` | 忘记密码 |
| POST | `/api/v1/auth/reset-password` | 重置密码 |
| GET/PATCH | `/api/v1/users/me` | 用户资料 |
| POST/DELETE | `/api/v1/users/me/avatar` | 头像上传/删除 |
| POST | `/api/v1/users/me/delete-account` | 注销账号（软删除，需密码） |
| POST | `/api/v1/agent/chat/stream` | Assist SSE 流式对话 |
| POST | `/api/v1/agent/chat/cancel` | 取消 Assist 任务 |
| POST | `/api/v1/agent/annotation/*` | 标注 LLM 服务 |
| POST | `/api/v1/agent/analysis/*` | 数据分析 LLM 服务 |
| POST | `/api/v1/agent/annotation-quality/*` | 质量报告 LLM 服务 |

## 测试

```bash
pytest
```

测试使用 fakeredis + 数据库事务回滚，需本地 PostgreSQL 可用且已执行迁移。
