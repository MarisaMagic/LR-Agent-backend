# LR-Agent Backend

FastAPI 云端后端，为 LR-Agent Electron 客户端提供**用户认证与账号管理**。

> Agent 编排（Assist 工具循环、标注 / 质量报告 LLM）已迁移至本机服务
> `LR-Agent-local`（由 Electron 主进程 spawn，仅监听 127.0.0.1），本后端不再承担。

## 架构说明

- **Electron 本地**：Agent 会话/消息、LLM Provider 配置（SQLite）
- **LR-Agent-local（本机）**：Assist 工具循环 SSE、标注/质量报告 LLM 编排
- **本后端（云端）**：用户认证与资料（auth / users），仅保留账号体系

## 技术栈

- FastAPI + Uvicorn
- SQLAlchemy 2.0 (async) + Alembic
- PostgreSQL 17 + Redis 7 + MinIO
- JWT (Access + Refresh) + Argon2

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
├── schemas/             # Pydantic DTO（user）
├── api/v1/              # 路由（auth / users）
├── services/            # 业务逻辑（认证、资料、头像、邮件）
└── middleware/          # 限流等中间件
```

## 已实现能力

### 用户与账号

- 用户注册 / 登录 / 刷新 / 登出
- 邮箱验证、忘记/重置密码
- `GET/PATCH /users/me`
- MinIO 头像上传
- Redis refresh token 轮换与 reuse 检测

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

## 测试

```bash
pytest
```

测试使用 fakeredis + 数据库事务回滚，需本地 PostgreSQL 可用且已执行迁移。
