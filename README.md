# LR-Agent Backend

FastAPI 后端服务，为 LR-Agent Electron 客户端提供用户认证与资料管理 API。

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
├── models/              # SQLAlchemy 模型
├── schemas/             # Pydantic DTO
├── api/v1/              # 路由（auth、users）
├── services/            # 业务逻辑
└── middleware/          # 限流等中间件
```

## Phase 1 已实现

- 用户注册 / 登录 / 刷新 / 登出
- 邮箱验证、忘记/重置密码
- `GET/PATCH /users/me`
- Redis refresh token 轮换与 reuse 检测
- 基础限流中间件

## Phase 3 已实现（Agent P0/P1）

- `GET/POST/PATCH/DELETE /api/v1/llm-providers` 大模型配置（API Key 使用 `LLM_SECRETS_MASTER_KEY` 加密，与 JWT `SECRET_KEY` 分离）
- Agent 安全：SSRF 校验 `base_url`、`/agent/chat/cancel` 需登录且 job 归属校验、流式限流、SSE 错误脱敏
- 删除会话级联删除消息；Redis 缓存 key 含 `user_id`
- `POST /api/v1/llm-providers/{id}/default` 设置默认模型
- `POST /api/v1/agent/chat/stream` SSE 流式对话（LangChain）
- `POST /api/v1/agent/chat/cancel` 取消生成任务
- `GET/POST/PATCH/DELETE /api/v1/agent/sessions` 会话 CRUD（列表游标分页 + 摘要字段）
- `GET /api/v1/agent/sessions/{id}` 会话详情（消息 `before_message_id` 分页，默认最近 N 条）
- 聊天记录 PostgreSQL 持久化 + Redis 热缓存
- 意图路由（chat / assist）与只读轻工具（账户、帮助、客户端上下文）

## Phase 2 已实现

- MinIO 头像上传（JPEG/PNG/WebP，Pillow 校验，多尺寸 WebP）
- `POST/DELETE /users/me/avatar`
- SMTP 邮件（未配置 SMTP 时回退为日志 mock）
- `POST /auth/revoke-all-sessions` 撤销所有设备
- `user_sessions` 审计表（IP、User-Agent、revoked_at）
- Docker 启动时自动迁移

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
