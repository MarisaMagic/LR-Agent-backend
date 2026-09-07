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

## 生产部署（VPS + 域名反代）

后端以 docker 镜像形式部署，PostgreSQL / Redis / MinIO 全部 compose 自托管，仅 Caddy 的 80/443 对公网开放（Let's Encrypt 自动 HTTPS）。

### 0. 前置条件

- 服务器已安装 Docker + Compose v2
- 域名两条 A 记录指向服务器 IP：`api.<domain>`、`minio.<domain>`
- 防火墙放行 80 / 443

### 1. 配置

```bash
cd docker
cp .env.prod.example .env.prod
# 编辑 .env.prod：填 DOMAIN、ACME_EMAIL、FRONTEND_HOST，
# 并用 openssl rand -hex 32 生成 SECRET_KEY、各类密码
```

### 2. 构建并启动

```bash
docker compose -f docker-compose.prod.yml --env-file .env.prod up -d --build
```

entrypoint 会自动执行数据库迁移；MinIO 桶与头像公开读策略在 api 启动时初始化。

### 3. 验证与操作

```bash
curl https://api.<domain>/health          # 期望 {"status":"ok"...}
docker compose -f docker-compose.prod.yml --env-file .env.prod logs -f api
docker compose -f docker-compose.prod.yml --env-file .env.prod down   # 停止（保留数据卷）
```

> 迁移/备份：数据在命名卷 `lr-agent-pg-data`、`lr-agent-redis-data`、`lr-agent-minio-data` 中；
> 升级代码一律 `up -d --build`，迁移由 entrypoint 幂等执行。

### 4. 客户端契约

Electron 客户端构建时必须把后端地址写入产物：`API_BASE_URL=https://api.<domain>/api/v1`
（dev 环境的 `resolveApiBaseUrl()` 按 serving host 的 8000 端口回退，反代 + HTTPS 场景不适用）。
对应的 CORS 白名单由 `FRONTEND_HOST` 控制；若打包客户端以 `file://` 协议直连后端仍被 CORS 拦截，
需在 `CORS_ORIGINS` 中追加 `"null"` 或经主进程代理转发。

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
