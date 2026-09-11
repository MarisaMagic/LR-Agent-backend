# LR-Agent Backend

LR-Agent 后端服务，目前仅承担**用户注册与登录**。

## 功能范围

- 用户注册
- 用户登录

## 技术栈

- FastAPI + Uvicorn
- SQLAlchemy 2.0 (async) + Alembic
- PostgreSQL + Redis
- JWT + Argon2

## 部署方式

### 方式一：一键快速部署（Docker）

前置条件：已安装 Docker 与 Docker Compose v2。

```bash
cd docker
docker compose up -d --build
```

容器启动时会自动完成数据库迁移。

> 生产环境可在 `docker/` 下复制 `.env.prod.example` 为 `.env.prod` 并填写配置后，
> 使用 `docker-compose.prod.yml` 部署（含域名与自动 HTTPS）。

### 方式二：本地部署环境（适用于开发者）

**1. 创建并激活环境**

```bash
conda create -n lr-agent-backend python=3.12 -y
conda activate lr-agent-backend
pip install -r requirements.txt
```

**2. 配置环境变量**

```bash
cp .env.example .env
# 按需编辑 .env
```

**3. 启动基础设施（PostgreSQL / Redis / MinIO）**

```bash
cd docker
docker compose up -d postgres redis minio
```

**4. 执行数据库迁移**

```bash
alembic upgrade head
```

**5. 启动服务**

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 验证

- 健康检查：http://localhost:8000/health
- 接口文档：http://localhost:8000/docs

## 测试

```bash
pytest
```
