FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# 将项目安装到 `/app`
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1

# 由于使用挂载卷，使用复制缓存而非硬链接
ENV UV_LINK_MODE=copy

# 基于锁文件与配置安装项目依赖
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# 再拷贝其余源码并安装项目本体
# 将依赖安装与源码安装分层可提升镜像缓存命中率
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# 将虚拟环境可执行文件放到 PATH 前部
ENV PATH="/app/.venv/bin:$PATH"

# 重置入口点，避免通过 `uv` 启动
ENTRYPOINT []

CMD ["uvicorn", "src.api.medical_qa_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
