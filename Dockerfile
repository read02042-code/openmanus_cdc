FROM python:3.12-slim

# 1. 设置工作目录 (与 docker-compose 保持一致)
WORKDIR /app
ENV PYTHONPATH=/app

# 2. 安装系统依赖 (添加了编译所需的 build-essential，因为 faiss 或 scipy 偶尔需要编译)
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 预复制依赖文件 (利用 Docker 缓存机制，只要要求没变，这一步就不会重新下载)
COPY requirements.txt .

# 4. 安装依赖 (确保你在 requirements.txt 里已经加了 fastapi, faiss-cpu, python-docx, scipy)
RUN uv pip install --system --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 5. 复制项目所有文件
COPY . .

# 6. 暴露 Web 端口 (任务书交互层需求)
EXPOSE 8000

# 7. 启动程序 (初期调试可以用 bash，后期改为运行 main.py)
CMD ["bash"]
