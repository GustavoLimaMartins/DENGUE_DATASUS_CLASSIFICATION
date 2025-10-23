# ==============================================================================
# ESTÁGIO 1: BUILDER (Instala dependências Python)
# Usa uma imagem Python completa para compilar pacotes complexos
# ==============================================================================
FROM python:3.11 AS builder

# Define o diretório de trabalho no container
WORKDIR /app

# Copia apenas o arquivo de dependências para aproveitar o cache do Docker
COPY requirements.txt .

# Instala as dependências Python
# --no-cache-dir economiza espaço
RUN pip install --no-cache-dir -r requirements.txt


# ==============================================================================
# ESTÁGIO 2: FINAL (Runtime)
# Usa a versão "slim" (mais leve) para o ambiente de execução
# ==============================================================================
FROM python:3.11-slim

# Instala a biblioteca de sistema libgomp1
# ESSENCIAL para o LightGBM, XGboost e outras libs de ML que usam paralelismo
RUN apt-get update && \
    apt-get install -y libgomp1 procps && \
    rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho final
WORKDIR /app

# 1. Copia as bibliotecas instaladas (do estágio builder para o estágio final)
# Adapte o path da versão do python se for diferente de 3.11
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# 2. Copia todos os seus arquivos de código e pastas
# A pasta "venv" será ignorada pelo .dockerignore
COPY . .

# Variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Comando de inicialização: Executa o seu arquivo principal
CMD ["python", "main.py"]