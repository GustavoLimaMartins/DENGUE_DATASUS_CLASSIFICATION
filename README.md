# 🦟 Desafio Dengue: Pipeline Completo de Machine Learning & Deep Learning com Docker

Este projeto implementa uma arquitetura **end-to-end** para análise e predição de notificações de dengue no Brasil (2019–2024), utilizando dados do **DATASUS/SINAN**. O pipeline cobre desde a extração dos arquivos `.dbf`, análise exploratória, pré-processamento avançado, até a modelagem com algoritmos de Machine Learning e Deep Learning, incluindo avaliação detalhada com métricas robustas.

---

## 🚀 Como Executar com Docker

1. **Baixe a imagem pronta:**
   ```sh
   docker pull gulimamartins/dengue-techchallenge:latest
   ```

2. **Execute o pipeline (recomenda-se 8GB de RAM):**
   ```sh
   docker run --memory="8g" gulimamartins/dengue-techchallenge:latest
   ```

---

## 🏗️ Arquitetura do Pipeline

⚠️ Por restrições de privacidade e tamanho, o dataset original está comprimido neste repositório.
Um arquivo compactado contendo o dataset normalizado `.parquet_normalizado.zip` é disponibilizado separadamente.
Antes de executar o modelo localmente via clone, é necessário descompactar o arquivo de dados na pasta `files/`

### 1️⃣ Extração dos Dados `.dbf` do DATASUS/SINAN

- **Arquivo:** `data_extract/a_extract_files_dbf.py`
- **Função:** Converte arquivos `.dbf` (expandidos do `.dbc` via TABWIN) em arquivos Parquet otimizados.
- **Fluxo:** 
  - Lê cada arquivo `.dbf` de 2019 a 2024.
  - Converte em DataFrame e salva em `files/.parquet`.
  - Permite processar milhões de registros de forma eficiente.

### 2️⃣ Normalização e Limpeza dos Dados

- **Arquivo:** `data_extract/b_convert_in_dataframe.py`
- **Função:** Aplica schema consistente, converte tipos de dados, trata valores nulos e salva Parquet normalizado.
- **Destaques:**
  - Casting de datas, inteiros, strings.
  - Remoção de linhas incompletas.
  - Pronto para análise e modelagem.

### 3️⃣ Engenharia de Features e Formatação SINAN

- **Arquivo:** `data_pre_processing/c_data_formatting.py`
- **Função:** 
  - Decodifica idade (`NU_IDADE_N`) para anos e faixa etária.
  - Extrai mês e semana ISO das datas.
  - Codifica sexo, escolaridade, gestante, evolução e classificação final em categorias reduzidas.
- **Benefício:** Facilita o uso dos dados em modelos ML/DL e reduz ruído.

### 4️⃣ Pré-processamento Avançado

- **Arquivo:** `data_pre_processing/e_data_pre_process.py`
- **Funções:**
  - **One-hot encoding** para variáveis categóricas.
  - **Codificação cíclica** (seno/cosseno) para mês e semana, preservando sazonalidade.
  - **Target encoding** para municípios, suavizando por média global.
  - **Normalização** (MinMax) de variáveis numéricas.
  - **Split** treino/teste estratificado.
- **Saída:** DataFrames prontos para modelagem.

### 5️⃣ Análise Exploratória de Dados (EDA)

- **Arquivo:** `data_tools_analisys/d_data_analisys.py`
- **Função:** 
  - Gera estatísticas descritivas, gráficos de correlação, histogramas e boxplots.
  - Permite entender padrões, outliers e relações entre variáveis.

### 6️⃣ Balanceamento de Classes (Oversampling)

- **Técnica:** SMOTE (`imblearn`)
- **Motivo:** Corrige desbalanceamento entre classes (ex: casos graves são raros).
- **Aplicação:** Apenas no conjunto de treino, evitando vazamento de dados.

### 7️⃣ Setup de Modelagem, Avaliação e Métricas
**Arquivo:** `f_modeling_setup_and_evaluate.py`
**Função:**
  - Centraliza o contexto dos dados (treino/teste), aplica oversampling, configura e executa GridSearchCV, e realiza avaliação detalhada dos modelos.

Fluxo:
  - Inicializa o contexto dos dados, separando X_train, X_test, y_train, y_test.
  - Configura GridSearchCV com scorer customizado (recall ponderado, priorizando casos graves).
  - Avalia modelos com métricas robustas: relatório de classificação, matriz de confusão, curva ROC multiclasses.
  - Calcula e plota Permutation Importance para todos os modelos e Feature Importance para modelos de árvore.
  - Garante que a avaliação seja feita de forma padronizada e visual, facilitando a comparação entre algoritmos.

### 8️⃣ Modelagem: Machine Learning & Deep Learning

#### 🔹 Machine Learning

- **Arquivo:** `models/g_model_ml.py`
- **Modelos:** 
  - **KNN:** Busca do melhor K via erro médio.
  - **Random Forest & LightGBM:** Seleção de hiperparâmetros via `GridSearchCV` com recall ponderado.
- **Pipeline:** 
  - Oversampling → Treino → Avaliação.

#### 🔸 Deep Learning

- **Arquivo:** `models/g_model_dl.py`
- **Modelo:** Rede neural densa (Keras) com camadas de BatchNorm e Dropout.
- **Ajustes:** 
  - `class_weight` para compensar desbalanceamento.
  - Early stopping para evitar overfitting.
- **Permutation Importance:** Avaliação da importância das features mesmo em modelos "caixa-preta".

---

## 📊 Avaliação e Métricas

- **Matriz de Confusão:** Visualiza acertos/erros por classe.
- **Relatório de Classificação:** Precision, recall, F1-score, suporte.
- **Curva ROC Multiclasse:** AUC macro e curvas ROC por classe.
- **Feature Importance (árvores):** Importância das variáveis em Random Forest/LightGBM.
- **Permutation Importance:** Importância das variáveis via embaralhamento, inclusive para redes neurais.

---

## 🗂️ Organização dos Arquivos

- `main.py` — Orquestrador do pipeline.
- `data_extract/` — Extração e normalização dos dados.
- `data_pre_processing/` — Engenharia de features e pré-processamento.
- `data_tools_analisys/` — EDA, setup de modelagem e avaliação.
- `models/` — Pipelines de ML e DL.
- `files/` — Dados intermediários, dicionários e resultados.

---

## 💡 Diferenciais Técnicos

- **Pipeline modular e reprodutível**: cada etapa pode ser executada isoladamente.
- **Pré-processamento robusto**: encoding cíclico, target encoding, normalização.
- **GridSearchCV com recall ponderado**: prioriza casos graves.
- **Avaliação exaustiva**: matriz de confusão, ROC, feature importance, permutation importance.
- **Compatível com Docker**: fácil de rodar em qualquer ambiente.

---

## 📚 Referências

- [DATASUS/SINAN](https://datasus.saude.gov.br/)
- [Scikit-learn](https://scikit-learn.org/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [Keras/TensorFlow](https://keras.io/)
- [Imbalanced-learn (SMOTE)](https://imbalanced-learn.org/)

---

## 👨‍💻 Para Desenvolvedores

- Para rodar localmente, instale as dependências do `requirements.txt` e execute `main.py`.
- Para modificar o pipeline, altere os scripts nas pastas correspondentes.
- Resultados e métricas foram salvos em `files/resultados/`.

---

## 🏁 Pronto para rodar?  
**Basta executar os comandos Docker acima e acompanhar os resultados!**

---
