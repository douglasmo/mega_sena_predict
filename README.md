# Previsão da Mega-Sena com GRU

Este projeto implementa um modelo de deep learning usando GRU (Gated Recurrent Unit) para analisar e prever números da Mega-Sena. O modelo é treinado com dados históricos dos sorteios e utiliza técnicas avançadas de processamento de sequências temporais, incluindo features estatísticas e temporais.

## Características

- Download automático de dados históricos da Mega-Sena
- Pré-processamento e normalização dos dados
- Modelo GRU com arquitetura otimizada e Batch Normalization
- Features estatísticas avançadas (paridade, soma, range, zonas, frequência rolante)
- Features temporais (sorteios desde última aparição)
- Sistema de cache para evitar downloads repetidos
- Visualizações detalhadas do treinamento e previsões
- Exportação de resultados em Excel
- Logging completo para rastreamento de erros
- Configuração flexível via arquivo JSON
- Suporte a TensorBoard para monitoramento do treinamento
- Otimização automática de hiperparâmetros

## Requisitos

- Python 3.8+
- Bibliotecas Python (instaladas via `pip install -r requirements.txt`):
  - numpy
  - pandas
  - tensorflow
  - scikit-learn
  - matplotlib
  - openpyxl
  - requests
  - tqdm

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/mega-sena.git
cd mega-sena
```

2. Configure o ambiente virtual e instale as dependências (Windows):
```bash
setup_env.bat
```

Ou manualmente:
```bash
python -m venv .env
source .env/bin/activate  # Linux/Mac
.env\Scripts\activate     # Windows
pip install -r requirements.txt
```

3. Configure o arquivo `configv3.json` conforme necessário (um exemplo é fornecido)

## Uso

### Usando menus (Windows)
Execute o arquivo batch e siga as instruções:
```bash
run_mega_sena.bat
```

### Linha de comando
```bash
# Execução normal
python run_mega_sena.py

# Teste de hiperparâmetros com Grid Search
python run_mega_sena.py --test-hyperparameters --method grid

# Teste de hiperparâmetros com Random Search (ex: 15 iterações)
python run_mega_sena.py --test-hyperparameters --method random --iterations 15
```

O programa irá:
1. Baixar os dados históricos (ou usar cache se disponível)
2. Calcular features estatísticas e temporais
3. Treinar o modelo GRU (ou otimizar hiperparâmetros se solicitado)
4. Gerar previsões para o próximo sorteio
5. Criar visualizações
6. Exportar resultados em Excel

## Otimização de Hiperparâmetros

O sistema permite buscar automaticamente a melhor configuração de hiperparâmetros:

1. **Grid Search**: Testa todas as combinações possíveis de parâmetros
2. **Random Search**: Testa um subconjunto aleatório de combinações

Os hiperparâmetros testáveis incluem:
- Tamanho da sequência (`sequence_length`)
- Unidades GRU (`gru_units`)
- Taxa de dropout (`dropout_rate`)
- Uso de Batch Normalization (`use_batch_norm`)
- Tamanho do batch (`batch_size`)

Configure os parâmetros a serem testados no arquivo `configv3.json`:

```json
"hyperparameter_search": {
    "method": "grid",
    "n_iterations": 20,
    "param_grid": {
        "sequence_length": [10, 15, 20, 25],
        "gru_units": [128, 192, 256, 320],
        ...
    }
}
```

## Arquivos Gerados

- `historico_e_previsoes_megasena_v3.xlsx`: Resultados detalhados em Excel
- `training_history_v3.png`: Gráficos do histórico de treinamento
- `prediction_analysis_v3.png`: Análise visual das previsões
- `hits_over_time_v3.png`: Gráfico de acertos ao longo do tempo
- `mega_sena_v3.log`: Log detalhado da execução
- `logs/fit/`: Diretório com logs do TensorBoard
- `hyperparameter_results.xlsx`: Resultados da otimização de hiperparâmetros
- `hyperparameter_analysis.png`: Análise gráfica dos hiperparâmetros
- `top_hyperparameters.png`: Gráfico das melhores configurações

## Estrutura do Projeto

```
mega_sena/
├── mega_sena_v3.py      # Script principal
├── hyperparameter_tuning.py # Módulo de otimização de hiperparâmetros
├── configv3.json        # Configurações do modelo
├── run_mega_sena.py     # Script para execução via terminal
├── run_mega_sena.bat    # Script para execução via Windows
├── setup_env.bat        # Script para configurar ambiente virtual
├── requirements.txt     # Dependências do projeto
├── README.md            # Este arquivo
├── .env/                # Ambiente virtual Python
├── output/              # Diretório para arquivos de saída
├── cache/               # Diretório para cache de dados
└── logs/                # Diretório para logs do TensorBoard
```

## Configuração

O arquivo `configv3.json` permite ajustar vários parâmetros:

```json
{
    "data_url": "https://loteriascaixa-api.herokuapp.com/api/megasena",
    "data_file": null,
    "export_file": "output/historico_e_previsoes_megasena_v3.xlsx",
    "sequence_length": 20,                   
    "num_features_base": 60,
    "num_features_time": 60,
    "rolling_freq_windows": [10, 50, 100],
    "gru_units": 256,                     
    "use_batch_norm": true,              
    "dropout_rate": 0.45,               
    "epochs": 250,                    
    "batch_size": 64,                     
    "test_size_ratio": 0.15,
    "validation_split_ratio": 0.15,
    "cache_duration_hours": 24,
    "cache_dir": "output/cache",
    "tensorboard_log_dir": "output/logs/fit/",
    "test_hyperparameters": false
}
```

## Contribuição

Contribuições são bem-vindas! Por favor, sinta-se à vontade para:
1. Fazer um fork do projeto
2. Criar uma branch para sua feature
3. Fazer commit das mudanças
4. Abrir um Pull Request

## Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Contato

Se você tiver dúvidas ou sugestões, por favor abra uma issue no GitHub.

---

## 🧠 **AULA EXPLICATIVA: Como o GRU está sendo usado na previsão da Mega-Sena**

---

### 📌 1. **O que é GRU (Gated Recurrent Unit)?**

**GRU** é um tipo de rede neural recorrente (RNN) similar ao LSTM, mas com uma arquitetura mais simples.  
Ela também é capaz de **"lembrar" informações importantes ao longo do tempo**, usando portas (gates) para controlar o fluxo de informação.

> **Exemplo de uso clássico:** Previsão de séries temporais, tradução automática, reconhecimento de fala, etc.

---

### 🎯 **No seu projeto, a GRU está sendo usada para prever o PRÓXIMO sorteio da Mega-Sena com base em sorteios anteriores e features estatísticas.**

---

## 📦 2. **Como os dados são estruturados**

Imagine a Mega-Sena como uma **série de eventos temporais**: cada sorteio é um momento no tempo, com 6 dezenas sorteadas.

### ✅ Etapas de preparação:

#### 🔹 a) One-Hot Encoding com `MultiLabelBinarizer`
Cada sorteio é transformado em um vetor de 60 posições (números de 1 a 60):
- Se o número 5 foi sorteado, a posição 5 recebe `1`.
- Se o número 23 **não** foi sorteado, a posição 23 recebe `0`.

Exemplo:
```text
Sorteio: [5, 12, 18, 33, 45, 60]
Vetor:   [0, 0, 0, 0, 1, ..., 1]  ← 60 posições
```

#### 🔹 b) Features Estatísticas
Para cada sorteio, calculamos:
- Contagem de números ímpares
- Soma dos números
- Range (diferença entre maior e menor número)
- Distribuição por zonas (4 zonas de 15 números)
- Frequência rolante em diferentes janelas (10, 50, 100 sorteios)

#### 🔹 c) Features Temporais
Para cada número (1-60), calculamos:
- Quantos sorteios se passaram desde a última aparição

---

#### 🔹 d) Sequência de entrada (X) e alvo (y)

Você usa uma **janela deslizante** para construir as sequências de entrada.  
Exemplo com `sequence_length = 15`:

| Entrada (X)                                       | Alvo (y)              |
|--------------------------------------------------|-----------------------|
| Sorteios 1-15 (15 vetores combinados)            | Sorteio 16 (vetor)    |
| Sorteios 2-16                                    | Sorteio 17            |
| ...                                              | ...                   |

Cada vetor de entrada combina:
- Vetor one-hot do sorteio
- Features temporais
- Features estatísticas

---

## 🏗️ 3. **A Estrutura do Modelo GRU**

No seu código, a função `build_model()` cria o modelo assim:

```python
model = Sequential()
model.add(Input(shape=(sequence_length, num_features_total)))
model.add(BatchNormalization())  # Opcional
model.add(GRU(192, return_sequences=True))
model.add(BatchNormalization())  # Opcional
model.add(Dropout(0.4))
model.add(GRU(96))  # Segunda camada GRU
model.add(BatchNormalization())  # Opcional
model.add(Dropout(0.4))
model.add(Dense(96, activation='relu'))
model.add(BatchNormalization())  # Opcional
model.add(Dropout(0.4))
model.add(Dense(60, activation='sigmoid'))
```

### 🧩 Interpretação:
- **Input:** Sequência de 15 sorteios anteriores com todas as features
- **GRU:** Processa a sequência e "lembra" de padrões
- **BatchNorm:** Normaliza as ativações entre camadas
- **Dropout:** Previne overfitting
- **Dense final:** Calcula a **probabilidade de cada número (1 a 60)** aparecer no próximo sorteio

---

## 🧮 4. **Como é feita a previsão**

Após o treinamento:

- O modelo recebe a **última sequência de sorteios com todas as features**
- Retorna 60 probabilidades (sigmoid output entre 0 e 1)
- Os **6 números com maiores probabilidades** são selecionados como a "previsão"

Exemplo:
```python
predicted_indices = np.argsort(probabilidades)[-6:]
predicted_numbers = sorted((predicted_indices + 1).tolist())
```

---

## 📈 5. **Avaliação real do modelo**

Você avalia o desempenho de duas formas:

### ✔️ Avaliação Técnica:
- `Loss`: Erro da previsão (função de perda binária)
- `Binary Accuracy`: Quantos bits acertou
- `AUC`: Quão bem o modelo diferencia 1s de 0s (números sorteados ou não)

### ✔️ Avaliação Prática (mais útil!):
- Compara os 6 números previstos com os sorteados reais
- Conta quantos acertos o modelo teve por sorteio
- Mostra a média e distribuição de acertos
- Gera gráficos de acertos ao longo do tempo

---

## 🧪 6. **Limitações importantes**

- A Mega-Sena é **aleatória por definição** (em teoria).
- Mesmo com redes neurais e features avançadas, não há garantia de prever corretamente.
- Esse projeto é **um exercício de Machine Learning**, não uma ferramenta infalível para ganhar na loteria.

---

## ✅ Conclusão

| Etapa | O que faz |
|-------|-----------|
| 1. Pré-processamento | Transforma sorteios em vetores e calcula features |
| 2. Criação das sequências | Constrói X (entrada) e y (alvo) |
| 3. Modelo GRU | Aprende padrões temporais e estatísticos |
| 4. Previsão | Gera 60 probabilidades, escolhe os 6 maiores |
| 5. Avaliação | Mede acertos reais + métricas de aprendizado |

---
