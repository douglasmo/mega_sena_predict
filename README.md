# Previsão da Mega-Sena com LSTM

Este projeto implementa um modelo de deep learning usando LSTM (Long Short-Term Memory) para analisar e prever números da Mega-Sena. O modelo é treinado com dados históricos dos sorteios e utiliza técnicas avançadas de processamento de sequências temporais.

## Características

- Download automático de dados históricos da Mega-Sena
- Pré-processamento e normalização dos dados
- Modelo LSTM com arquitetura otimizada
- Sistema de cache para evitar downloads repetidos
- Visualizações detalhadas do treinamento e previsões
- Exportação de resultados em Excel
- Logging completo para rastreamento de erros
- Configuração flexível via arquivo JSON

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

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/mega-sena.git
cd mega-sena
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure o arquivo `config.json` conforme necessário (um exemplo é fornecido)

## Uso

Execute o script principal:
```bash
python mega_sena.py
```

O programa irá:
1. Baixar os dados históricos (ou usar cache se disponível)
2. Treinar o modelo LSTM
3. Gerar previsões para o próximo sorteio
4. Criar visualizações
5. Exportar resultados em Excel

## Arquivos Gerados

- `historico_e_previsoes_megasena.xlsx`: Resultados detalhados em Excel
- `training_history.png`: Gráficos do histórico de treinamento
- `prediction_analysis.png`: Análise visual das previsões
- `mega_sena.log`: Log detalhado da execução

## Estrutura do Projeto

```
mega_sena/
├── mega_sena.py          # Script principal
├── config.json           # Configurações do modelo
├── requirements.txt      # Dependências do projeto
├── README.md            # Este arquivo
└── cache/               # Diretório para cache de dados
```

## Configuração

O arquivo `config.json` permite ajustar vários parâmetros:

```json
{
    "data_url": "https://loteriascaixa-api.herokuapp.com/api/megasena",
    "data_file": null,
    "export_file": "historico_e_previsoes_megasena.xlsx",
    "sequence_length": 10,
    "num_features": 60,
    "lstm_units": 128,
    "dropout_rate": 0.3,
    "epochs": 100,
    "batch_size": 32,
    "test_size_ratio": 0.15,
    "cache_duration_hours": 24,
    "cache_dir": "cache"
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


Claro! Você montou um **projeto muito bem estruturado** e profissional, parabéns! Agora vamos fazer uma aula detalhada e didática sobre como o **LSTM está sendo usado para prever números da Mega-Sena** nesse seu projeto.

---

## 🧠 **AULA EXPLICATIVA: Como o LSTM está sendo usado na previsão da Mega-Sena**

---

### 📌 1. **O que é LSTM (Long Short-Term Memory)?**

**LSTM** é um tipo de rede neural recorrente (RNN) especialmente eficaz para **sequências temporais**.  
Ela é capaz de **"lembrar" informações importantes ao longo do tempo**, filtrando o que esquecer e o que manter, diferente das RNNs tradicionais.

> **Exemplo de uso clássico:** Previsão de séries temporais, tradução automática, reconhecimento de fala, etc.

---

### 🎯 **No seu projeto, a LSTM está sendo usada para prever o PRÓXIMO sorteio da Mega-Sena com base em sorteios anteriores.**

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

---

#### 🔹 b) Sequência de entrada (X) e alvo (y)

Você usa uma **janela deslizante** para construir as sequências de entrada.  
Exemplo com `sequence_length = 3`:

| Entrada (X)                                       | Alvo (y)              |
|--------------------------------------------------|-----------------------|
| Sorteios 1, 2, 3 (3 vetores de 60 bits)          | Sorteio 4 (vetor)     |
| Sorteios 2, 3, 4                                  | Sorteio 5             |
| ...                                              | ...                   |

Assim, o LSTM aprende **padrões de como as dezenas evoluem** ao longo do tempo.

---

## 🏗️ 3. **A Estrutura do Modelo LSTM**

No seu código, a função `build_lstm_model()` cria o modelo assim:

```python
model = Sequential()
model.add(Input(shape=(sequence_length, num_features)))  # Ex: (30, 60)
model.add(LSTM(128, return_sequences=True))              # Primeira camada LSTM
model.add(Dropout(0.3))
model.add(LSTM(64))                                      # Segunda camada LSTM
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))                  # Camada densa intermediária
model.add(Dense(60, activation='sigmoid'))               # Saída com 60 probabilidades
```

### 🧩 Interpretação:
- **Input:** Sequência de 30 sorteios anteriores (cada um com 60 bits).
- **LSTM:** Processa a sequência e "lembra" de padrões.
- **Dense final:** Calcula a **probabilidade de cada número (1 a 60)** aparecer no próximo sorteio.

---

## 🧮 4. **Como é feita a previsão**

Após o treinamento:

- O modelo recebe a **última sequência de sorteios (X[-1])**
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

---

## 🧪 6. **Limitações importantes**

- A Mega-Sena é **aleatória por definição** (em teoria).
- Mesmo com redes neurais, não há garantia de prever corretamente.
- Esse projeto é **um exercício de Machine Learning**, não uma ferramenta infalível para ganhar na loteria.

---

## ✅ Conclusão

| Etapa | O que faz |
|-------|-----------|
| 1. Pré-processamento | Transforma sorteios em vetores binários |
| 2. Criação das sequências | Constrói X (entrada) e y (alvo) |
| 3. Modelo LSTM | Aprende padrões temporais nos sorteios |
| 4. Previsão | Gera 60 probabilidades, escolhe os 6 maiores |
| 5. Avaliação | Mede acertos reais + métricas de aprendizado |

---
