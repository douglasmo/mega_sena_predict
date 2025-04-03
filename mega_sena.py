# -*- coding: utf-8 -*-
"""
Script de Exemplo para "Previsão" da Mega-Sena usando LSTM - Versão Aprimorada.

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Usaremos para validação, mas teste será cronológico
from sklearn.preprocessing import MultiLabelBinarizer
# Use TensorFlow/Keras para a Rede Neural LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import requests # Para baixar os dados (opcional)
from io import StringIO # Para ler os dados baixados
import os
import warnings
import matplotlib.pyplot as plt # Para gráficos
import logging
import json
from datetime import datetime, timedelta
import hashlib
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mega_sena.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ignorar warnings de performance do TensorFlow (opcional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Configuração via Arquivo ---
def load_config(config_file='config.json'):
    """Carrega configurações de um arquivo JSON."""
    default_config = {
        "data_url": "https://loteriascaixa-api.herokuapp.com/api/megasena",
        "data_file": None,
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
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                # Atualiza apenas as chaves existentes no arquivo
                for key in config:
                    if key in default_config:
                        default_config[key] = config[key]
                logger.info(f"Configurações carregadas de {config_file}")
        else:
            logger.warning(f"Arquivo de configuração {config_file} não encontrado. Usando configurações padrão.")
    except Exception as e:
        logger.error(f"Erro ao carregar configurações: {e}")
    
    return default_config

# Carrega configurações
config = load_config()

# --- Sistema de Cache ---
def get_cache_key(url):
    """Gera uma chave única para o cache baseada na URL."""
    return hashlib.md5(url.encode()).hexdigest()

def is_cache_valid(cache_file, duration_hours):
    """Verifica se o cache ainda é válido."""
    if not os.path.exists(cache_file):
        return False
    cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
    return datetime.now() - cache_time < timedelta(hours=duration_hours)

def save_to_cache(data, cache_file):
    """Salva dados no cache."""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        logger.info(f"Dados salvos no cache: {cache_file}")
    except Exception as e:
        logger.error(f"Erro ao salvar cache: {e}")

def load_from_cache(cache_file):
    """Carrega dados do cache."""
    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Erro ao carregar cache: {e}")
        return None

# --- Funções (incluindo as de download, preprocessamento, criação de sequências - mantidas da versão anterior) ---

def download_and_prepare_data(url=None, file_path=None):
    """
    Baixa os dados da Mega-Sena de uma URL ou carrega de um arquivo CSV local.
    Implementa cache e melhor tratamento de erros.
    """
    logger.info("Iniciando carregamento de dados...")
    df = None

    if url:
        cache_key = get_cache_key(url)
        cache_file = os.path.join(config['cache_dir'], f"{cache_key}.json")
        
        # Tenta carregar do cache primeiro
        if is_cache_valid(cache_file, config['cache_duration_hours']):
            logger.info("Carregando dados do cache...")
            data = load_from_cache(cache_file)
            if data:
                logger.info("Dados carregados com sucesso do cache.")
            else:
                logger.warning("Cache inválido ou corrompido. Baixando dados novamente.")
        else:
            logger.info("Cache expirado ou não encontrado. Baixando dados...")
            try:
                response = requests.get(url, verify=False, timeout=30)
                response.raise_for_status()
                data = response.json()
                save_to_cache(data, cache_file)
            except requests.exceptions.RequestException as e:
                logger.error(f"Erro ao baixar dados: {e}")
                return None

        if isinstance(data, list) and data and 'dezenas' in data[0]:
            results = []
            concursos = []
            datas = []
            for sorteio in data:
                try:
                    dezenas = sorted([int(d) for d in sorteio.get('dezenas', [])])
                    if len(dezenas) == 6 and all(1 <= d <= 60 for d in dezenas):
                        results.append(dezenas)
                        concursos.append(sorteio.get('concurso'))
                        datas.append(sorteio.get('data'))
                    else:
                        logger.warning(f"Sorteio inválido encontrado: {sorteio}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Erro ao processar sorteio: {e}")
                    continue

            if not results:
                logger.error("Nenhum sorteio válido encontrado nos dados baixados.")
                return None

            df = pd.DataFrame(results, columns=[f'Bola{i+1}' for i in range(6)])
            if concursos: df['Concurso'] = concursos
            if datas: df['Data'] = pd.to_datetime(datas, format='%d/%m/%Y')

            # Ordena pelo concurso (mais antigo primeiro)
            if 'Concurso' in df.columns:
                df = df.sort_values(by='Concurso').reset_index(drop=True)
            elif 'Data' in df.columns:
                df = df.sort_values(by='Data').reset_index(drop=True)

            logger.info(f"Dados processados com sucesso ({len(df)} sorteios).")

        else:
            print("Formato de dados JSON da API não reconhecido ou inesperado.")
            # Tenta ler como CSV como fallback (menos provável para APIs modernas)
            try:
                print("Tentando ler a resposta como CSV...")
                df = pd.read_csv(StringIO(response.text))
                print("Dados lidos como CSV.")
            except Exception as e_csv:
                print(f"Não foi possível interpretar a resposta como JSON ou CSV: {e_csv}")
                return None # Falha em ambas as tentativas

    # Se o download falhou ou não foi tentado, tenta o arquivo local
    if df is None and file_path and os.path.exists(file_path):
        print(f"Tentando carregar do arquivo local: {file_path}")
        try:
            df = pd.read_csv(file_path, sep=None, engine='python') # Tenta detectar separador
            # Verifica se a detecção funcionou (se temos mais de 1 coluna)
            if df.shape[1] < 6:
                 print(f"Arquivo CSV lido, mas parece ter poucas colunas ({df.shape[1]}). Verifique o separador.")
                 # Tenta alguns separadores comuns
                 for sep in [';', ',', '\t']:
                     try:
                         df_try = pd.read_csv(file_path, sep=sep)
                         if df_try.shape[1] >= 6:
                             df = df_try
                             print(f"Separador '{sep}' funcionou.")
                             break
                     except:
                         continue
            print(f"Dados carregados de {file_path}")
        except Exception as e_file:
            print(f"Erro ao carregar arquivo local: {e_file}")
            return None

    # Se ainda não temos DataFrame, sai
    if df is None:
        print("Nenhuma fonte de dados (URL ou arquivo) funcionou ou foi fornecida.")
        return None

    # --- Bloco de identificação e renomeação de colunas (aprimorado) ---
    bola_cols_found = []
    potential_col_names = [f'Bola{i+1}' for i in range(6)] + \
                          [f'bola{i+1}' for i in range(6)] + \
                          [f'Dezena{i+1}' for i in range(6)] + \
                          [f'dezena{i+1}' for i in range(6)] + \
                          [f'N{i+1}' for i in range(6)] # Adiciona mais padrões

    # Prioriza nomes exatos ou variações comuns
    for pattern_list in [[f'Bola{i+1}' for i in range(6)], [f'Dezena{i+1}' for i in range(6)]]:
        if all(col in df.columns for col in pattern_list):
            bola_cols_found = pattern_list
            break

    # Se não encontrou, tenta identificar heuristicamente
    if not bola_cols_found:
        numeric_cols = df.select_dtypes(include=np.number).columns
        potential_bola_cols = [c for c in numeric_cols if df[c].between(1, 60, inclusive='both').all() and df[c].notna().all()]
        if len(potential_bola_cols) >= 6:
            # Pega as primeiras 6 colunas que se encaixam no critério
            bola_cols_found = potential_bola_cols[:6]
            print(f"Colunas de bolas identificadas heuristicamente como: {bola_cols_found}")
        else:
            print(f"Erro: Não foi possível identificar 6 colunas com números entre 1 e 60.")
            print(f"Colunas encontradas: {list(df.columns)}")
            return None

    # Renomeia para o padrão 'BolaX' se necessário e seleciona
    rename_map = {found_col: f'Bola{i+1}' for i, found_col in enumerate(bola_cols_found)}
    df.rename(columns=rename_map, inplace=True)
    bola_cols = [f'Bola{i+1}' for i in range(6)]

    # Garante que as colunas das bolas são numéricas inteiras
    try:
        for col in bola_cols:
            df[col] = pd.to_numeric(df[col]).astype(int)
        print("Colunas das bolas verificadas e convertidas para inteiro.")
    except Exception as e_num:
        print(f"Erro ao converter colunas de bolas para numérico: {e_num}")
        return None

    # Seleciona e retorna apenas as colunas relevantes (Bolas e talvez Concurso/Data para referência)
    cols_to_keep = bola_cols
    if 'Concurso' in df.columns: cols_to_keep.append('Concurso')
    if 'Data' in df.columns: cols_to_keep.append('Data')

    final_df = df[cols_to_keep].copy()

    # Ordena novamente para garantir, caso a ordenação original tenha se perdido
    if 'Concurso' in final_df.columns:
        final_df = final_df.sort_values(by='Concurso').reset_index(drop=True)
    elif 'Data' in final_df.columns:
        final_df = final_df.sort_values(by='Data').reset_index(drop=True)

    print(f"Total de {len(final_df)} sorteios carregados e formatados.")
    return final_df # Retorna o DataFrame com Bolas e possivelmente Concurso/Data


def preprocess_data(df_balls_only):
    """
    Transforma os números sorteados (DataFrame apenas com colunas BolaX)
    em formato MultiLabelBinarizer (One-Hot Encoding para múltiplas labels).
    """
    logger.info("Iniciando pré-processamento dos dados...")
    try:
        # Validação dos dados
        if df_balls_only.empty:
            logger.error("DataFrame vazio recebido para pré-processamento")
            return None, None
            
        # Verifica se todas as colunas necessárias existem
        required_cols = [f'Bola{i+1}' for i in range(6)]
        missing_cols = [col for col in required_cols if col not in df_balls_only.columns]
        if missing_cols:
            logger.error(f"Colunas necessárias ausentes: {missing_cols}")
            return None, None
            
        # Seleciona apenas as colunas das bolas para processamento
        balls_df = df_balls_only[required_cols].copy()
            
        # Verifica valores válidos
        invalid_rows = balls_df[~balls_df.apply(lambda x: all(1 <= val <= 60 for val in x), axis=1)]
        if not invalid_rows.empty:
            logger.warning(f"Encontradas {len(invalid_rows)} linhas com valores inválidos")
            balls_df = balls_df[balls_df.apply(lambda x: all(1 <= val <= 60 for val in x), axis=1)]
            
        # Converte para lista e aplica o MultiLabelBinarizer
        draws_list = balls_df.values.tolist()
        mlb = MultiLabelBinarizer(classes=list(range(1, config['num_features'] + 1)))
        encoded_data = mlb.fit_transform(draws_list)
        
        logger.info(f"Dados transformados com sucesso ({encoded_data.shape[1]} features)")
        return encoded_data, mlb
        
    except Exception as e:
        logger.error(f"Erro durante o pré-processamento: {e}")
        return None, None

def create_sequences(data, sequence_length):
    """
    Cria sequências de dados para o modelo LSTM.
    X: Sequências de 'sequence_length' sorteios.
    y: O sorteio seguinte a cada sequência.
    """
    logger.info(f"Criando sequências de tamanho {sequence_length}...")
    try:
        if data is None or len(data) == 0:
            logger.error("Dados vazios recebidos para criação de sequências")
            return np.array([]), np.array([])
            
        if len(data) <= sequence_length:
            logger.error(f"Dados insuficientes ({len(data)}) para sequência de tamanho {sequence_length}")
            return np.array([]), np.array([])
            
        # Cria as sequências de forma vetorizada para melhor performance
        X = np.array([data[i:i + sequence_length] for i in range(len(data) - sequence_length)])
        y = np.array([data[i + sequence_length] for i in range(len(data) - sequence_length)])
        
        logger.info(f"{len(X)} sequências criadas com sucesso")
        return X, y
        
    except Exception as e:
        logger.error(f"Erro ao criar sequências: {e}")
        return np.array([]), np.array([])

def build_lstm_model(sequence_length, num_features, lstm_units=128, dropout_rate=0.3):
    """ Constrói o modelo LSTM com arquitetura otimizada. """
    logger.info("Construindo o modelo LSTM...")
    try:
        model = Sequential(name="Modelo_LSTM_MegaSena")
        
        # Camada de entrada
        model.add(Input(shape=(sequence_length, num_features)))
        
        # Primeira camada LSTM com mais unidades
        model.add(LSTM(lstm_units, return_sequences=True, 
                      kernel_initializer='he_normal',
                      recurrent_initializer='orthogonal'))
        model.add(Dropout(dropout_rate))
        
        # Segunda camada LSTM com menos unidades
        model.add(LSTM(lstm_units // 2,
                      kernel_initializer='he_normal',
                      recurrent_initializer='orthogonal'))
        model.add(Dropout(dropout_rate))
        
        # Camada densa intermediária
        model.add(Dense(lstm_units // 4, activation='relu'))
        model.add(Dropout(dropout_rate))
        
        # Camada de saída
        model.add(Dense(num_features, activation='sigmoid'))
        
        # Compilação com otimizador Adam e learning rate personalizado
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['binary_accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        model.summary(print_fn=logger.info)
        return model
        
    except Exception as e:
        logger.error(f"Erro ao construir o modelo: {e}")
        return None

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """ Treina o modelo LSTM com callbacks otimizados. """
    logger.info("Iniciando o treinamento do modelo...")
    try:
        # Callbacks otimizados
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,
            min_lr=0.0001,
            verbose=1
        )
        
        # Callback para logging do treinamento
        class TrainingLogger(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logger.info(f"Época {epoch + 1}/{epochs} - Loss: {logs['loss']:.4f} - Val Loss: {logs['val_loss']:.4f}")
        
        # Treinamento com callbacks
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr, TrainingLogger()],
            verbose=1
        )
        
        logger.info("Treinamento concluído com sucesso")
        return history
        
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {e}")
        return None

def evaluate_real_hits(model, X_test, y_test, batch_size=32):
    """
    Avalia quantos números o modelo acertou entre os 6 mais prováveis em cada sorteio.
    """
    logger.info("\nAvaliando acertos reais nas previsões...")
    try:
        if model is None or X_test is None or y_test is None:
            logger.error("Dados inválidos para avaliação de acertos reais")
            return None
            
        # Faz as previsões
        y_pred_probs = model.predict(X_test, batch_size=batch_size)
        
        # Lista para armazenar os acertos de cada sorteio
        hits_per_draw = []
        detailed_hits = []
        
        # Analisa cada sorteio
        for i in range(len(y_pred_probs)):
            # Obtém os 6 números mais prováveis
            top6_pred_indices = np.argsort(y_pred_probs[i])[-6:]
            predicted_numbers = sorted((top6_pred_indices + 1).tolist())
            
            # Obtém os números que realmente foram sorteados
            actual_winning_indices = np.where(y_test[i] == 1)[0]
            actual_numbers = sorted((actual_winning_indices + 1).tolist())
            
            # Calcula a interseção (números acertados)
            hits = set(predicted_numbers) & set(actual_numbers)
            num_hits = len(hits)
            
            # Armazena os detalhes
            detailed_hits.append({
                'sorteio': i + 1,
                'previstos': predicted_numbers,
                'sorteados': actual_numbers,
                'acertos': sorted(list(hits)),
                'num_acertos': num_hits
            })
            
            hits_per_draw.append(num_hits)
        
        # Calcula estatísticas
        avg_hits = np.mean(hits_per_draw)
        max_hits = np.max(hits_per_draw)
        hits_distribution = {i: hits_per_draw.count(i) for i in range(max_hits + 1)}
        
        # Log detalhado dos resultados
        logger.info("-" * 50)
        logger.info("ANÁLISE DE ACERTOS REAIS")
        logger.info("-" * 50)
        logger.info(f"Total de sorteios analisados: {len(hits_per_draw)}")
        logger.info(f"Média de acertos por sorteio: {avg_hits:.3f}")
        logger.info(f"Máximo de acertos em um sorteio: {max_hits}")
        
        logger.info("\nDistribuição de acertos:")
        for hits, count in hits_distribution.items():
            percentage = (count/len(hits_per_draw))*100
            logger.info(f"Sorteios com {hits} acerto(s): {count} ({percentage:.1f}%)")
        
        # Log dos últimos 5 sorteios como exemplo
        logger.info("\nExemplo dos últimos 5 sorteios:")
        for hit in detailed_hits[-5:]:
            logger.info(f"\nSorteio {hit['sorteio']}:")
            logger.info(f"Números previstos: {hit['previstos']}")
            logger.info(f"Números sorteados: {hit['sorteados']}")
            logger.info(f"Números acertados: {hit['acertos']} ({hit['num_acertos']} acertos)")
        
        logger.info("-" * 50)
        logger.info("AVISO: Acertos passados não garantem acertos futuros.")
        logger.info("-" * 50)
        
        return {
            'hits_per_draw': hits_per_draw,
            'avg_hits': avg_hits,
            'max_hits': max_hits,
            'hits_distribution': hits_distribution,
            'detailed_hits': detailed_hits
        }
        
    except Exception as e:
        logger.error(f"Erro ao avaliar acertos reais: {e}")
        return None

def evaluate_model(model, X_test, y_test, batch_size=32):
    """ Avalia o modelo no conjunto de teste com métricas expandidas. """
    logger.info("\nAvaliando o modelo no conjunto de teste...")
    try:
        if model is None or X_test is None or y_test is None:
            logger.error("Dados inválidos para avaliação")
            return None
            
        # Avaliação básica
        results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        
        # Avaliação de acertos reais
        real_hits_results = evaluate_real_hits(model, X_test, y_test, batch_size)
        
        if real_hits_results is None:
            logger.error("Falha na avaliação de acertos reais")
            return None
            
        # Log das métricas
        logger.info("-" * 50)
        logger.info("Métricas de Avaliação:")
        logger.info(f"Loss no Teste: {results[0]:.4f}")
        logger.info(f"Binary Accuracy no Teste: {results[1]:.4f}")
        if len(results) > 2:
            logger.info(f"AUC no Teste: {results[2]:.4f}")
        
        logger.info("\nEstatísticas de Acertos Reais:")
        logger.info(f"Média de acertos por sorteio: {real_hits_results['avg_hits']:.3f}")
        logger.info(f"Máximo de acertos em um sorteio: {real_hits_results['max_hits']}")
        
        logger.info("\nDistribuição de Acertos:")
        for hits, count in real_hits_results['hits_distribution'].items():
            percentage = (count/len(real_hits_results['hits_per_draw']))*100
            logger.info(f"Sorteios com {hits} acerto(s): {count} ({percentage:.1f}%)")
        
        logger.info("-" * 50)
        logger.info("AVISO: Métricas refletem o ajuste aos dados históricos, não previsão real.")
        logger.info("-" * 50)
        
        return {
            'basic_metrics': results,
            'real_hits': real_hits_results
        }
        
    except Exception as e:
        logger.error(f"Erro durante a avaliação: {e}")
        return None

def predict_next_draw(model, last_sequence, mlb, num_predictions=6):
    """ Faz a previsão para o próximo sorteio com análise de confiança. """
    logger.info("\nFazendo a previsão para o PRÓXIMO sorteio...")
    try:
        if model is None or last_sequence is None:
            logger.error("Modelo ou sequência inválidos para previsão")
            return None, None
            
        # Faz a previsão
        last_sequence_batch = np.expand_dims(last_sequence, axis=0)
        predicted_probabilities = model.predict(last_sequence_batch)[0]
        
        # Obtém os números previstos
        predicted_indices = np.argsort(predicted_probabilities)[-num_predictions:]
        predicted_numbers = sorted((predicted_indices + 1).tolist())
        
        # Calcula métricas de confiança
        confidence_scores = predicted_probabilities[predicted_indices]
        avg_confidence = np.mean(confidence_scores)
        max_confidence = np.max(confidence_scores)
        min_confidence = np.min(confidence_scores)
        
        # Log da previsão
        logger.info("-" * 50)
        logger.info(f"Previsão dos {num_predictions} números mais prováveis:")
        logger.info(f"Números: {predicted_numbers}")
        logger.info("\nMétricas de Confiança:")
        logger.info(f"Confiança Média: {avg_confidence:.4f}")
        logger.info(f"Confiança Máxima: {max_confidence:.4f}")
        logger.info(f"Confiança Mínima: {min_confidence:.4f}")
        
        logger.info("\nProbabilidades por número:")
        for num_idx in predicted_indices:
            logger.info(f"Número {num_idx+1}: {predicted_probabilities[num_idx]:.4f}")
        
        logger.info("-" * 50)
        logger.info("AVISO: Esta previsão é um exercício técnico. NÃO HÁ GARANTIA DE ACERTO.")
        logger.info("-" * 50)
        
        return predicted_numbers, predicted_probabilities
        
    except Exception as e:
        logger.error(f"Erro durante a previsão: {e}")
        return None, None

# --- Novas Funções de Visualização ---

def plot_training_history(history):
    """ Plota o histórico de treinamento com métricas expandidas. """
    logger.info("\nGerando gráficos do histórico de treinamento...")
    try:
        if history is None or not hasattr(history, 'history'):
            logger.error("Histórico de treinamento inválido")
            return
            
        plt.figure(figsize=(15, 10))
        
        # Loss
        plt.subplot(2, 2, 1)
        plt.plot(history.history['loss'], label='Treino')
        plt.plot(history.history['val_loss'], label='Validação')
        plt.title('Loss por Época')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy
        plt.subplot(2, 2, 2)
        plt.plot(history.history['binary_accuracy'], label='Treino')
        plt.plot(history.history['val_binary_accuracy'], label='Validação')
        plt.title('Acurácia por Época')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.legend()
        plt.grid(True)
        
        # AUC (se disponível)
        if 'auc' in history.history:
            plt.subplot(2, 2, 3)
            plt.plot(history.history['auc'], label='Treino')
            plt.plot(history.history['val_auc'], label='Validação')
            plt.title('AUC por Época')
            plt.xlabel('Época')
            plt.ylabel('AUC')
            plt.legend()
            plt.grid(True)
            
        # Learning Rate (se disponível)
        plt.subplot(2, 2, 4)
        if 'lr' in history.history:
            plt.plot(history.history['lr'], label='Taxa de Aprendizado')
            plt.title('Taxa de Aprendizado por Época')
            plt.xlabel('Época')
            plt.ylabel('Learning Rate')
        else:
            plt.text(0.5, 0.5, 'Taxa de Aprendizado não disponível', 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Taxa de Aprendizado')
            plt.axis('off')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        logger.info("Gráficos salvos em 'training_history.png'")
        
    except Exception as e:
        logger.error(f"Erro ao gerar gráficos: {e}")

def plot_prediction_analysis(predicted_numbers, predicted_probabilities, last_draws, mlb):
    """ Gera análise visual das previsões. """
    logger.info("\nGerando análise visual das previsões...")
    try:
        if predicted_numbers is None or predicted_probabilities is None:
            logger.error("Dados de previsão inválidos")
            return
            
        plt.figure(figsize=(15, 10))
        
        # Gráfico de barras das probabilidades
        plt.subplot(2, 2, 1)
        plt.bar(range(1, 61), predicted_probabilities)
        plt.title('Probabilidades para Todos os Números')
        plt.xlabel('Número')
        plt.ylabel('Probabilidade')
        plt.grid(True)
        
        # Gráfico de barras dos números previstos
        plt.subplot(2, 2, 2)
        predicted_numbers = np.array(predicted_numbers)
        plt.bar(predicted_numbers, predicted_probabilities[predicted_numbers-1])
        plt.title('Probabilidades dos Números Previstos')
        plt.xlabel('Número')
        plt.ylabel('Probabilidade')
        plt.grid(True)
        
        # Análise de frequência dos últimos sorteios
        plt.subplot(2, 2, 3)
        # Concatena todos os sorteios e converte para array
        last_numbers = np.concatenate([np.array(draw) for draw in last_draws])
        # Filtra apenas números válidos (entre 1 e 60)
        valid_numbers = last_numbers[(last_numbers >= 1) & (last_numbers <= 60)]
        # Calcula a frequência usando value_counts
        number_freq = np.zeros(60)
        for num in valid_numbers:
            number_freq[int(num)-1] += 1
        # Plota apenas para números válidos
        plt.bar(range(1, 61), number_freq)
        plt.title('Frequência nos Últimos Sorteios')
        plt.xlabel('Número')
        plt.ylabel('Frequência')
        plt.grid(True)
        
        # Comparação entre frequência e probabilidade
        plt.subplot(2, 2, 4)
        plt.scatter(number_freq, predicted_probabilities)
        plt.title('Frequência vs Probabilidade')
        plt.xlabel('Frequência nos Últimos Sorteios')
        plt.ylabel('Probabilidade Prevista')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('prediction_analysis.png')
        logger.info("Análise visual salva em 'prediction_analysis.png'")
        
    except Exception as e:
        logger.error(f"Erro ao gerar análise visual: {e}")

def plot_hits_over_time(model, X_test, y_test, mlb):
    """
    Plota o número de acertos (números sorteados que estavam entre os 6 mais prováveis
    previstos pelo modelo) para cada sorteio no conjunto de teste.
    """
    if X_test is None or y_test is None or X_test.shape[0] == 0:
        print("Dados de teste insuficientes para plotar acertos ao longo do tempo.")
        return

    print("\nCalculando acertos no conjunto de teste (histórico)...")
    y_pred_probs_test = model.predict(X_test)
    hits_per_draw = []

    for i in range(len(y_pred_probs_test)):
        pred_probs = y_pred_probs_test[i]
        actual_encoded = y_test[i]

        # Índices dos 6 números com maior probabilidade prevista
        top6_pred_indices = np.argsort(pred_probs)[-6:]

        # Índices dos números que realmente foram sorteados
        actual_winning_indices = np.where(actual_encoded == 1)[0]

        # Calcula a interseção (quantos números previstos estavam corretos)
        num_hits = len(set(top6_pred_indices) & set(actual_winning_indices))
        hits_per_draw.append(num_hits)

    plt.figure(figsize=(15, 6))
    plt.plot(hits_per_draw, marker='o', linestyle='-', markersize=4, label='Nº de Acertos por Sorteio no Teste')
    plt.xlabel("Índice do Sorteio no Conjunto de Teste (Ordem Cronológica)")
    plt.ylabel("Número de Acertos (entre os Top 6 previstos)")
    plt.title("Número de Acertos do Modelo no Conjunto de Teste Histórico")
    plt.yticks(np.arange(0, 7, 1)) # Eixo Y de 0 a 6 acertos
    plt.grid(True)
    plt.legend()

    # Calcula e exibe estatísticas básicas de acertos
    avg_hits = np.mean(hits_per_draw)
    max_hits = np.max(hits_per_draw)
    print(f"\nEstatísticas de Acertos no Conjunto de Teste ({len(X_test)} sorteios):")
    print(f" - Média de acertos por sorteio: {avg_hits:.3f}")
    print(f" - Máximo de acertos em um sorteio: {max_hits}")
    for i in range(max_hits + 1):
        count = hits_per_draw.count(i)
        print(f" - Sorteios com {i} acerto(s): {count} ({count/len(hits_per_draw)*100:.1f}%)")
    print("Lembre-se: Acertos passados não garantem acertos futuros.")


# --- Nova Função de Exportação ---

def export_results(df, predicted_numbers, predicted_probabilities, evaluation_results, config):
    """ Exporta os resultados para Excel com análises expandidas. """
    logger.info("\nExportando resultados para Excel...")
    try:
        if df is None or predicted_numbers is None:
            logger.error("Dados inválidos para exportação")
            return
            
        # Cria um novo DataFrame para as previsões
        predictions_df = pd.DataFrame({
            'Número': range(1, 61),
            'Probabilidade': predicted_probabilities
        })
        predictions_df = predictions_df.sort_values('Probabilidade', ascending=False)
        
        # Adiciona informações sobre os números previstos
        predictions_df['Previsto'] = predictions_df['Número'].isin(predicted_numbers)
        
        # Cria um DataFrame para as métricas de avaliação
        metrics_df = pd.DataFrame({
            'Métrica': ['Loss', 'Binary Accuracy', 'AUC', 'Média de Acertos', 'Máximo de Acertos'],
            'Valor': [
                evaluation_results['basic_metrics'][0],
                evaluation_results['basic_metrics'][1],
                evaluation_results['basic_metrics'][2] if len(evaluation_results['basic_metrics']) > 2 else 'N/A',
                evaluation_results['real_hits']['avg_hits'],
                evaluation_results['real_hits']['max_hits']
            ]
        })
        
        # Cria um DataFrame para a distribuição de acertos
        hits_df = pd.DataFrame({
            'Número de Acertos': list(evaluation_results['real_hits']['hits_distribution'].keys()),
            'Quantidade': list(evaluation_results['real_hits']['hits_distribution'].values()),
            'Porcentagem': [f"{(count/len(evaluation_results['real_hits']['hits_per_draw']))*100:.1f}%" 
                           for count in evaluation_results['real_hits']['hits_distribution'].values()]
        })
        
        # Exporta para Excel
        with pd.ExcelWriter(config['export_file']) as writer:
            df.to_excel(writer, sheet_name='Histórico', index=False)
            predictions_df.to_excel(writer, sheet_name='Previsões', index=False)
            metrics_df.to_excel(writer, sheet_name='Métricas', index=False)
            hits_df.to_excel(writer, sheet_name='Distribuição de Acertos', index=False)
            
        logger.info(f"Resultados exportados para '{config['export_file']}'")
        
    except Exception as e:
        logger.error(f"Erro ao exportar resultados: {e}")

def split_data(X, y, test_size_ratio):
    """
    Divide os dados em conjuntos de treino, validação e teste.
    """
    logger.info("Dividindo os dados em conjuntos de treino, validação e teste...")
    try:
        # Primeiro, divide em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_ratio, shuffle=False
        )
        
        # Depois, divide o treino em treino e validação
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=test_size_ratio, shuffle=False
        )
        
        logger.info(f"Tamanho dos conjuntos:")
        logger.info(f"- Treino: {len(X_train)} amostras")
        logger.info(f"- Validação: {len(X_val)} amostras")
        logger.info(f"- Teste: {len(X_test)} amostras")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except Exception as e:
        logger.error(f"Erro ao dividir os dados: {e}")
        return None, None, None, None, None, None

# --- Fluxo Principal Atualizado ---
def main():
    """ Função principal do programa. """
    try:
        # Configuração inicial
        logger.info("Iniciando o programa de previsão da Mega-Sena...")
        config = load_config()
        
        # Validação da configuração
        if not validate_config(config):
            logger.error("Configuração inválida. Verifique o arquivo config.json")
            return
            
        # 1. Download e preparação dos dados
        logger.info("Baixando e preparando os dados...")
        df_full = download_and_prepare_data(url=config['data_url'], file_path=config['data_file'])
        if df_full is None or len(df_full) == 0:
            logger.error("Não foi possível obter os dados")
            return
            
        # 2. Pré-processamento dos dados
        logger.info("Pré-processando os dados...")
        encoded_data, mlb = preprocess_data(df_full)
        if encoded_data is None or len(encoded_data) == 0:
            logger.error("Falha no pré-processamento dos dados")
            return
            
        # 3. Criação das sequências
        logger.info("Criando sequências para o modelo...")
        X, y = create_sequences(encoded_data, config['sequence_length'])
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            logger.error("Falha na criação das sequências")
            return
            
        # 4. Divisão treino/validação/teste
        logger.info("Dividindo os dados em conjuntos de treino, validação e teste...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config['test_size_ratio'])
        
        # 5. Construção do modelo
        logger.info("Construindo o modelo LSTM...")
        model = build_lstm_model(config['sequence_length'], config['num_features'], config['lstm_units'], config['dropout_rate'])
        if model is None:
            logger.error("Falha na construção do modelo")
            return
            
        # 6. Treinamento do modelo
        logger.info("Iniciando o treinamento do modelo...")
        history = train_model(model, X_train, y_train, X_val, y_val, config['epochs'], config['batch_size'])
        if history is None:
            logger.error("Falha no treinamento do modelo")
            return
            
        # 7. Avaliação do modelo
        logger.info("Avaliando o modelo...")
        evaluation_results = evaluate_model(model, X_test, y_test, config['batch_size'])
        if evaluation_results is None:
            logger.error("Falha na avaliação do modelo")
            return
            
        # 8. Previsão do próximo sorteio
        logger.info("Gerando previsão para o próximo sorteio...")
        last_sequence = X[-1]  # Última sequência disponível
        predicted_numbers, predicted_probabilities = predict_next_draw(model, last_sequence, mlb)
        if predicted_numbers is None or predicted_probabilities is None:
            logger.error("Falha na geração da previsão")
            return
            
        # 9. Visualizações
        logger.info("Gerando visualizações...")
        plot_training_history(history)
        plot_prediction_analysis(predicted_numbers, predicted_probabilities, encoded_data[-config['sequence_length']:], mlb)
        
        # 10. Exportação dos resultados
        logger.info("Exportando resultados...")
        export_results(df_full, predicted_numbers, predicted_probabilities, evaluation_results, config)
        
        logger.info("\nProcesso concluído com sucesso!")
        logger.info("Verifique os arquivos gerados:")
        logger.info(f"- {config['export_file']}: Resultados detalhados em Excel")
        logger.info("- training_history.png: Gráficos do histórico de treinamento")
        logger.info("- prediction_analysis.png: Análise visual das previsões")
        
    except Exception as e:
        logger.error(f"Erro durante a execução do programa: {e}")
        logger.error("Stack trace:", exc_info=True)
        return

def validate_config(config):
    """ Valida as configurações do arquivo config.json. """
    try:
        required_fields = [
            'data_url', 'export_file', 'sequence_length', 'num_features',
            'lstm_units', 'dropout_rate', 'epochs', 'batch_size',
            'test_size_ratio', 'cache_duration_hours', 'cache_dir'
        ]
        
        # Verifica campos obrigatórios
        for field in required_fields:
            if field not in config:
                logger.error(f"Campo obrigatório ausente no config.json: {field}")
                return False
                
        # Valida valores numéricos
        if not isinstance(config['sequence_length'], int) or config['sequence_length'] < 1:
            logger.error("sequence_length deve ser um número inteiro positivo")
            return False
            
        if not isinstance(config['num_features'], int) or config['num_features'] != 60:
            logger.error("num_features deve ser 60 (número de bolas da Mega-Sena)")
            return False
            
        if not isinstance(config['lstm_units'], int) or config['lstm_units'] < 1:
            logger.error("lstm_units deve ser um número inteiro positivo")
            return False
            
        if not isinstance(config['dropout_rate'], (int, float)) or not 0 <= config['dropout_rate'] <= 1:
            logger.error("dropout_rate deve ser um número entre 0 e 1")
            return False
            
        if not isinstance(config['epochs'], int) or config['epochs'] < 1:
            logger.error("epochs deve ser um número inteiro positivo")
            return False
            
        if not isinstance(config['batch_size'], int) or config['batch_size'] < 1:
            logger.error("batch_size deve ser um número inteiro positivo")
            return False
            
        if not isinstance(config['test_size_ratio'], (int, float)) or not 0 < config['test_size_ratio'] < 1:
            logger.error("test_size_ratio deve ser um número entre 0 e 1")
            return False
            
        if not isinstance(config['cache_duration_hours'], (int, float)) or config['cache_duration_hours'] < 0:
            logger.error("cache_duration_hours deve ser um número não negativo")
            return False
            
        # Valida strings
        if not isinstance(config['data_url'], str) or not config['data_url'].startswith(('http://', 'https://')):
            logger.error("data_url deve ser uma URL válida")
            return False
            
        if not isinstance(config['export_file'], str) or not config['export_file'].endswith('.xlsx'):
            logger.error("export_file deve ser um nome de arquivo Excel válido")
            return False
            
        if not isinstance(config['cache_dir'], str):
            logger.error("cache_dir deve ser uma string")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Erro ao validar configurações: {e}")
        return False

if __name__ == "__main__":
    main()