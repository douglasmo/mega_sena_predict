# -*- coding: utf-8 -*-
"""
Script de Exemplo para "Previsão" da Mega-Sena usando LSTM - Versão Aprimorada.
MODIFIED: Added time features, switched to GRU, adjusted scaling.
"""

import pandas as pd
import numpy as np
# ### MODIFICATION START ###
# Added MinMaxScaler and StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler, StandardScaler
# ### MODIFICATION END ###
# Use TensorFlow/Keras for the Rede Neural LSTM
import tensorflow as tf
# ### MODIFICATION START ###
# Added GRU layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GRU # Added GRU
# ### MODIFICATION END ###
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
        logging.FileHandler('mega_sena_v2.log'), # Changed log file name
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ignorar warnings de performance do TensorFlow (opcional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Configuração via Arquivo ---
def load_config(config_file='configv2.json'):
    """Carrega configurações de um arquivo JSON."""
    default_config = {
        "data_url": "https://loteriascaixa-api.herokuapp.com/api/megasena",
        "data_file": None,
        "export_file": "historico_e_previsoes_megasena_v2.xlsx", # Changed export file name
        "sequence_length": 10,
        "num_features_base": 60, # Renamed from num_features
        "num_features_time": 60, # Added for time features
        "gru_units": 128, # Changed from lstm_units
        "dropout_rate": 0.35, # Slightly increased dropout
        "epochs": 100,
        "batch_size": 32,
        "test_size_ratio": 0.15, # Used for both test and validation splits
        "validation_split_ratio": 0.15, # Ratio of *training* data to use for validation
        "cache_duration_hours": 24,
        "cache_dir": "cache"
    }

    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_loaded = json.load(f)
                # Update default config with loaded values, handling potential renames/new keys
                for key, value in config_loaded.items():
                    # Handle potential old key name
                    if key == "lstm_units" and "gru_units" not in config_loaded:
                        default_config["gru_units"] = value
                    elif key == "num_features" and "num_features_base" not in config_loaded:
                         default_config["num_features_base"] = value
                    elif key in default_config:
                        default_config[key] = value
                    else:
                        logger.warning(f"Ignoring unknown key '{key}' from {config_file}")

                logger.info(f"Configurações carregadas e mescladas de {config_file}")
        else:
            logger.warning(f"Arquivo de configuração {config_file} não encontrado. Usando configurações padrão.")
    except Exception as e:
        logger.error(f"Erro ao carregar configurações: {e}")

    # Calculate total features after loading/setting defaults
    default_config['num_features_total'] = default_config['num_features_base'] + default_config['num_features_time']

    return default_config

# Carrega configurações
config = load_config()

# --- Sistema de Cache ---
# (Cache functions remain the same - get_cache_key, is_cache_valid, save_to_cache, load_from_cache)
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
    (Code remains largely the same as provided, ensures 'BolaX' columns are present and numeric)
    """
    logger.info("Iniciando carregamento de dados...")
    df = None
    data = None # Initialize data

    # --- Cache and Download Logic (same as before) ---
    if url:
        cache_key = get_cache_key(url)
        cache_file = os.path.join(config['cache_dir'], f"{cache_key}.json")

        if is_cache_valid(cache_file, config['cache_duration_hours']):
            logger.info("Carregando dados do cache...")
            data = load_from_cache(cache_file)
            if data:
                logger.info("Dados carregados com sucesso do cache.")
            else:
                logger.warning("Cache inválido ou corrompido. Baixando dados novamente.")
                # Force download if cache load failed
                data = None # Reset data to ensure download attempt

        # Download if cache was invalid, missing, or failed to load
        if data is None:
             logger.info("Cache expirado, não encontrado ou inválido. Baixando dados...")
             try:
                 # Consider adding retries or more robust error handling here
                 response = requests.get(url, verify=False, timeout=60) # Increased timeout
                 response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
                 data = response.json()
                 save_to_cache(data, cache_file)
                 logger.info("Dados baixados e salvos no cache com sucesso.")
             except requests.exceptions.Timeout:
                 logger.error(f"Timeout ao baixar dados de {url}")
                 return None
             except requests.exceptions.HTTPError as http_err:
                 logger.error(f"Erro HTTP ao baixar dados: {http_err}")
                 # Attempt to load from local file as fallback if specified
                 if file_path and os.path.exists(file_path):
                     logger.info(f"Tentando carregar do arquivo local {file_path} como fallback...")
                     # Proceed to file loading logic below
                 else:
                     return None
             except requests.exceptions.RequestException as e:
                 logger.error(f"Erro de rede/conexão ao baixar dados: {e}")
                 # Attempt to load from local file as fallback if specified
                 if file_path and os.path.exists(file_path):
                     logger.info(f"Tentando carregar do arquivo local {file_path} como fallback...")
                     # Proceed to file loading logic below
                 else:
                     return None
             except json.JSONDecodeError as json_err:
                 logger.error(f"Erro ao decodificar JSON da resposta: {json_err}")
                 # Log part of the response text for debugging if possible
                 try:
                     logger.error(f"Resposta recebida (início): {response.text[:500]}...")
                 except Exception:
                     pass # Ignore if response text itself is problematic
                 return None


        # --- JSON Processing Logic (Improved Error Handling) ---
        if isinstance(data, list) and data:
            results = []
            concursos = []
            datas = []
            required_keys = {'dezenas', 'concurso', 'data'} # Check for essential keys

            for i, sorteio in enumerate(data):
                if not isinstance(sorteio, dict):
                    logger.warning(f"Item {i} nos dados não é um dicionário, pulando: {sorteio}")
                    continue

                if not required_keys.issubset(sorteio.keys()):
                     missing = required_keys - sorteio.keys()
                     logger.warning(f"Sorteio {sorteio.get('concurso', i)} com chaves ausentes ({missing}), pulando.")
                     continue

                try:
                    # Attempt to convert dezenas carefully
                    dezenas_str = sorteio.get('dezenas', [])
                    if not isinstance(dezenas_str, list):
                         logger.warning(f"Dezenas no sorteio {sorteio.get('concurso')} não é uma lista, pulando.")
                         continue

                    dezenas = sorted([int(d) for d in dezenas_str]) # Convert inside list comprehension

                    if len(dezenas) == 6 and all(1 <= d <= 60 for d in dezenas):
                        results.append(dezenas)
                        concursos.append(sorteio.get('concurso')) # Already checked existence
                        datas.append(sorteio.get('data'))       # Already checked existence
                    else:
                        logger.warning(f"Sorteio {sorteio.get('concurso')} inválido encontrado (número de dezenas ou valor fora do range): {sorteio}")

                except (ValueError, TypeError) as e:
                    logger.warning(f"Erro ao processar dezenas/data/concurso no sorteio {sorteio.get('concurso', i)}: {e} - Sorteio: {sorteio}")
                    continue # Skip this draw

            if not results:
                logger.error("Nenhum sorteio válido encontrado nos dados baixados/processados.")
                # Attempt local file load if download failed or yielded no results
                df = None # Explicitly set df to None to trigger file load attempt
            else:
                df = pd.DataFrame(results, columns=[f'Bola{i+1}' for i in range(6)])
                if concursos: df['Concurso'] = concursos
                if datas:
                    try:
                        df['Data'] = pd.to_datetime(datas, format='%d/%m/%Y', errors='coerce')
                        if df['Data'].isnull().any():
                             logger.warning("Algumas datas não puderam ser convertidas e foram definidas como NaT.")
                             # Optionally drop rows with NaT dates: df.dropna(subset=['Data'], inplace=True)
                    except Exception as e_date:
                        logger.error(f"Erro ao converter coluna 'Data': {e_date}")
                        # Decide how to handle - drop column, proceed without it?
                        # For now, keep potentially problematic column for inspection
                        pass # Keep the column even if conversion failed partially

                # Ordena pelo concurso (mais antigo primeiro)
                sort_col = None
                if 'Concurso' in df.columns and pd.api.types.is_numeric_dtype(df['Concurso']):
                    sort_col = 'Concurso'
                elif 'Data' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Data']):
                    sort_col = 'Data'

                if sort_col:
                    df = df.sort_values(by=sort_col).reset_index(drop=True)
                    logger.info(f"Dados ordenados por '{sort_col}'.")
                else:
                    logger.warning("Não foi possível ordenar os dados (sem coluna 'Concurso' numérica ou 'Data' datetime).")


                logger.info(f"Dados processados com sucesso da API/Cache ({len(df)} sorteios).")

        elif data is not None: # If data was loaded but not in expected list format
            logger.error("Formato de dados JSON da API/Cache não reconhecido ou inesperado (esperava uma lista de sorteios).")
            df = None # Trigger local file load attempt

    # --- Local File Loading Logic (Fallback) ---
    if df is None and file_path and os.path.exists(file_path):
        logger.info(f"Tentando carregar do arquivo local: {file_path}")
        try:
            # Try common separators
            df_loaded = None
            for sep in [';', ',', '\t', '|']: # Add more if needed
                 try:
                     df_try = pd.read_csv(file_path, sep=sep)
                     # Basic check: does it have at least 6 columns?
                     if df_try.shape[1] >= 6:
                         logger.info(f"Arquivo CSV lido com sucesso usando separador '{sep}'.")
                         df_loaded = df_try
                         break # Stop trying separators
                 except Exception:
                     logger.debug(f"Falha ao ler CSV com separador '{sep}'.") # Debug level
                     continue # Try next separator

            if df_loaded is None:
                 # Last attempt with automatic detection
                 try:
                     logger.info("Tentando detecção automática de separador...")
                     df_loaded = pd.read_csv(file_path, sep=None, engine='python')
                     if df_loaded.shape[1] < 6:
                          logger.warning(f"Detecção automática resultou em poucas colunas ({df_loaded.shape[1]}). Verifique o arquivo.")
                          df_loaded = None # Mark as failed
                     else:
                          logger.info("Detecção automática de separador parece ter funcionado.")
                 except Exception as e_auto:
                     logger.error(f"Falha na detecção automática de separador: {e_auto}")
                     df_loaded = None # Mark as failed

            if df_loaded is not None:
                df = df_loaded # Assign successfully loaded df
                logger.info(f"Dados carregados de {file_path}")
            else:
                logger.error(f"Não foi possível ler o arquivo CSV {file_path} com separadores comuns ou detecção automática.")
                return None

        except Exception as e_file:
            logger.error(f"Erro crítico ao tentar carregar arquivo local {file_path}: {e_file}")
            return None

    # --- Final DataFrame Check and Column Processing ---
    if df is None:
        logger.error("Nenhuma fonte de dados (URL, Cache ou Arquivo Local) funcionou ou forneceu dados válidos.")
        return None

    # --- Column Identification and Renaming (Improved Robustness) ---
    bola_cols_found = []
    # Expand potential names (case-insensitive is harder here without iterating all columns)
    potential_patterns = [
        [f'Bola{i+1}' for i in range(6)],
        [f'bola{i+1}' for i in range(6)],
        [f'Dezena{i+1}' for i in range(6)],
        [f'dezena{i+1}' for i in range(6)],
        [f'N{i+1}' for i in range(6)],
         [f'n{i+1}' for i in range(6)]
    ]

    df_cols_lower = {c.lower(): c for c in df.columns} # Map lower case to original case

    # Try exact patterns first
    for pattern_list in potential_patterns:
        if all(col in df.columns for col in pattern_list):
            bola_cols_found = pattern_list
            logger.info(f"Colunas de bolas encontradas usando o padrão: {pattern_list}")
            break
        # Try lower case match
        elif all(col.lower() in df_cols_lower for col in pattern_list):
             bola_cols_found = [df_cols_lower[col.lower()] for col in pattern_list]
             logger.info(f"Colunas de bolas encontradas (case-insensitive) como: {bola_cols_found}")
             break

    # Heuristic identification if no pattern matched
    if not bola_cols_found:
        logger.warning("Nenhum padrão de nome de coluna conhecido encontrado. Tentando heurística...")
        numeric_cols = df.select_dtypes(include=np.number).columns
        potential_bola_cols = []
        for c in numeric_cols:
             # Check if column looks like lottery numbers (integer-like, within range, few NaNs)
             try:
                 # Attempt conversion to numeric, coercing errors, then check range and NaNs
                 numeric_col = pd.to_numeric(df[c], errors='coerce')
                 is_likely_bola = numeric_col.between(1, 60, inclusive='both').all() and \
                                  numeric_col.notna().all() and \
                                  (numeric_col.fillna(-1) == numeric_col.fillna(-1).astype(int)).all() # Check if integer-like

                 if is_likely_bola:
                     potential_bola_cols.append(c)
             except Exception as e_heur:
                 logger.warning(f"Erro ao avaliar coluna '{c}' para heurística: {e_heur}")


        if len(potential_bola_cols) >= 6:
            # Be cautious - take the first 6, but warn the user
            bola_cols_found = potential_bola_cols[:6]
            logger.warning(f"Colunas de bolas identificadas heuristicamente como: {bola_cols_found}. VERIFIQUE SE ESTÃO CORRETAS!")
        else:
            logger.error(f"Erro: Não foi possível identificar 6 colunas numéricas válidas (1-60, sem NaNs).")
            logger.error(f"Colunas numéricas encontradas: {list(numeric_cols)}")
            logger.error(f"Colunas totais: {list(df.columns)}")
            return None # Cannot proceed without ball columns

    # Rename to standard 'BolaX' and ensure numeric type
    rename_map = {found_col: f'Bola{i+1}' for i, found_col in enumerate(bola_cols_found)}
    df.rename(columns=rename_map, inplace=True)
    bola_cols = [f'Bola{i+1}' for i in range(6)]

    try:
        for col in bola_cols:
            # Convert to numeric first, coercing errors, then fill potential NaNs before int conversion
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                logger.warning(f"Valores não numéricos encontrados na coluna '{col}' foram convertidos para NaN.")
                # Decide on handling NaNs: fill with a placeholder (e.g., 0 or median) or drop rows
                # Filling with 0 might be problematic if 0 is not a valid number. Dropping is safer.
                logger.warning(f"Removendo {df[col].isnull().sum()} linhas com valores inválidos em '{col}'.")
                df.dropna(subset=[col], inplace=True)
            df[col] = df[col].astype(int) # Convert to int after handling NaNs

        logger.info("Colunas das bolas verificadas e convertidas para inteiro.")
    except Exception as e_num:
        logger.error(f"Erro ao converter colunas de bolas para numérico/inteiro: {e_num}")
        return None

    # Select and return relevant columns
    cols_to_keep = bola_cols
    if 'Concurso' in df.columns: cols_to_keep.append('Concurso')
    if 'Data' in df.columns: cols_to_keep.append('Data')

    # Use columns present in the dataframe, avoid KeyError if 'Concurso' or 'Data' wasn't found/kept
    cols_to_keep = [col for col in cols_to_keep if col in df.columns]
    final_df = df[cols_to_keep].copy()

    # Final sort check
    sort_col = None
    if 'Concurso' in final_df.columns and pd.api.types.is_numeric_dtype(final_df['Concurso']):
        sort_col = 'Concurso'
    elif 'Data' in final_df.columns and pd.api.types.is_datetime64_any_dtype(final_df['Data']):
        sort_col = 'Data'

    if sort_col:
        final_df = final_df.sort_values(by=sort_col).reset_index(drop=True)
    else:
         # If sorting failed earlier, attempt index sort as last resort if index looks sequential
         if final_df.index.is_monotonic_increasing:
             logger.info("Dados parecem estar em ordem pelo índice original.")
         else:
             logger.warning("Não foi possível garantir a ordem cronológica final dos dados.")


    logger.info(f"Processamento final: {len(final_df)} sorteios carregados e formatados.")
    if len(final_df) < config['sequence_length'] * 3: # Need enough for train/val/test sequences
        logger.error(f"Dados insuficientes ({len(final_df)} sorteios) para criar sequências e divisões. Mínimo recomendado: ~{config['sequence_length']*3}")
        return None

    return final_df


# ### MODIFICATION START ###
def add_time_features(df_balls_only, num_features_base):
    """
    Calcula o número de sorteios desde a última vez que cada número apareceu.
    Retorna um array NumPy não normalizado.
    """
    logger.info("Calculando features de tempo (sorteios desde a última aparição)...")
    try:
        bola_cols = [f'Bola{i+1}' for i in range(6)]
        draws = df_balls_only[bola_cols].values
        num_draws = len(draws)
        time_features_list = []
        # Initialize last_seen_draw with -1 to indicate number hasn't been seen
        last_seen_draw = {num: -1 for num in range(1, num_features_base + 1)}
        # Max interval will be the current draw index + 1 if a number hasn't appeared
        max_possible_interval = num_draws

        for i in range(num_draws):
            current_features = np.zeros(num_features_base)
            numbers_in_current_draw = set(draws[i])

            for num in range(1, num_features_base + 1):
                if last_seen_draw[num] == -1:
                    # If number hasn't been seen yet, interval is effectively max possible up to this point
                    interval = i + 1 # Or consider using max_possible_interval
                else:
                    interval = i - last_seen_draw[num]
                current_features[num - 1] = interval

            time_features_list.append(current_features)

            # Update last_seen_draw *after* calculating features for the current draw
            for drawn_num in numbers_in_current_draw:
                if 1 <= drawn_num <= num_features_base: # Ensure valid number
                    last_seen_draw[drawn_num] = i

        time_features_raw = np.array(time_features_list)
        logger.info(f"Features de tempo calculadas. Shape: {time_features_raw.shape}")
        return time_features_raw

    except Exception as e:
        logger.error(f"Erro ao calcular features de tempo: {e}")
        return None
# ### MODIFICATION END ###


def preprocess_data_labels(df_balls_only, num_features_base): # Renamed function
    """
    Transforma os números sorteados (DataFrame apenas com colunas BolaX)
    em formato MultiLabelBinarizer (One-Hot Encoding para múltiplas labels).
    AGORA SÓ PROCESSA OS LABELS.
    """
    logger.info("Iniciando pré-processamento dos labels (MultiLabelBinarizer)...")
    try:
        # Validação dos dados
        if df_balls_only.empty:
            logger.error("DataFrame vazio recebido para pré-processamento de labels")
            return None, None

        # Verifica se todas as colunas necessárias existem
        required_cols = [f'Bola{i+1}' for i in range(6)]
        missing_cols = [col for col in required_cols if col not in df_balls_only.columns]
        if missing_cols:
            logger.error(f"Colunas necessárias ausentes para labels: {missing_cols}")
            return None, None

        # Seleciona apenas as colunas das bolas para processamento
        balls_df = df_balls_only[required_cols].copy()

        # Verifica valores válidos
        invalid_rows_mask = ~balls_df.apply(lambda x: all(1 <= val <= num_features_base for val in x if pd.notna(val)), axis=1)
        if invalid_rows_mask.any():
            logger.warning(f"Encontradas {invalid_rows_mask.sum()} linhas com valores inválidos (fora de [1, {num_features_base}]) ou NaNs nas colunas de bolas. Essas linhas serão removidas.")
            balls_df = balls_df[~invalid_rows_mask]
            if balls_df.empty:
                 logger.error("Nenhuma linha válida restante após remover valores inválidos.")
                 return None, None


        # Converte para lista e aplica o MultiLabelBinarizer
        draws_list = balls_df.values.tolist()
        mlb = MultiLabelBinarizer(classes=list(range(1, num_features_base + 1)))
        encoded_data = mlb.fit_transform(draws_list)

        logger.info(f"Labels transformados com sucesso ({encoded_data.shape[0]} amostras, {encoded_data.shape[1]} features base)")
        # Return encoded_data AND the indices of the valid rows from the original df
        valid_indices = df_balls_only.index[~invalid_rows_mask] # Get original indices
        return encoded_data, mlb, valid_indices

    except Exception as e:
        logger.error(f"Erro durante o pré-processamento dos labels: {e}", exc_info=True)
        return None, None, None

# ### MODIFICATION START ###
def create_sequences(encoded_labels, time_features_scaled, sequence_length):
    """
    Cria sequências de dados para o modelo GRU.
    X: Sequências combinadas de labels codificados e features de tempo escaladas.
    y: O label codificado do sorteio seguinte a cada sequência.
    """
    logger.info(f"Criando sequências combinadas de tamanho {sequence_length}...")
    try:
        if encoded_labels is None or time_features_scaled is None or len(encoded_labels) == 0:
            logger.error("Dados de labels ou features de tempo vazios recebidos para criação de sequências")
            return np.array([]), np.array([])

        if len(encoded_labels) != len(time_features_scaled):
             logger.error(f"Inconsistência no número de amostras entre labels ({len(encoded_labels)}) e features de tempo ({len(time_features_scaled)})")
             return np.array([]), np.array([])

        if len(encoded_labels) <= sequence_length:
            logger.error(f"Dados insuficientes ({len(encoded_labels)} amostras) para sequência de tamanho {sequence_length}")
            return np.array([]), np.array([])

        num_samples = len(encoded_labels) - sequence_length
        num_features_base = encoded_labels.shape[1]
        num_features_time = time_features_scaled.shape[1]
        num_features_total = num_features_base + num_features_time

        # Pre-allocate arrays for efficiency
        X = np.zeros((num_samples, sequence_length, num_features_total), dtype=np.float32)
        y = np.zeros((num_samples, num_features_base), dtype=encoded_labels.dtype) # y contains only the base labels

        for i in range(num_samples):
            # Get sequences for labels and time features
            seq_labels = encoded_labels[i : i + sequence_length]
            seq_time = time_features_scaled[i : i + sequence_length]

            # Combine features along the last axis
            X[i] = np.concatenate((seq_labels, seq_time), axis=-1)

            # Get the target label (the draw after the sequence)
            y[i] = encoded_labels[i + sequence_length]

        logger.info(f"{len(X)} sequências combinadas criadas com sucesso. Shape X: {X.shape}, Shape y: {y.shape}")
        return X, y

    except Exception as e:
        logger.error(f"Erro ao criar sequências combinadas: {e}", exc_info=True)
        return np.array([]), np.array([])

def build_model(sequence_length, num_features_total, num_features_base, gru_units, dropout_rate): # Modified parameters
    """ Constrói o modelo GRU com arquitetura otimizada. """
    logger.info("Construindo o modelo GRU...")
    try:
        model = Sequential(name="Modelo_GRU_MegaSena")

        # Camada de entrada com o número total de features
        model.add(Input(shape=(sequence_length, num_features_total))) # Use total features

        # Primeira camada GRU
        model.add(GRU(gru_units, return_sequences=True, # Changed LSTM to GRU
                      kernel_initializer='he_normal',
                      recurrent_initializer='orthogonal'))
        model.add(Dropout(dropout_rate))

        # Segunda camada GRU
        model.add(GRU(gru_units // 2, # Changed LSTM to GRU
                      kernel_initializer='he_normal',
                      recurrent_initializer='orthogonal'))
        model.add(Dropout(dropout_rate))

        # Camada densa intermediária
        model.add(Dense(gru_units // 4, activation='relu'))
        model.add(Dropout(dropout_rate))

        # Camada de saída - PREDICTS ONLY THE BASE FEATURES (LABELS)
        model.add(Dense(num_features_base, activation='sigmoid')) # Output size is num_features_base

        # Compilação com otimizador Adam e learning rate personalizado
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Consider AdamW or tuning LR
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy', # Appropriate for multi-label
            metrics=['binary_accuracy', tf.keras.metrics.AUC(name='auc')]
        )

        model.summary(print_fn=logger.info)
        return model

    except Exception as e:
        logger.error(f"Erro ao construir o modelo GRU: {e}")
        return None
# ### MODIFICATION END ###

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """ Treina o modelo GRU com callbacks otimizados. """ # Changed LSTM to GRU
    logger.info("Iniciando o treinamento do modelo GRU...")
    try:
        # Callbacks otimizados (patience might need adjustment)
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20, # Slightly increased patience
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=8, # Slightly increased patience
            min_lr=0.00005, # Lower min_lr
            verbose=1
        )

        # Callback para logging do treinamento
        class TrainingLogger(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                # Log more metrics if available
                log_str = f"Época {epoch + 1}/{self.params['epochs']} - Loss: {logs.get('loss', -1):.4f}"
                if 'binary_accuracy' in logs: log_str += f" - Acc: {logs.get('binary_accuracy', -1):.4f}"
                if 'auc' in logs: log_str += f" - AUC: {logs.get('auc', -1):.4f}"
                log_str += f" - Val Loss: {logs.get('val_loss', -1):.4f}"
                if 'val_binary_accuracy' in logs: log_str += f" - Val Acc: {logs.get('val_binary_accuracy', -1):.4f}"
                if 'val_auc' in logs: log_str += f" - Val AUC: {logs.get('val_auc', -1):.4f}"
                if 'lr' in logs: log_str += f" - LR: {logs.get('lr', -1):.6f}"
                logger.info(log_str)


        # Treinamento com callbacks
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr, TrainingLogger()],
            verbose=0 # Set to 0 to rely on custom logger callback
        )

        logger.info("Treinamento concluído com sucesso")
        return history

    except Exception as e:
        logger.error(f"Erro durante o treinamento: {e}")
        return None

# --- Evaluation Functions (evaluate_real_hits, evaluate_model) ---
# These functions remain the same conceptually, as they operate on the model's
# predictions (y_pred_probs) and the true labels (y_test), which are still
# the multi-hot encoded base features.
def evaluate_real_hits(model, X_test, y_test, batch_size=32):
    """
    Avalia quantos números o modelo acertou entre os 6 mais prováveis em cada sorteio.
    (No changes needed here - operates on prediction probabilities and true labels)
    """
    logger.info("\nAvaliando acertos reais nas previsões...")
    try:
        if model is None or X_test is None or y_test is None or len(X_test) == 0:
             logger.error("Dados inválidos ou vazios para avaliação de acertos reais (X_test, y_test)")
             return None

        # Faz as previsões
        y_pred_probs = model.predict(X_test, batch_size=batch_size)

        # Check prediction shape consistency
        if y_pred_probs.shape != y_test.shape:
             logger.error(f"Inconsistência de shape entre previsões ({y_pred_probs.shape}) e teste ({y_test.shape})")
             # Attempt to reshape if only batch dimension differs and makes sense
             if y_pred_probs.shape[1:] == y_test.shape[1:] and len(y_pred_probs) == len(X_test):
                 logger.warning("Shapes de previsão e teste diferem apenas na dimensão do lote, continuando...")
             else:
                 return None # Incompatible shapes

        # Lista para armazenar os acertos de cada sorteio
        hits_per_draw = []
        detailed_hits = [] # To store more details

        # Analisa cada sorteio
        for i in range(len(y_pred_probs)):
            # Obtém os 6 números mais prováveis (índices de 0 a 59)
            # Argsort gives indices of smallest first, so take the last 6
            top6_pred_indices = np.argsort(y_pred_probs[i])[-6:]
            # Convert indices (0-59) to numbers (1-60)
            predicted_numbers = sorted((top6_pred_indices + 1).tolist())

            # Obtém os números que realmente foram sorteados (índices onde y_test[i] == 1)
            actual_winning_indices = np.where(y_test[i] == 1)[0]
             # Convert indices (0-59) to numbers (1-60)
            actual_numbers = sorted((actual_winning_indices + 1).tolist())

            # Calcula a interseção (números acertados)
            hits = set(predicted_numbers) & set(actual_numbers)
            num_hits = len(hits)

            # Armazena os detalhes
            detailed_hits.append({
                'sorteio_index_teste': i, # Index within the test set
                'previstos': predicted_numbers,
                'sorteados': actual_numbers,
                'acertos': sorted(list(hits)),
                'num_acertos': num_hits
            })

            hits_per_draw.append(num_hits)

        # Calcula estatísticas
        if not hits_per_draw: # Handle case where test set might be empty after all
             logger.warning("Nenhum sorteio no conjunto de teste para calcular estatísticas de acertos.")
             avg_hits = 0
             max_hits = 0
             hits_distribution = {}
        else:
             avg_hits = np.mean(hits_per_draw)
             max_hits = np.max(hits_per_draw)
             # Ensure distribution covers 0 to max_hits even if some counts are 0
             hits_distribution = {i: hits_per_draw.count(i) for i in range(max_hits + 1)}


        # Log detalhado dos resultados
        logger.info("-" * 60)
        logger.info("ANÁLISE DE ACERTOS REAIS (TOP 6 PREVISTOS vs SORTEADOS)")
        logger.info("-" * 60)
        total_sorteios = len(hits_per_draw)
        logger.info(f"Total de sorteios no conjunto de teste analisados: {total_sorteios}")
        if total_sorteios > 0:
             logger.info(f"Média de acertos por sorteio: {avg_hits:.3f}")
             logger.info(f"Máximo de acertos em um sorteio: {max_hits}")

             logger.info("\nDistribuição de acertos:")
             for hits_count, count in hits_distribution.items():
                 if count > 0: # Only show counts > 0 for brevity
                     percentage = (count / total_sorteios) * 100
                     logger.info(f"Sorteios com {hits_count} acerto(s): {count} ({percentage:.1f}%)")

             # Log dos últimos 5 sorteios como exemplo (if available)
             logger.info(f"\nExemplo dos últimos {min(5, total_sorteios)} sorteios do conjunto de teste:")
             for hit_detail in detailed_hits[-min(5, total_sorteios):]:
                 logger.info(f"\nSorteio Teste Índice {hit_detail['sorteio_index_teste']}:")
                 logger.info(f"  Números previstos (Top 6): {hit_detail['previstos']}")
                 logger.info(f"  Números sorteados reais:   {hit_detail['sorteados']}")
                 logger.info(f"  Números acertados:         {hit_detail['acertos']} ({hit_detail['num_acertos']} acertos)")
        else:
             logger.info("Nenhum sorteio disponível para mostrar distribuição ou exemplos.")

        logger.info("-" * 60)
        logger.info("AVISO IMPORTANTE: Acertos passados observados neste conjunto de teste histórico")
        logger.info("NÃO SÃO GARANTIA NENHUMA de acertos futuros. Loterias são fundamentalmente aleatórias.")
        logger.info("-" * 60)

        return {
            'hits_per_draw': hits_per_draw,
            'avg_hits': avg_hits,
            'max_hits': max_hits,
            'hits_distribution': hits_distribution,
            'detailed_hits': detailed_hits # Return detailed info as well
        }

    except Exception as e:
        logger.error(f"Erro ao avaliar acertos reais: {e}", exc_info=True)
        return None

def evaluate_model(model, X_test, y_test, batch_size=32):
    """ Avalia o modelo no conjunto de teste com métricas expandidas. """
    logger.info("\nAvaliando o modelo final no conjunto de teste...")
    try:
        if model is None or X_test is None or y_test is None or len(X_test) == 0:
            logger.error("Dados inválidos ou vazios para avaliação (model, X_test, y_test)")
            return None

        # Avaliação básica (Loss, Accuracy, AUC)
        logger.info("Calculando métricas básicas (Loss, Accuracy, AUC)...")
        results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        metric_names = model.metrics_names
        basic_metrics_dict = dict(zip(metric_names, results))

        # Avaliação de acertos reais (Top 6)
        logger.info("Calculando acertos reais (Top 6)...")
        real_hits_results = evaluate_real_hits(model, X_test, y_test, batch_size)

        if real_hits_results is None:
            logger.error("Falha na avaliação de acertos reais durante a avaliação final.")
            # Return only basic metrics if hits failed
            return {'basic_metrics': basic_metrics_dict, 'real_hits': None}


        # Log consolidado das métricas
        logger.info("-" * 60)
        logger.info("Resultados da Avaliação no Conjunto de Teste")
        logger.info("-" * 60)
        logger.info("Métricas Padrão:")
        for name, value in basic_metrics_dict.items():
             logger.info(f"  - {name}: {value:.4f}")


        logger.info("\nEstatísticas de Acertos Reais (Top 6):")
        if real_hits_results: # Check if results were obtained
             logger.info(f"  - Média de acertos por sorteio: {real_hits_results['avg_hits']:.3f}")
             logger.info(f"  - Máximo de acertos em um sorteio: {real_hits_results['max_hits']}")

             logger.info("\n  Distribuição de Acertos:")
             total_test_draws = len(real_hits_results.get('hits_per_draw', []))
             if total_test_draws > 0:
                 hits_dist = real_hits_results.get('hits_distribution', {})
                 for hits_count, count in hits_dist.items():
                      if count > 0:
                          percentage = (count / total_test_draws) * 100
                          logger.info(f"    * Sorteios com {hits_count} acerto(s): {count} ({percentage:.1f}%)")
             else:
                 logger.info("    N/A (sem sorteios no teste)")
        else:
             logger.info("  N/A (falha no cálculo)")


        logger.info("-" * 60)
        logger.info("Lembrete: Métricas refletem o desempenho no passado (conjunto de teste).")
        logger.info("O desempenho futuro em sorteios reais é incerto e provavelmente baixo.")
        logger.info("-" * 60)

        return {
            'basic_metrics': basic_metrics_dict, # Return dict for clarity
            'real_hits': real_hits_results
        }

    except Exception as e:
        logger.error(f"Erro durante a avaliação final do modelo: {e}", exc_info=True)
        return None


# ### MODIFICATION START ###
def predict_next_draw(model, last_sequence_combined, mlb, num_predictions=6):
    """
    Faz a previsão para o próximo sorteio usando a última sequência COMBINADA.
    """
    logger.info("\nFazendo a previsão para o PRÓXIMO sorteio...")
    try:
        if model is None:
             logger.error("Modelo inválido para previsão.")
             return None, None
        if last_sequence_combined is None or last_sequence_combined.ndim != 2:
             logger.error(f"Última sequência combinada inválida para previsão. Esperado 2D (seq_len, features), recebido shape: {getattr(last_sequence_combined, 'shape', 'N/A')}")
             return None, None

        # O input já deve ser a sequência combinada (labels + time features scaled)
        # Adiciona a dimensão do batch
        last_sequence_batch = np.expand_dims(last_sequence_combined, axis=0)
        logger.info(f"Shape da sequência de entrada para previsão: {last_sequence_batch.shape}") # Log shape

        # Faz a previsão - o modelo retorna probabilidades para os LABELS BASE
        predicted_probabilities = model.predict(last_sequence_batch)[0] # Index [0] to remove batch dim

        # Verifica o shape da saída
        expected_output_shape = config['num_features_base']
        if predicted_probabilities.shape[0] != expected_output_shape:
            logger.error(f"Shape inesperado da saída da previsão: {predicted_probabilities.shape}. Esperado: ({expected_output_shape},)")
            return None, None

        # Obtém os NÚMEROS previstos (índices + 1)
        # Indices dos N mais prováveis (0 a num_features_base-1)
        predicted_indices = np.argsort(predicted_probabilities)[-num_predictions:]
        # Converte para números (1 a num_features_base)
        predicted_numbers = sorted((predicted_indices + 1).tolist())

        # Calcula métricas de confiança (baseadas nas probabilidades previstas)
        confidence_scores = predicted_probabilities[predicted_indices]
        # Handle potential empty confidence_scores if num_predictions=0 or error
        if confidence_scores.size > 0:
             avg_confidence = np.mean(confidence_scores)
             max_confidence = np.max(confidence_scores)
             min_confidence = np.min(confidence_scores)
        else:
             avg_confidence = max_confidence = min_confidence = 0.0


        # Log da previsão
        logger.info("-" * 50)
        logger.info(f"Previsão dos {num_predictions} números mais prováveis:")
        logger.info(f"Números: {predicted_numbers}")
        logger.info("\nMétricas de Confiança (Probabilidade):")
        logger.info(f"Confiança Média: {avg_confidence:.4f}")
        logger.info(f"Confiança Máxima: {max_confidence:.4f}")
        logger.info(f"Confiança Mínima: {min_confidence:.4f}")

        logger.info("\nProbabilidades individuais para os números previstos:")
        # Sort indices by probability descending for logging
        sorted_pred_indices = predicted_indices[np.argsort(confidence_scores)[::-1]]
        for num_idx in sorted_pred_indices:
            logger.info(f"  - Número {num_idx + 1}: {predicted_probabilities[num_idx]:.4f}")

        logger.info("-" * 50)
        logger.info("AVISO CRÍTICO: Esta previsão é um exercício técnico baseado em dados")
        logger.info("históricos e um modelo estatístico. NÃO HÁ QUALQUER GARANTIA DE ACERTO.")
        logger.info("A Mega-Sena é um jogo de azar. Jogue com responsabilidade.")
        logger.info("-" * 50)

        # Retorna os números previstos e o array COMPLETO de probabilidades (0-59)
        return predicted_numbers, predicted_probabilities

    except Exception as e:
        logger.error(f"Erro durante a previsão do próximo sorteio: {e}", exc_info=True)
        return None, None
# ### MODIFICATION END ###

# --- Novas Funções de Visualização ---
# plot_training_history remains the same conceptually
def plot_training_history(history, filename='training_history_v2.png'):
    """ Plota o histórico de treinamento com métricas expandidas. """
    logger.info(f"\nGerando gráficos do histórico de treinamento em {filename}...")
    try:
        if history is None or not hasattr(history, 'history') or not history.history:
            logger.error("Histórico de treinamento inválido ou vazio.")
            return

        plt.figure(figsize=(15, 10))
        history_dict = history.history

        # Determine available metrics
        metrics = [m for m in ['loss', 'binary_accuracy', 'auc', 'lr'] if m in history_dict]
        num_plots = len(metrics)
        num_cols = 2
        num_rows = (num_plots + 1) // num_cols # Calculate rows needed

        plot_index = 1
        for metric in metrics:
            if metric == 'lr': continue # Plot LR separately if needed, or skip
            plt.subplot(num_rows, num_cols, plot_index)
            plt.plot(history_dict[metric], label=f'Treino {metric.capitalize()}')
            if f'val_{metric}' in history_dict:
                plt.plot(history_dict[f'val_{metric}'], label=f'Validação {metric.capitalize()}')
            plt.title(f'{metric.replace("_", " ").capitalize()} por Época')
            plt.xlabel('Época')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            plot_index += 1

        # Plot Learning Rate if available
        if 'lr' in history_dict:
             plt.subplot(num_rows, num_cols, plot_index)
             plt.plot(history_dict['lr'], label='Taxa de Aprendizado')
             plt.title('Taxa de Aprendizado por Época')
             plt.xlabel('Época')
             plt.ylabel('Learning Rate')
             plt.legend()
             plt.grid(True)
             plot_index +=1


        # Adjust layout if odd number of plots
        if plot_index <= num_rows * num_cols:
             # If there's an empty subplot slot, potentially remove its axis
             try:
                 # This might fail if the calculated layout is exact, hence the try-except
                 fig = plt.gcf()
                 ax_to_remove = fig.add_subplot(num_rows, num_cols, plot_index)
                 ax_to_remove.axis('off')
             except ValueError:
                 pass # Subplot configuration was likely exact


        plt.tight_layout()
        plt.savefig(filename)
        logger.info(f"Gráficos salvos em '{filename}'")
        plt.close() # Close the figure to free memory

    except Exception as e:
        logger.error(f"Erro ao gerar gráficos de treinamento: {e}", exc_info=True)


# ### MODIFICATION START ###
# plot_prediction_analysis needs access to the raw data (or frequency) for the last N draws
def plot_prediction_analysis(predicted_numbers, predicted_probabilities, df_full, sequence_length, filename='prediction_analysis_v2.png'):
    """ Gera análise visual das previsões, incluindo frequência recente. """
    logger.info(f"\nGerando análise visual das previsões em {filename}...")
    try:
        if predicted_numbers is None or predicted_probabilities is None:
            logger.error("Dados de previsão inválidos para análise visual.")
            return
        if df_full is None or df_full.empty:
             logger.error("DataFrame histórico inválido para análise de frequência.")
             return

        plt.figure(figsize=(15, 12)) # Increased size slightly

        # 1. Gráfico de barras das probabilidades (Todos os Números)
        plt.subplot(2, 2, 1)
        all_numbers = np.arange(1, config['num_features_base'] + 1)
        plt.bar(all_numbers, predicted_probabilities, width=0.8)
        plt.title(f'Probabilidades Previstas para Todos os {config["num_features_base"]} Números')
        plt.xlabel('Número')
        plt.ylabel('Probabilidade Prevista')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.xticks(np.arange(0, 61, 5)) # Adjust x-ticks for readability
        plt.xlim(0.5, 60.5)

        # 2. Gráfico de barras das probabilidades (Apenas Números Previstos)
        plt.subplot(2, 2, 2)
        predicted_numbers_arr = np.array(predicted_numbers)
        # Get probabilities corresponding to predicted numbers (indices are number-1)
        probs_for_predicted = predicted_probabilities[predicted_numbers_arr - 1]
        bars = plt.bar(predicted_numbers_arr, probs_for_predicted, width=0.6, color='red') # Highlight predicted
        plt.title(f'Probabilidades dos {len(predicted_numbers)} Números Previstos')
        plt.xlabel('Número Previsto')
        plt.ylabel('Probabilidade Prevista')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        # Add probability values on top of bars
        plt.bar_label(bars, fmt='%.4f', padding=3)
        # Set y-limit based on max probability for better visualization
        if probs_for_predicted.size > 0:
             plt.ylim(0, max(probs_for_predicted) * 1.15) # Add 15% margin
        plt.xticks(predicted_numbers_arr) # Show only predicted numbers on x-axis


        # 3. Análise de frequência dos últimos 'sequence_length' sorteios
        plt.subplot(2, 2, 3)
        # Get the actual numbers from the last 'sequence_length' draws from the original df
        last_n_draws_df = df_full.iloc[-sequence_length:]
        bola_cols = [f'Bola{i+1}' for i in range(6)]
        # Concatenate all numbers from these draws
        last_numbers_flat = pd.concat([last_n_draws_df[col] for col in bola_cols]).dropna().astype(int).values
        # Calculate frequency
        number_freq = np.zeros(config['num_features_base'])
        unique_nums, counts = np.unique(last_numbers_flat, return_counts=True)
        # Populate frequency array, ensuring numbers are within valid range
        valid_mask = (unique_nums >= 1) & (unique_nums <= config['num_features_base'])
        valid_unique_nums = unique_nums[valid_mask]
        valid_counts = counts[valid_mask]
        # Use number - 1 as index
        number_freq[valid_unique_nums - 1] = valid_counts

        plt.bar(all_numbers, number_freq, width=0.8)
        plt.title(f'Frequência nos Últimos {sequence_length} Sorteios Históricos')
        plt.xlabel('Número')
        plt.ylabel('Frequência de Aparição')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.xticks(np.arange(0, 61, 5))
        plt.xlim(0.5, 60.5)
        if number_freq.max() > 0:
             plt.yticks(np.arange(0, number_freq.max() + 1, 1)) # Integer ticks for frequency


        # 4. Comparação entre frequência recente e probabilidade prevista (Scatter Plot)
        plt.subplot(2, 2, 4)
        plt.scatter(number_freq, predicted_probabilities, alpha=0.6)
        # Highlight the predicted numbers
        freq_for_predicted = number_freq[predicted_numbers_arr - 1]
        plt.scatter(freq_for_predicted, probs_for_predicted, color='red', s=80, label='Previstos', edgecolors='black')
        plt.title('Frequência Recente vs Probabilidade Prevista')
        plt.xlabel(f'Frequência nos Últimos {sequence_length} Sorteios')
        plt.ylabel('Probabilidade Prevista')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
         # Add labels to predicted points
        for i, num in enumerate(predicted_numbers_arr):
            plt.text(freq_for_predicted[i] + 0.05 * number_freq.max(), # Offset x slightly
                     probs_for_predicted[i], # y position
                     str(num), fontsize=9)


        plt.tight_layout()
        plt.savefig(filename)
        logger.info(f"Análise visual salva em '{filename}'")
        plt.close()

    except Exception as e:
        logger.error(f"Erro ao gerar análise visual das previsões: {e}", exc_info=True)
# ### MODIFICATION END ###

# plot_hits_over_time remains the same conceptually
def plot_hits_over_time(model, X_test, y_test, mlb, filename='hits_over_time_v2.png'):
    """
    Plota o número de acertos (números sorteados que estavam entre os 6 mais prováveis
    previstos pelo modelo) para cada sorteio no conjunto de teste.
    (No significant changes needed here, operates on model output and test labels)
    """
    logger.info(f"\nGerando gráfico de acertos ao longo do tempo no conjunto de teste em {filename}...")
    if X_test is None or y_test is None or X_test.shape[0] == 0:
        logger.warning("Dados de teste insuficientes para plotar acertos ao longo do tempo.")
        return None # Return None to indicate plot wasn't generated

    logger.info("Calculando acertos no conjunto de teste (histórico) para plotagem...")
    try:
        y_pred_probs_test = model.predict(X_test)
        hits_per_draw = []

        for i in range(len(y_pred_probs_test)):
            pred_probs = y_pred_probs_test[i]
            actual_encoded = y_test[i]

            # Índices dos 6 números com maior probabilidade prevista (0-59)
            top6_pred_indices = np.argsort(pred_probs)[-6:]

            # Índices dos números que realmente foram sorteados (0-59)
            actual_winning_indices = np.where(actual_encoded == 1)[0]

            # Calcula a interseção (quantos números previstos estavam corretos)
            num_hits = len(set(top6_pred_indices) & set(actual_winning_indices))
            hits_per_draw.append(num_hits)

        if not hits_per_draw:
             logger.warning("Nenhum resultado de acerto calculado para plotagem.")
             return None

        plt.figure(figsize=(15, 6))
        plt.plot(hits_per_draw, marker='o', linestyle='-', markersize=4, label='Nº de Acertos (Top 6) por Sorteio no Teste')
        # Add a rolling average line
        if len(hits_per_draw) >= 10: # Only plot if enough data points
             rolling_avg = pd.Series(hits_per_draw).rolling(window=10, min_periods=1).mean()
             plt.plot(rolling_avg, linestyle='--', color='red', label='Média Móvel (10 sorteios)')

        plt.xlabel("Índice do Sorteio no Conjunto de Teste (Ordem Cronológica)")
        plt.ylabel("Número de Acertos (entre os Top 6 previstos)")
        plt.title("Número de Acertos do Modelo no Conjunto de Teste Histórico")
        plt.yticks(np.arange(0, 7, 1)) # Eixo Y de 0 a 6 acertos
        plt.ylim(bottom=-0.2) # Add small bottom margin
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.savefig(filename)
        logger.info(f"Gráfico de acertos ao longo do tempo salvo em '{filename}'")
        plt.close()

        # Return the hits data for potential further use
        return hits_per_draw

    except Exception as e:
        logger.error(f"Erro ao gerar gráfico de acertos ao longo do tempo: {e}", exc_info=True)
        return None


# --- Nova Função de Exportação ---
# export_results remains the same conceptually, but ensure it uses the correct keys from evaluation_results
def export_results(df, predicted_numbers, predicted_probabilities, evaluation_results, config):
    """ Exporta os resultados para Excel com análises expandidas. """
    logger.info(f"\nExportando resultados para Excel ({config['export_file']})...")
    try:
        if df is None:
            logger.error("DataFrame histórico inválido para exportação.")
            return
        if predicted_numbers is None or predicted_probabilities is None:
            logger.error("Dados de previsão inválidos para exportação.")
            return
        if evaluation_results is None:
            logger.warning("Resultados da avaliação inválidos, exportando sem eles.")
            # Initialize empty structures to avoid errors below
            evaluation_results = {'basic_metrics': {}, 'real_hits': None}


        # Cria um novo DataFrame para as previsões (todos os 60 números)
        predictions_df = pd.DataFrame({
            'Número': range(1, config['num_features_base'] + 1),
            'Probabilidade_Prevista': predicted_probabilities
        })
        # Adiciona coluna indicando se o número está nos Top 6 previstos
        predictions_df['Previsto_Top_6'] = predictions_df['Número'].isin(predicted_numbers)
        predictions_df = predictions_df.sort_values('Probabilidade_Prevista', ascending=False).reset_index(drop=True)


        # Cria um DataFrame para as métricas de avaliação básica
        basic_metrics_dict = evaluation_results.get('basic_metrics', {})
        if not basic_metrics_dict:
             logger.warning("Métricas básicas de avaliação não encontradas para exportação.")
             metrics_df = pd.DataFrame({'Métrica': ['N/A'], 'Valor': ['N/A']})
        else:
             metrics_df = pd.DataFrame({
                 'Métrica': list(basic_metrics_dict.keys()),
                 'Valor': [f"{v:.5f}" for v in basic_metrics_dict.values()] # Format values
             })


        # Cria um DataFrame para a análise de acertos reais (se disponível)
        real_hits_results = evaluation_results.get('real_hits')
        if real_hits_results and real_hits_results.get('hits_distribution') is not None:
            hits_dist = real_hits_results['hits_distribution']
            total_draws = len(real_hits_results.get('hits_per_draw', []))

            hits_summary_df = pd.DataFrame({
                 'Estatística': ['Média de Acertos (Top 6)', 'Máximo de Acertos (Top 6)', 'Total Sorteios Teste'],
                 'Valor': [
                      f"{real_hits_results.get('avg_hits', 'N/A'):.3f}",
                      f"{real_hits_results.get('max_hits', 'N/A')}",
                      total_draws
                 ]
            })

            hits_dist_df = pd.DataFrame({
                'Número de Acertos': list(hits_dist.keys()),
                'Quantidade Sorteios': list(hits_dist.values()),
            })
            # Calculate percentage only if total_draws > 0
            if total_draws > 0:
                 hits_dist_df['Porcentagem (%)'] = hits_dist_df['Quantidade Sorteios'].apply(
                     lambda count: f"{(count / total_draws) * 100:.1f}%"
                 )
            else:
                 hits_dist_df['Porcentagem (%)'] = 'N/A'

            hits_dist_df = hits_dist_df.sort_values('Número de Acertos').reset_index(drop=True)

            # Include detailed hits if available
            detailed_hits = real_hits_results.get('detailed_hits')
            if detailed_hits:
                 detailed_hits_df = pd.DataFrame(detailed_hits)
                 # Convert lists to strings for Excel compatibility
                 for col in ['previstos', 'sorteados', 'acertos']:
                     if col in detailed_hits_df.columns:
                         detailed_hits_df[col] = detailed_hits_df[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
            else:
                 detailed_hits_df = pd.DataFrame({'Info': ['Detalhes de acertos não disponíveis']})


        else:
            logger.warning("Resultados de acertos reais não encontrados ou incompletos para exportação.")
            hits_summary_df = pd.DataFrame({'Estatística': ['N/A'], 'Valor': ['N/A']})
            hits_dist_df = pd.DataFrame({'Número de Acertos': ['N/A'], 'Quantidade Sorteios': ['N/A'], 'Porcentagem (%)': ['N/A']})
            detailed_hits_df = pd.DataFrame({'Info': ['Detalhes de acertos não disponíveis']})


        # Exporta para Excel
        logger.info("Escrevendo abas no arquivo Excel...")
        with pd.ExcelWriter(config['export_file'], engine='openpyxl') as writer:
            # Aba 1: Previsão Detalhada (Todos os Números)
            predictions_df.to_excel(writer, sheet_name='Previsao_Probabilidades', index=False)
            logger.debug("Aba 'Previsao_Probabilidades' escrita.")

            # Aba 2: Métricas de Avaliação
            metrics_df.to_excel(writer, sheet_name='Metricas_Avaliacao', index=False)
            logger.debug("Aba 'Metricas_Avaliacao' escrita.")

            # Aba 3: Sumário de Acertos Reais
            hits_summary_df.to_excel(writer, sheet_name='Sumario_Acertos_Reais', index=False)
            logger.debug("Aba 'Sumario_Acertos_Reais' escrita.")

            # Aba 4: Distribuição de Acertos Reais
            hits_dist_df.to_excel(writer, sheet_name='Distribuicao_Acertos_Reais', index=False)
            logger.debug("Aba 'Distribuicao_Acertos_Reais' escrita.")

            # Aba 5: Detalhes dos Acertos por Sorteio (Teste)
            if detailed_hits_df is not None:
                 detailed_hits_df.to_excel(writer, sheet_name='Detalhes_Acertos_Teste', index=False)
                 logger.debug("Aba 'Detalhes_Acertos_Teste' escrita.")

            # Aba 6: Histórico Original
            df.to_excel(writer, sheet_name='Historico_Completo', index=False)
            logger.debug("Aba 'Historico_Completo' escrita.")

            # Aba 7: Configuração Usada
            config_df = pd.DataFrame(list(config.items()), columns=['Parametro', 'Valor'])
            config_df.to_excel(writer, sheet_name='Configuracao_Usada', index=False)
            logger.debug("Aba 'Configuracao_Usada' escrita.")


        logger.info(f"Resultados exportados com sucesso para '{config['export_file']}'")

    except PermissionError:
         logger.error(f"Erro de permissão ao tentar escrever o arquivo '{config['export_file']}'. Verifique se ele não está aberto em outro programa.")
    except Exception as e:
        logger.error(f"Erro ao exportar resultados para Excel: {e}", exc_info=True)


# ### MODIFICATION START ###
# Splitting needs to handle labels and raw time features separately before scaling
def split_data(encoded_labels, time_features_raw, test_size_ratio, validation_split_ratio, sequence_length):
    """
    Divide os dados (labels e features raw) em conjuntos de treino, validação e teste
    mantendo a ordem cronológica. Escala as features de tempo APÓS a divisão.
    Cria as sequências para cada conjunto.
    """
    logger.info("Dividindo dados e escalando features de tempo...")
    try:
        if len(encoded_labels) != len(time_features_raw):
             logger.error("Disparidade no tamanho dos labels e features raw antes da divisão.")
             return [None] * 7 # Return 7 Nones

        # 1. Split into initial Train+Validation and Test sets (chronological)
        test_split_index = int(len(encoded_labels) * (1 - test_size_ratio))

        train_val_labels = encoded_labels[:test_split_index]
        test_labels = encoded_labels[test_split_index:]

        train_val_time_raw = time_features_raw[:test_split_index]
        test_time_raw = time_features_raw[test_split_index:]

        # Check if test set is large enough to create at least one sequence
        if len(test_labels) <= sequence_length:
             logger.warning(f"Conjunto de teste ({len(test_labels)} amostras) não é grande o suficiente para criar sequências de tamanho {sequence_length}. Teste será vazio.")
             # Adjust split to make test set empty or handle differently?
             # For now, proceed, create_sequences will handle empty test set later.
             # Alternatively, re-allocate data or raise error. Let's allow empty test for now.
             test_labels = np.array([])
             test_time_raw = np.array([])


        # 2. Split Train+Validation into Train and Validation sets (chronological)
        val_split_index = int(len(train_val_labels) * (1 - validation_split_ratio))

        train_labels = train_val_labels[:val_split_index]
        val_labels = train_val_labels[val_split_index:]

        train_time_raw = train_val_time_raw[:val_split_index]
        val_time_raw = train_val_time_raw[val_split_index:]

         # Check if train/validation sets are large enough
        if len(train_labels) <= sequence_length:
            logger.error(f"Conjunto de treino ({len(train_labels)} amostras) não é grande o suficiente para criar sequências de tamanho {sequence_length}.")
            return [None] * 7
        if len(val_labels) <= sequence_length:
            # Optional: Allow empty validation set? Or error out? Let's warn and proceed with potentially empty val set.
            logger.warning(f"Conjunto de validação ({len(val_labels)} amostras) não é grande o suficiente para criar sequências de tamanho {sequence_length}. Validação será vazia.")
            val_labels = np.array([])
            val_time_raw = np.array([])


        logger.info(f"Tamanhos dos conjuntos (antes de sequenciar):")
        logger.info(f"- Treino Bruto: {len(train_labels)} amostras")
        logger.info(f"- Validação Bruta: {len(val_labels)} amostras")
        logger.info(f"- Teste Bruto: {len(test_labels)} amostras")


        # 3. Scale Time Features (Fit ONLY on Training Data)
        # Using StandardScaler is generally robust for features without fixed bounds
        scaler = StandardScaler()
        logger.info("Ajustando o Scaler (StandardScaler) nas features de tempo do treino...")
        train_time_scaled = scaler.fit_transform(train_time_raw)
        logger.info("Escalando features de tempo de validação e teste...")
        # Handle potentially empty validation/test sets before scaling
        val_time_scaled = scaler.transform(val_time_raw) if len(val_time_raw) > 0 else np.array([])
        test_time_scaled = scaler.transform(test_time_raw) if len(test_time_raw) > 0 else np.array([])


        # 4. Create Sequences for each set
        logger.info("Criando sequências para o conjunto de treino...")
        X_train, y_train = create_sequences(train_labels, train_time_scaled, sequence_length)

        logger.info("Criando sequências para o conjunto de validação...")
        # Handle case where val set was too small
        if len(val_labels) > sequence_length:
             X_val, y_val = create_sequences(val_labels, val_time_scaled, sequence_length)
        else:
             X_val, y_val = np.array([]), np.array([])
             logger.warning("Conjunto de validação final vazio devido ao tamanho insuficiente para sequências.")


        logger.info("Criando sequências para o conjunto de teste...")
         # Handle case where test set was too small
        if len(test_labels) > sequence_length:
            X_test, y_test = create_sequences(test_labels, test_time_scaled, sequence_length)
        else:
            X_test, y_test = np.array([]), np.array([])
            logger.warning("Conjunto de teste final vazio devido ao tamanho insuficiente para sequências.")


        logger.info(f"Tamanho final dos conjuntos de sequências:")
        logger.info(f"- Treino: X={X_train.shape}, y={y_train.shape}" if X_train.size > 0 else "- Treino: Vazio")
        logger.info(f"- Validação: X={X_val.shape}, y={y_val.shape}" if X_val.size > 0 else "- Validação: Vazio")
        logger.info(f"- Teste: X={X_test.shape}, y={y_test.shape}" if X_test.size > 0 else "- Teste: Vazio")


        # Return all sets and the fitted scaler (needed for predicting future draws)
        return X_train, X_val, X_test, y_train, y_val, y_test, scaler

    except Exception as e:
        logger.error(f"Erro ao dividir/escalar dados ou criar sequências finais: {e}", exc_info=True)
        return [None] * 7 # Return 7 Nones

# ### MODIFICATION END ###


# --- Fluxo Principal Atualizado ---
def main():
    """ Função principal do programa. """
    run_start_time = datetime.now()
    logger.info(f"Iniciando o programa de previsão da Mega-Sena (v2) em {run_start_time.strftime('%Y-%m-%d %H:%M:%S')}...")

    try:
        # Configuração inicial
        global config # Use global config dictionary
        config = load_config()

        # Validação da configuração
        if not validate_config(config): # Assuming validate_config exists and works
            logger.error("Configuração inválida. Verifique o arquivo config.json ou os padrões.")
            return # Stop execution if config is invalid

        logger.info(f"Configurações carregadas: {json.dumps(config, indent=2)}")


        # 1. Download e preparação dos dados
        logger.info("Etapa 1: Baixando e preparando os dados...")
        df_full = download_and_prepare_data(url=config['data_url'], file_path=config['data_file'])
        if df_full is None or df_full.empty:
            logger.error("Falha na Etapa 1: Não foi possível obter ou processar os dados históricos.")
            return
        logger.info(f"Dados históricos carregados: {len(df_full)} sorteios.")


        # 2. Pré-processamento dos Labels (MultiLabelBinarizer)
        logger.info("Etapa 2: Pré-processando os labels dos sorteios...")
        # Pass only ball columns to avoid issues if Concurso/Data are missing/problematic
        bola_cols = [f'Bola{i+1}' for i in range(6)]
        encoded_labels, mlb, valid_indices = preprocess_data_labels(df_full[bola_cols], config['num_features_base'])
        if encoded_labels is None or mlb is None or valid_indices is None:
            logger.error("Falha na Etapa 2: Pré-processamento dos labels.")
            return
        # Filter the original DataFrame and time features based on valid indices from label processing
        df_full_valid = df_full.loc[valid_indices].reset_index(drop=True)
        logger.info(f"Labels processados: {len(encoded_labels)} sorteios válidos restantes.")


        # 3. Cálculo das Features de Tempo
        logger.info("Etapa 3: Calculando features de tempo...")
        # Use the valid DataFrame subset
        time_features_raw = add_time_features(df_full_valid[bola_cols], config['num_features_base'])
        if time_features_raw is None or len(time_features_raw) != len(encoded_labels):
            logger.error("Falha na Etapa 3: Cálculo ou alinhamento das features de tempo.")
            return
        logger.info(f"Features de tempo calculadas. Shape: {time_features_raw.shape}")


        # 4. Divisão, Escalonamento e Criação de Sequências
        logger.info("Etapa 4: Dividindo dados, escalando features e criando sequências...")
        X_train, X_val, X_test, y_train, y_val, y_test, time_feature_scaler = split_data(
            encoded_labels,
            time_features_raw,
            config['test_size_ratio'],
            config['validation_split_ratio'],
            config['sequence_length']
        )
        # Check if all splits were successful
        if any(data is None for data in [X_train, X_val, X_test, y_train, y_val, y_test, time_feature_scaler]):
             logger.error("Falha na Etapa 4: Divisão/escalonamento/sequenciamento retornou None.")
             return
        # Check specifically if training data is present
        if X_train.size == 0 or y_train.size == 0:
             logger.error("Falha na Etapa 4: Conjunto de treinamento está vazio após divisão/sequenciamento.")
             return


        # 5. Construção do modelo
        logger.info("Etapa 5: Construindo o modelo GRU...")
        model = build_model(
            config['sequence_length'],
            config['num_features_total'], # Total features (base + time)
            config['num_features_base'], # Base features (for output layer size)
            config['gru_units'],
            config['dropout_rate']
        )
        if model is None:
            logger.error("Falha na Etapa 5: Construção do modelo.")
            return


        # 6. Treinamento do modelo
        logger.info("Etapa 6: Treinando o modelo GRU...")
        # Only train if validation data exists
        validation_data = (X_val, y_val) if X_val.size > 0 and y_val.size > 0 else None
        if validation_data is None:
             logger.warning("Conjunto de validação está vazio. Treinando sem validação em tempo real (EarlyStopping/ReduceLR podem não funcionar como esperado).")

        history = train_model(
            model, X_train, y_train,
            X_val, y_val, # Pass validation data even if empty, Keras handles it
            config['epochs'], config['batch_size']
        )
        if history is None:
            logger.error("Falha na Etapa 6: Treinamento do modelo.")
            # Optionally save the untrained model for inspection?
            return
        logger.info("Treinamento concluído.")


        # 7. Avaliação do modelo
        logger.info("Etapa 7: Avaliando o modelo no conjunto de teste...")
        evaluation_results = None
        if X_test.size > 0 and y_test.size > 0:
             evaluation_results = evaluate_model(model, X_test, y_test, config['batch_size'])
             if evaluation_results is None:
                 logger.warning("Falha na Etapa 7: Avaliação do modelo no conjunto de teste. Continuando sem resultados de avaliação.")
                 # Initialize evaluation_results to avoid errors later
                 evaluation_results = {'basic_metrics': {}, 'real_hits': None}
        else:
             logger.warning("Conjunto de teste vazio. Pulando avaliação final do modelo.")
             evaluation_results = {'basic_metrics': {}, 'real_hits': None} # Set empty results


        # 8. Previsão do próximo sorteio
        logger.info("Etapa 8: Gerando previsão para o próximo sorteio...")
        # Need the last sequence from the *combined, scaled* data used for training/testing
        # Get the last raw sequence of labels and time features
        last_sequence_labels_raw = encoded_labels[-config['sequence_length']:]
        last_sequence_time_raw = time_features_raw[-config['sequence_length']:]

        # Scale the time features using the *fitted* scaler
        last_sequence_time_scaled = time_feature_scaler.transform(last_sequence_time_raw)

        # Combine the scaled features
        last_sequence_combined = np.concatenate(
            (last_sequence_labels_raw, last_sequence_time_scaled),
            axis=-1
        ).astype(np.float32) # Ensure correct dtype

        predicted_numbers, predicted_probabilities = predict_next_draw(
             model, last_sequence_combined, mlb # mlb isn't strictly needed by predict anymore, but kept for consistency
        )
        if predicted_numbers is None or predicted_probabilities is None:
            logger.error("Falha na Etapa 8: Geração da previsão para o próximo sorteio.")
            # Optionally proceed without prediction for export? Or stop? Let's stop.
            return


        # 9. Visualizações
        logger.info("Etapa 9: Gerando visualizações...")
        plot_training_history(history) # Plot training graphs
        plot_prediction_analysis(predicted_numbers, predicted_probabilities, df_full_valid, config['sequence_length']) # Plot prediction analysis
        if X_test.size > 0 and y_test.size > 0:
             plot_hits_over_time(model, X_test, y_test, mlb) # Plot hits if test set exists
        else:
             logger.info("Pulando gráfico de acertos ao longo do tempo (conjunto de teste vazio).")


        # 10. Exportação dos resultados
        logger.info("Etapa 10: Exportando resultados...")
        export_results(df_full_valid, predicted_numbers, predicted_probabilities, evaluation_results, config)


        run_end_time = datetime.now()
        total_duration = run_end_time - run_start_time
        logger.info("-" * 60)
        logger.info(f"Processo concluído com sucesso em: {total_duration}")
        logger.info("-" * 60)
        logger.info("Verifique os arquivos gerados:")
        logger.info(f"- Log: mega_sena_v2.log")
        logger.info(f"- Excel: {config['export_file']}")
        logger.info("- Gráficos PNG: training_history_v2.png, prediction_analysis_v2.png, hits_over_time_v2.png (se aplicável)")
        logger.info("-" * 60)
        logger.info("Lembre-se: Os resultados são experimentais e não garantem acertos.")
        logger.info("-" * 60)


    except Exception as e:
        logger.error(f"Erro GERAL durante a execução do programa: {e}", exc_info=True) # Log traceback for general errors
        return

# Assuming validate_config exists from the original code
def validate_config(config):
    """ Valida as configurações do arquivo config.json. """
    logger.info("Validando configuração...")
    is_valid = True
    try:
        required_fields = [
            'data_url', 'export_file', 'sequence_length', 'num_features_base',
            'num_features_time', 'num_features_total', 'gru_units', 'dropout_rate',
            'epochs', 'batch_size', 'test_size_ratio', 'validation_split_ratio',
             'cache_duration_hours', 'cache_dir'
        ]

        # Check required fields
        for field in required_fields:
            if field not in config:
                logger.error(f"Campo obrigatório ausente na configuração: {field}")
                is_valid = False

        # Early exit if required fields are missing
        if not is_valid: return False

        # Validate types and values
        if not isinstance(config['sequence_length'], int) or config['sequence_length'] < 1:
            logger.error("sequence_length deve ser um inteiro positivo.")
            is_valid = False
        if not isinstance(config['num_features_base'], int) or config['num_features_base'] <= 0:
             logger.error("num_features_base deve ser um inteiro positivo (geralmente 60).")
             is_valid = False
        if not isinstance(config['num_features_time'], int) or config['num_features_time'] < 0:
             logger.error("num_features_time deve ser um inteiro não negativo.")
             is_valid = False
        if config['num_features_total'] != config['num_features_base'] + config['num_features_time']:
             logger.error("num_features_total não corresponde à soma de num_features_base e num_features_time.")
             is_valid = False
        if not isinstance(config['gru_units'], int) or config['gru_units'] < 1:
            logger.error("gru_units deve ser um inteiro positivo.")
            is_valid = False
        if not isinstance(config['dropout_rate'], (int, float)) or not 0 <= config['dropout_rate'] < 1: # Dropout < 1
            logger.error("dropout_rate deve ser um número >= 0 e < 1.")
            is_valid = False
        if not isinstance(config['epochs'], int) or config['epochs'] < 1:
            logger.error("epochs deve ser um inteiro positivo.")
            is_valid = False
        if not isinstance(config['batch_size'], int) or config['batch_size'] < 1:
            logger.error("batch_size deve ser um inteiro positivo.")
            is_valid = False
        if not isinstance(config['test_size_ratio'], (int, float)) or not 0 < config['test_size_ratio'] < 1:
            logger.error("test_size_ratio deve ser um número entre 0 e 1 (exclusivo).")
            is_valid = False
        if not isinstance(config['validation_split_ratio'], (int, float)) or not 0 < config['validation_split_ratio'] < 1:
            logger.error("validation_split_ratio deve ser um número entre 0 e 1 (exclusivo).")
            is_valid = False
        if (config['test_size_ratio'] + config['validation_split_ratio']) >= 1.0:
             logger.error("A soma de test_size_ratio e validation_split_ratio deve ser menor que 1.")
             is_valid = False
        if not isinstance(config['cache_duration_hours'], (int, float)) or config['cache_duration_hours'] < 0:
            logger.error("cache_duration_hours deve ser um número não negativo.")
            is_valid = False
        if not isinstance(config['data_url'], (str, type(None))): # Allow None for local file only
             logger.error("data_url deve ser uma string (URL) ou None.")
             is_valid = False
        if config['data_url'] and not config['data_url'].startswith(('http://', 'https://')):
             logger.error("data_url, se fornecida, deve ser uma URL HTTP/HTTPS válida.")
             is_valid = False
        if not isinstance(config['export_file'], str) or not config['export_file'].endswith('.xlsx'):
            logger.error("export_file deve ser um nome de arquivo .xlsx válido.")
            is_valid = False
        if not isinstance(config['cache_dir'], str):
            logger.error("cache_dir deve ser uma string (nome do diretório).")
            is_valid = False
        # Add check: data_url or data_file must be provided
        if config.get('data_url') is None and config.get('data_file') is None:
             logger.error("É necessário fornecer 'data_url' ou 'data_file' na configuração.")
             is_valid = False


        if is_valid:
             logger.info("Configuração validada com sucesso.")
        else:
             logger.error("Validação da configuração falhou. Verifique os erros acima.")
        return is_valid

    except Exception as e:
        logger.error(f"Erro inesperado durante a validação da configuração: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    main()