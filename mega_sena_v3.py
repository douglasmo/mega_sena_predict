# -*- coding: utf-8 -*-
"""
Script de Exemplo para "Previsão" da Mega-Sena - Versão V3.
MODIFIED: Added Statistical Features (Parity, Sum, Range, Zones, Rolling Freq).
          Uses separate scalers for time and statistical features.
"""

import pandas as pd
import numpy as np
# ### MODIFICATION START V3 ###
# Added joblib for potential scaler saving/loading later if needed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler, StandardScaler
# import joblib # Optional: if you want to save/load scalers
# ### MODIFICATION END V3 ###
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, BatchNormalization # Added BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard # Added TensorBoard
import requests
from io import StringIO
import os
import sys
import warnings
import matplotlib.pyplot as plt
import logging
import json
from datetime import datetime, timedelta
import hashlib
from pathlib import Path

# Importar módulo de otimização de hiperparâmetros
try:
    from hyperparameter_tuning import HyperparameterTuner
    hyperparameter_tuning_available = True
except ImportError:
    hyperparameter_tuning_available = False
    warnings.warn("Módulo de otimização de hiperparâmetros não encontrado. A funcionalidade de teste de hiperparâmetros não estará disponível.")

# Criação da pasta output se não existir
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Configuração de logging com codificação UTF-8
log_file = os.path.join(output_dir, 'mega_sena_v3.log')
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')

# Configurar o StreamHandler para usar a codificação UTF-8 se possível
# ou fallback para ASCII com substituição
class EncodingStreamHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__(stream=sys.stdout)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            try:
                self.stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                # Fallback para ASCII se não puder usar UTF-8
                self.stream.write(msg.encode('ascii', 'replace').decode('ascii') + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s', # Added funcName
    handlers=[
        file_handler,
        EncodingStreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ignorar warnings de performance do TensorFlow (opcional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Configuração via Arquivo ---
def load_config(config_file='configv3.json'):
    """Carrega configurações de um arquivo JSON."""
    # ### MODIFICATION START V3 ###
    # Added statistical features config
    default_config = {
        "data_url": "https://loteriascaixa-api.herokuapp.com/api/megasena",
        "data_file": None,
        "export_file": os.path.join(output_dir, "historico_e_previsoes_megasena_v3.xlsx"), # Path com output_dir
        "sequence_length": 15, # Adjusted sequence length
        "num_features_base": 60, # Base labels (MLBed numbers)
        "num_features_time": 60, # Time since last seen
        "num_features_statistical": 187, # Calculated below (1 odd + 1 sum + 1 range + 4 zones + 3*60 freq)
        "rolling_freq_windows": [10, 50, 100], # Windows for frequency calc
        "gru_units": 192, # Increased GRU units for more features
        "use_batch_norm": True, # Option to use Batch Normalization
        "dropout_rate": 0.4, # Increased dropout
        "epochs": 200, # Increased epochs
        "batch_size": 64, # Increased batch size
        "test_size_ratio": 0.15,
        "validation_split_ratio": 0.15,
        "cache_duration_hours": 24,
        "cache_dir": os.path.join(output_dir, "cache"), # Path com output_dir
        "tensorboard_log_dir": os.path.join(output_dir, "logs/fit/") # Path com output_dir
    }
    # Calculate statistical features count based on windows
    # 1 (odd) + 1 (sum) + 1 (range) + 4 (zones) + len(windows)*60 (freq)
    num_stat_features = 1 + 1 + 1 + 4 + len(default_config["rolling_freq_windows"]) * 60
    default_config["num_features_statistical"] = num_stat_features
    # ### MODIFICATION END V3 ###

    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_loaded = json.load(f)
                # Update default config with loaded values, handling potential renames/new keys
                for key, value in config_loaded.items():
                    if key == "lstm_units" and "gru_units" not in config_loaded:
                        default_config["gru_units"] = value
                    elif key == "num_features" and "num_features_base" not in config_loaded:
                         default_config["num_features_base"] = value
                    # Don't allow overriding calculated stat features count unless explicitly in file
                    elif key == "num_features_statistical" and key in config_loaded:
                         default_config[key] = value
                    elif key != "num_features_statistical" and key in default_config:
                         # Se for um caminho de arquivo, adicionar output_dir se não for absoluto
                         if key in ['export_file', 'cache_dir', 'tensorboard_log_dir']:
                             if not os.path.isabs(value):
                                 value = os.path.join(output_dir, value)
                         default_config[key] = value
                    else:
                        logger.warning(f"Ignoring unknown key '{key}' from {config_file}")

                # Recalculate stat features if windows changed and stat count wasn't explicit
                if "rolling_freq_windows" in config_loaded and "num_features_statistical" not in config_loaded:
                     num_stat_features = 1 + 1 + 1 + 4 + len(default_config["rolling_freq_windows"]) * 60
                     default_config["num_features_statistical"] = num_stat_features
                     logger.info(f"Recalculated num_features_statistical to {num_stat_features} based on loaded rolling_freq_windows.")

                logger.info(f"Configurações carregadas e mescladas de {config_file}")
        else:
            logger.warning(f"Arquivo de configuração {config_file} não encontrado. Usando configurações padrão.")
    except Exception as e:
        logger.error(f"Erro ao carregar configurações: {e}")

    # Calculate total features after loading/setting defaults
    default_config['num_features_total'] = (
        default_config['num_features_base'] +
        default_config['num_features_time'] +
        default_config['num_features_statistical']
    )
    logger.info(f"Total features calculated: {default_config['num_features_total']}")

    # Create TensorBoard log directory if it doesn't exist
    Path(default_config['tensorboard_log_dir']).mkdir(parents=True, exist_ok=True)
    
    # Ensure cache directory exists
    Path(default_config['cache_dir']).mkdir(parents=True, exist_ok=True)

    return default_config

# --- Sistema de Cache ---
# (Cache functions remain the same)
def get_cache_key(url):
    return hashlib.md5(url.encode()).hexdigest()

def is_cache_valid(cache_file, duration_hours):
    if not os.path.exists(cache_file): return False
    cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
    return datetime.now() - cache_time < timedelta(hours=duration_hours)

def save_to_cache(data, cache_file):
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w') as f: json.dump(data, f)
        logger.info(f"Dados salvos no cache: {cache_file}")
    except Exception as e: logger.error(f"Erro ao salvar cache: {e}")

def load_from_cache(cache_file):
    try:
        with open(cache_file, 'r') as f: return json.load(f)
    except Exception as e:
        logger.error(f"Erro ao carregar cache: {e}")
        return None

# --- Funções de Dados ---

# download_and_prepare_data remains the same as V2
def download_and_prepare_data(url=None, file_path=None):
    """Downloads/loads data, ensures 'BolaX' columns are present/numeric."""
    # (Implementation from V2 - robust loading, parsing, cleaning)
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
                data = None

        if data is None:
             logger.info("Cache expirado, não encontrado ou inválido. Baixando dados...")
             try:
                 response = requests.get(url, verify=False, timeout=60)
                 response.raise_for_status()
                 data = response.json()
                 save_to_cache(data, cache_file)
                 logger.info("Dados baixados e salvos no cache com sucesso.")
             except requests.exceptions.Timeout:
                 logger.error(f"Timeout ao baixar dados de {url}")
                 return None
             except requests.exceptions.HTTPError as http_err:
                 logger.error(f"Erro HTTP ao baixar dados: {http_err}")
                 if file_path and os.path.exists(file_path):
                     logger.info(f"Tentando carregar do arquivo local {file_path} como fallback...")
                 else: return None
             except requests.exceptions.RequestException as e:
                 logger.error(f"Erro de rede/conexão ao baixar dados: {e}")
                 if file_path and os.path.exists(file_path):
                     logger.info(f"Tentando carregar do arquivo local {file_path} como fallback...")
                 else: return None
             except json.JSONDecodeError as json_err:
                 logger.error(f"Erro ao decodificar JSON da resposta: {json_err}")
                 try: logger.error(f"Resposta recebida (início): {response.text[:500]}...")
                 except Exception: pass
                 return None

        # --- JSON Processing Logic (Improved Error Handling) ---
        if isinstance(data, list) and data:
            results, concursos, datas = [], [], []
            required_keys = {'dezenas', 'concurso', 'data'}

            for i, sorteio in enumerate(data):
                if not isinstance(sorteio, dict):
                    logger.warning(f"Item {i} nos dados não é um dicionário, pulando: {sorteio}")
                    continue
                if not required_keys.issubset(sorteio.keys()):
                     missing = required_keys - sorteio.keys()
                     logger.warning(f"Sorteio {sorteio.get('concurso', i)} com chaves ausentes ({missing}), pulando.")
                     continue
                try:
                    dezenas_str = sorteio.get('dezenas', [])
                    if not isinstance(dezenas_str, list):
                         logger.warning(f"Dezenas no sorteio {sorteio.get('concurso')} não é uma lista, pulando.")
                         continue
                    dezenas = sorted([int(d) for d in dezenas_str])
                    if len(dezenas) == 6 and all(1 <= d <= 60 for d in dezenas):
                        results.append(dezenas)
                        concursos.append(sorteio.get('concurso'))
                        datas.append(sorteio.get('data'))
                    else:
                        logger.warning(f"Sorteio {sorteio.get('concurso')} inválido (número/valor de dezenas): {sorteio}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Erro ao processar sorteio {sorteio.get('concurso', i)}: {e} - Sorteio: {sorteio}")
                    continue

            if not results:
                logger.error("Nenhum sorteio válido encontrado nos dados da API/Cache.")
                df = None
            else:
                df = pd.DataFrame(results, columns=[f'Bola{i+1}' for i in range(6)])
                if concursos: df['Concurso'] = concursos
                if datas:
                    try:
                        df['Data'] = pd.to_datetime(datas, format='%d/%m/%Y', errors='coerce')
                        if df['Data'].isnull().any():
                             logger.warning("Algumas datas não puderam ser convertidas (NaT).")
                    except Exception as e_date: logger.error(f"Erro ao converter coluna 'Data': {e_date}")
                sort_col = None
                if 'Concurso' in df.columns and pd.api.types.is_numeric_dtype(df['Concurso']): sort_col = 'Concurso'
                elif 'Data' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Data']): sort_col = 'Data'
                if sort_col:
                    df = df.sort_values(by=sort_col).reset_index(drop=True)
                    logger.info(f"Dados ordenados por '{sort_col}'.")
                else: logger.warning("Não foi possível ordenar os dados automaticamente.")
                logger.info(f"Dados processados com sucesso da API/Cache ({len(df)} sorteios).")
        elif data is not None:
            logger.error("Formato de dados JSON da API/Cache não reconhecido.")
            df = None

    # --- Local File Loading Logic (Fallback) ---
    if df is None and file_path and os.path.exists(file_path):
        logger.info(f"Tentando carregar do arquivo local: {file_path}")
        # (Robust CSV reading logic from V2)
        try:
            df_loaded = None
            for sep in [';', ',', '\t', '|']:
                 try:
                     df_try = pd.read_csv(file_path, sep=sep)
                     if df_try.shape[1] >= 6:
                         logger.info(f"CSV lido com sucesso (separador '{sep}').")
                         df_loaded = df_try
                         break
                 except Exception: logger.debug(f"Falha ao ler CSV com sep='{sep}'.")
                 continue
            if df_loaded is None:
                 try:
                     logger.info("Tentando detecção automática de separador CSV...")
                     df_loaded = pd.read_csv(file_path, sep=None, engine='python')
                     if df_loaded.shape[1] < 6:
                          logger.warning(f"Detecção automática resultou em < 6 colunas ({df_loaded.shape[1]}).")
                          df_loaded = None
                     else: logger.info("Detecção automática de separador funcionou.")
                 except Exception as e_auto: logger.error(f"Falha na detecção automática: {e_auto}")
            if df_loaded is not None:
                df = df_loaded
                logger.info(f"Dados carregados de {file_path}")
            else:
                logger.error(f"Não foi possível ler o arquivo CSV {file_path}.")
                return None
        except Exception as e_file:
            logger.error(f"Erro crítico ao carregar arquivo local {file_path}: {e_file}")
            return None

    # --- Final DataFrame Check and Column Processing ---
    if df is None:
        logger.error("Nenhuma fonte de dados (URL, Cache ou Arquivo Local) funcionou.")
        return None

    # --- Column Identification and Renaming (Improved Robustness) ---
    # (Logic from V2 to find 'BolaX' columns)
    bola_cols_found = []
    potential_patterns = [ [f'Bola{i+1}' for i in range(6)], [f'bola{i+1}' for i in range(6)],
                           [f'Dezena{i+1}' for i in range(6)], [f'dezena{i+1}' for i in range(6)],
                           [f'N{i+1}' for i in range(6)], [f'n{i+1}' for i in range(6)] ]
    df_cols_lower = {c.lower(): c for c in df.columns}
    for pattern_list in potential_patterns:
        if all(col in df.columns for col in pattern_list):
            bola_cols_found = pattern_list; logger.info(f"Colunas encontradas: {pattern_list}"); break
        elif all(col.lower() in df_cols_lower for col in pattern_list):
             bola_cols_found = [df_cols_lower[col.lower()] for col in pattern_list]; logger.info(f"Colunas encontradas (case-insensitive): {bola_cols_found}"); break
    if not bola_cols_found:
        logger.warning("Nenhum padrão de coluna conhecido. Tentando heurística...")
        numeric_cols = df.select_dtypes(include=np.number).columns
        potential_bola_cols = []
        for c in numeric_cols:
             try:
                 numeric_col = pd.to_numeric(df[c], errors='coerce')
                 is_likely_bola = numeric_col.between(1, 60, inclusive='both').all() and numeric_col.notna().all() and (numeric_col.fillna(-1) == numeric_col.fillna(-1).astype(int)).all()
                 if is_likely_bola: potential_bola_cols.append(c)
             except Exception as e_heur: logger.warning(f"Erro avaliando coluna '{c}' para heurística: {e_heur}")
        if len(potential_bola_cols) >= 6:
            bola_cols_found = potential_bola_cols[:6]
            logger.warning(f"Colunas identificadas heuristicamente: {bola_cols_found}. VERIFIQUE!")
        else:
            logger.error(f"Erro: Não foi possível identificar 6 colunas numéricas válidas (1-60). Colunas: {list(df.columns)}")
            return None
    rename_map = {found_col: f'Bola{i+1}' for i, found_col in enumerate(bola_cols_found)}
    df.rename(columns=rename_map, inplace=True)
    bola_cols = [f'Bola{i+1}' for i in range(6)]
    try:
        for col in bola_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                logger.warning(f"Removendo {df[col].isnull().sum()} linhas com valores inválidos em '{col}'.")
                df.dropna(subset=[col], inplace=True)
            df[col] = df[col].astype(int)
        logger.info("Colunas das bolas verificadas e convertidas para inteiro.")
    except Exception as e_num:
        logger.error(f"Erro ao converter colunas de bolas: {e_num}")
        return None
    cols_to_keep = bola_cols + [col for col in ['Concurso', 'Data'] if col in df.columns]
    final_df = df[cols_to_keep].copy()
    sort_col = None
    if 'Concurso' in final_df.columns and pd.api.types.is_numeric_dtype(final_df['Concurso']): sort_col = 'Concurso'
    elif 'Data' in final_df.columns and pd.api.types.is_datetime64_any_dtype(final_df['Data']): sort_col = 'Data'
    if sort_col: final_df = final_df.sort_values(by=sort_col).reset_index(drop=True)
    else: logger.warning("Não foi possível garantir a ordem cronológica final.")
    logger.info(f"Processamento final: {len(final_df)} sorteios carregados.")
    if len(final_df) < config['sequence_length'] * 3:
        logger.error(f"Dados insuficientes ({len(final_df)} sorteios) para criar sequências/divisões. Mínimo ~{config['sequence_length']*3}")
        return None
    return final_df

# preprocess_data_labels remains the same as V2
def preprocess_data_labels(df_balls_only, num_features_base):
    """Transforms winning numbers into MultiLabelBinarizer format (labels y)."""
    logger.info("Iniciando pré-processamento dos labels (MultiLabelBinarizer)...")
    try:
        if df_balls_only.empty: logger.error("DataFrame vazio para pré-processar labels"); return None, None, None
        required_cols = [f'Bola{i+1}' for i in range(6)]
        if not all(col in df_balls_only.columns for col in required_cols):
            logger.error(f"Colunas ausentes para labels: {[c for c in required_cols if c not in df_balls_only.columns]}"); return None, None, None
        balls_df = df_balls_only[required_cols].copy()
        invalid_rows_mask = ~balls_df.apply(lambda x: all(1 <= val <= num_features_base for val in x if pd.notna(val)), axis=1)
        if invalid_rows_mask.any():
            logger.warning(f"Removendo {invalid_rows_mask.sum()} linhas com valores inválidos/NaNs nas bolas.")
            balls_df = balls_df[~invalid_rows_mask]
            if balls_df.empty: logger.error("Nenhuma linha válida restante."); return None, None, None
        draws_list = balls_df.values.tolist()
        mlb = MultiLabelBinarizer(classes=list(range(1, num_features_base + 1)))
        encoded_data = mlb.fit_transform(draws_list)
        valid_indices = df_balls_only.index[~invalid_rows_mask] # Original indices kept
        logger.info(f"Labels transformados: {encoded_data.shape[0]} amostras, {encoded_data.shape[1]} features base.")
        return encoded_data, mlb, valid_indices
    except Exception as e:
        logger.error(f"Erro durante pré-processamento dos labels: {e}", exc_info=True)
        return None, None, None

# add_time_features remains the same as V2
def add_time_features(df_balls_only, num_features_base):
    """Calculates 'draws since last seen' for each number."""
    logger.info("Calculando features de tempo (sorteios desde última aparição)...")
    try:
        bola_cols = [f'Bola{i+1}' for i in range(6)]
        draws = df_balls_only[bola_cols].values
        num_draws = len(draws)
        time_features_list = []
        last_seen_draw = {num: -1 for num in range(1, num_features_base + 1)}

        for i in range(num_draws):
            current_features = np.zeros(num_features_base)
            numbers_in_current_draw = set(draws[i])
            for num in range(1, num_features_base + 1):
                interval = i + 1 if last_seen_draw[num] == -1 else i - last_seen_draw[num]
                current_features[num - 1] = interval
            time_features_list.append(current_features)
            for drawn_num in numbers_in_current_draw:
                if 1 <= drawn_num <= num_features_base:
                    last_seen_draw[drawn_num] = i
        time_features_raw = np.array(time_features_list)
        logger.info(f"Features de tempo calculadas. Shape: {time_features_raw.shape}")
        return time_features_raw
    except Exception as e:
        logger.error(f"Erro ao calcular features de tempo: {e}", exc_info=True)
        return None

# ### MODIFICATION START V3 ###
def add_statistical_features(df_balls_only, num_features_base, rolling_windows):
    """
    Calculates statistical features for each draw:
    - Parity (Odd Count)
    - Sum
    - Range (Max - Min)
    - Zone Counts (4 zones)
    - Rolling Frequencies (for specified windows)
    Returns a NumPy array (n_draws, n_stat_features).
    """
    logger.info(f"Calculando features estatísticas (Soma, Range, Zonas, Freq. {rolling_windows})...")
    try:
        bola_cols = [f'Bola{i+1}' for i in range(6)]
        draws = df_balls_only[bola_cols].values
        num_draws = len(draws)

        # Pre-allocate lists for simple stats
        odd_counts = []
        sums = []
        ranges = []
        zone_defs = [(1, 15), (16, 30), (31, 45), (46, 60)]
        zone_counts_list = [] # List of lists/arrays

        # Pre-allocate array for frequency features
        num_freq_features = len(rolling_windows) * num_features_base
        rolling_freq_features = np.zeros((num_draws, num_freq_features), dtype=np.float32)

        # Efficiently get all numbers flattened for frequency calculation later
        all_numbers_flat = pd.concat([df_balls_only[col] for col in bola_cols], ignore_index=True)
        # Create MultiLabelBinarized version for efficient rolling sum
        mlb_freq = MultiLabelBinarizer(classes=list(range(1, num_features_base + 1)))
        encoded_draws_freq = mlb_freq.fit_transform(draws.tolist()) # Shape (num_draws, 60)
        encoded_draws_df = pd.DataFrame(encoded_draws_freq, columns=mlb_freq.classes_)

        # Calculate rolling frequencies efficiently
        logger.info("Calculando frequências rolantes...")
        freq_col_offset = 0
        for window in rolling_windows:
            logger.debug(f"  Janela: {window}")
            # Rolling sum on the one-hot encoded data gives the frequency count
            # min_periods=1 ensures output starts from the first draw
            rolling_sum = encoded_draws_df.rolling(window=window, min_periods=1).sum()
            # Shift by 1 so that the frequency for draw 'i' reflects draws *before* 'i'
            rolling_sum_shifted = rolling_sum.shift(1).fillna(0)

            # Store in the correct slice of the main frequency array
            rolling_freq_features[:, freq_col_offset : freq_col_offset + num_features_base] = rolling_sum_shifted.values
            freq_col_offset += num_features_base
        logger.info("Frequências rolantes calculadas.")

        # Calculate draw-specific stats
        logger.info("Calculando estatísticas por sorteio (Par/Ímpar, Soma, Range, Zonas)...")
        for i in range(num_draws):
            current_numbers = draws[i]
            # Parity
            odd_counts.append(np.sum(current_numbers % 2 != 0))
            # Sum
            sums.append(np.sum(current_numbers))
            # Range
            ranges.append(np.max(current_numbers) - np.min(current_numbers))
            # Zones
            counts_in_zones = []
            for z_min, z_max in zone_defs:
                counts_in_zones.append(np.sum((current_numbers >= z_min) & (current_numbers <= z_max)))
            zone_counts_list.append(counts_in_zones)

        # Combine all statistical features
        odd_counts_arr = np.array(odd_counts).reshape(-1, 1)
        sums_arr = np.array(sums).reshape(-1, 1)
        ranges_arr = np.array(ranges).reshape(-1, 1)
        zone_counts_arr = np.array(zone_counts_list) # Shape (num_draws, 4)

        statistical_features_raw = np.concatenate([
            odd_counts_arr,
            sums_arr,
            ranges_arr,
            zone_counts_arr,
            rolling_freq_features
        ], axis=1).astype(np.float32) # Ensure float32

        expected_cols = 1 + 1 + 1 + len(zone_defs) + num_freq_features
        logger.info(f"Features estatísticas calculadas. Shape: {statistical_features_raw.shape}")
        if statistical_features_raw.shape[1] != expected_cols:
             logger.error(f"Erro de shape nas features estatísticas! Esperado {expected_cols} colunas, obtido {statistical_features_raw.shape[1]}")
             return None

        return statistical_features_raw

    except Exception as e:
        logger.error(f"Erro ao calcular features estatísticas: {e}", exc_info=True)
        return None
# ### MODIFICATION END V3 ###


# ### MODIFICATION START V3 ###
# Updated split_data to handle statistical features and use two scalers
def split_data(encoded_labels, time_features_raw, statistical_features_raw,
               test_size_ratio, validation_split_ratio, sequence_length):
    """
    Divides labels and features (time, statistical) into train/val/test.
    Scales time and statistical features separately using StandardScaler fit on train data.
    Creates sequences for each set.
    """
    logger.info("Dividindo dados, escalando features (Tempo e Estatísticas separadamente)...")
    try:
        # Basic size validation
        n_samples = len(encoded_labels)
        if not (n_samples == len(time_features_raw) == len(statistical_features_raw)):
             logger.error("Disparidade no tamanho dos labels/features antes da divisão.")
             return [None] * 9 # Return 9 Nones (incl. 2 scalers)

        # 1. Split indices chronologically
        test_split_index = int(n_samples * (1 - test_size_ratio))
        val_split_index = int(test_split_index * (1 - validation_split_ratio))

        train_indices = np.arange(val_split_index)
        val_indices = np.arange(val_split_index, test_split_index)
        test_indices = np.arange(test_split_index, n_samples)

        # Check for sufficient data in each potential split *before* slicing
        min_len_for_seq = sequence_length + 1 # Need at least one sequence + target
        if len(train_indices) < min_len_for_seq:
             logger.error(f"Conjunto de treino ({len(train_indices)}) muito pequeno para seq={sequence_length}.")
             return [None] * 9
        if len(val_indices) < min_len_for_seq:
             logger.warning(f"Conjunto de validação ({len(val_indices)}) pequeno demais para seq={sequence_length}. Validação será vazia.")
             # Adjust indices to make validation empty
             train_indices = np.arange(test_split_index) # Train gets all non-test data
             val_indices = np.array([], dtype=int) # Validation is empty
        if len(test_indices) < min_len_for_seq:
             logger.warning(f"Conjunto de teste ({len(test_indices)}) pequeno demais para seq={sequence_length}. Teste será vazio.")
             test_indices = np.array([], dtype=int)

        logger.info(f"Índices Brutos - Treino: {len(train_indices)}, Val: {len(val_indices)}, Teste: {len(test_indices)}")

        # 2. Slice data based on indices
        train_labels = encoded_labels[train_indices]
        val_labels = encoded_labels[val_indices] if len(val_indices) > 0 else np.array([])
        test_labels = encoded_labels[test_indices] if len(test_indices) > 0 else np.array([])

        train_time_raw = time_features_raw[train_indices]
        val_time_raw = time_features_raw[val_indices] if len(val_indices) > 0 else np.array([])
        test_time_raw = time_features_raw[test_indices] if len(test_indices) > 0 else np.array([])

        train_stat_raw = statistical_features_raw[train_indices]
        val_stat_raw = statistical_features_raw[val_indices] if len(val_indices) > 0 else np.array([])
        test_stat_raw = statistical_features_raw[test_indices] if len(test_indices) > 0 else np.array([])


        # 3. Scale Features (Fit ONLY on Training Data, separately)
        logger.info("Ajustando Scalers (StandardScaler) nas features do treino...")
        time_scaler = StandardScaler()
        stat_scaler = StandardScaler()

        train_time_scaled = time_scaler.fit_transform(train_time_raw)
        train_stat_scaled = stat_scaler.fit_transform(train_stat_raw)
        logger.info("Escalando features de validação e teste...")

        val_time_scaled = time_scaler.transform(val_time_raw) if len(val_indices) > 0 else np.array([])
        test_time_scaled = time_scaler.transform(test_time_raw) if len(test_indices) > 0 else np.array([])

        val_stat_scaled = stat_scaler.transform(val_stat_raw) if len(val_indices) > 0 else np.array([])
        test_stat_scaled = stat_scaler.transform(test_stat_raw) if len(test_indices) > 0 else np.array([])

        # Optional: Save scalers
        # joblib.dump(time_scaler, 'time_scaler.gz')
        # joblib.dump(stat_scaler, 'stat_scaler.gz')
        # logger.info("Scalers salvos.")


        # 4. Create Sequences for each set using the specific create_sequences function
        logger.info("Criando sequências para treino...")
        X_train, y_train = create_sequences(train_labels, train_time_scaled, train_stat_scaled, sequence_length)

        logger.info("Criando sequências para validação...")
        X_val, y_val = create_sequences(val_labels, val_time_scaled, val_stat_scaled, sequence_length) if len(val_indices) >= min_len_for_seq else (np.array([]), np.array([]))

        logger.info("Criando sequências para teste...")
        X_test, y_test = create_sequences(test_labels, test_time_scaled, test_stat_scaled, sequence_length) if len(test_indices) >= min_len_for_seq else (np.array([]), np.array([]))

        logger.info(f"Tamanho final dos conjuntos de sequências:")
        logger.info(f"- Treino: X={X_train.shape if X_train.size>0 else 'Vazio'}, y={y_train.shape if y_train.size>0 else 'Vazio'}")
        logger.info(f"- Validação: X={X_val.shape if X_val.size>0 else 'Vazio'}, y={y_val.shape if y_val.size>0 else 'Vazio'}")
        logger.info(f"- Teste: X={X_test.shape if X_test.size>0 else 'Vazio'}, y={y_test.shape if y_test.size>0 else 'Vazio'}")


        # Return all sets and the fitted scalers
        return X_train, X_val, X_test, y_train, y_val, y_test, time_scaler, stat_scaler

    except Exception as e:
        logger.error(f"Erro ao dividir/escalar dados ou criar sequências: {e}", exc_info=True)
        return [None] * 9 # Return 9 Nones

# Updated create_sequences to handle three input arrays
def create_sequences(encoded_labels, time_features_scaled, statistical_features_scaled, sequence_length):
    """
    Creates sequences by combining labels, scaled time features, and scaled statistical features.
    X: Combined sequence.
    y: Next draw's labels.
    Handles potentially empty inputs gracefully.
    """
    # Check if inputs are valid and consistent before proceeding
    if (encoded_labels is None or encoded_labels.size == 0 or
        time_features_scaled is None or time_features_scaled.size == 0 or
        statistical_features_scaled is None or statistical_features_scaled.size == 0):
        # If any input is empty, return empty arrays, log warning if called for non-empty sets previously
        # logger.debug("Input vazio para create_sequences, retornando vazio.") # Optional Debug
        return np.array([]), np.array([])

    n_samples_total = len(encoded_labels)
    if not (n_samples_total == len(time_features_scaled) == len(statistical_features_scaled)):
        logger.error(f"Inconsistência de tamanho em create_sequences: "
                     f"Labels({n_samples_total}), Time({len(time_features_scaled)}), Stat({len(statistical_features_scaled)})")
        return np.array([]), np.array([])

    if n_samples_total <= sequence_length:
        # logger.debug(f"Dados insuficientes ({n_samples_total}) para seq={sequence_length} em create_sequences.") # Optional Debug
        return np.array([]), np.array([])

    # Proceed with sequence creation
    logger.debug(f"Criando sequências de tamanho {sequence_length} a partir de {n_samples_total} amostras...")
    try:
        num_sequences = n_samples_total - sequence_length
        num_features_base = encoded_labels.shape[1]
        num_features_time = time_features_scaled.shape[1]
        num_features_stat = statistical_features_scaled.shape[1]
        num_features_total = num_features_base + num_features_time + num_features_stat

        # Pre-allocate arrays
        # Ensure consistent dtype, float32 is common for TF
        X = np.zeros((num_sequences, sequence_length, num_features_total), dtype=np.float32)
        y = np.zeros((num_sequences, num_features_base), dtype=encoded_labels.dtype)

        for i in range(num_sequences):
            seq_labels = encoded_labels[i : i + sequence_length]
            seq_time = time_features_scaled[i : i + sequence_length]
            seq_stat = statistical_features_scaled[i : i + sequence_length]

            # Combine features along the last axis
            X[i] = np.concatenate((seq_labels, seq_time, seq_stat), axis=-1)
            y[i] = encoded_labels[i + sequence_length]

        logger.debug(f"{len(X)} sequências combinadas criadas. Shape X: {X.shape}, Shape y: {y.shape}")
        return X, y

    except Exception as e:
        logger.error(f"Erro ao criar sequências combinadas: {e}", exc_info=True)
        return np.array([]), np.array([])
# ### MODIFICATION END V3 ###


# --- Modelo ---

# ### MODIFICATION START V3 ###
# Updated build_model
def build_model(sequence_length, num_features_total, num_features_base, gru_units, dropout_rate, use_batch_norm):
    """ Constrói o modelo GRU com opção de Batch Normalization. """
    logger.info(f"Construindo modelo GRU: units={gru_units}, dropout={dropout_rate}, batch_norm={use_batch_norm}")
    try:
        
        # Cria o modelo sequencial com um nome
        model = Sequential(name="Modelo_GRU_MegaSena_V3")

        # Primeira camada: entrada com o formato (tamanho da sequência, número total de features)
        model.add(Input(shape=(sequence_length, num_features_total)))

        # Se o uso de batch normalization estiver ativado, aplica logo após a entrada
        if use_batch_norm:
            model.add(BatchNormalization())  # Normaliza os dados para facilitar o aprendizado

        # Primeira camada GRU
        # return_sequences=True mantém a saída em formato de sequência (necessário se houver outra GRU depois)
        model.add(GRU(
            gru_units,  # número de neurônios
            return_sequences=True,
            kernel_initializer='he_normal',  # forma de iniciar os pesos
            recurrent_initializer='orthogonal'  # forma de iniciar os pesos recorrentes
        ))

        # Normaliza a saída da primeira GRU (opcional, mas ajuda em alguns casos)
        if use_batch_norm:
            model.add(BatchNormalization())

        # Dropout para evitar overfitting (desliga aleatoriamente parte dos neurônios)
        model.add(Dropout(dropout_rate))

        # Segunda camada GRU (opcional), com metade dos neurônios e sem retornar sequência
        model.add(GRU(
            gru_units // 2,
            return_sequences=False,
            kernel_initializer='he_normal',
            recurrent_initializer='orthogonal'
        ))

        # Novamente, aplica normalização (se estiver ativado)
        if use_batch_norm:
            model.add(BatchNormalization())

        # Mais um dropout para regularização
        model.add(Dropout(dropout_rate))

        # Camada densa (totalmente conectada) com ReLU para aprender padrões não lineares
        model.add(Dense(gru_units // 2, activation='relu'))

        # Normaliza os dados antes da ativação ReLU (alguns testes indicam que isso pode ser melhor)
        if use_batch_norm:
            model.add(BatchNormalization())

        # Dropout antes da saída
        model.add(Dropout(dropout_rate))

        # Camada de saída com ativação sigmoid para prever os 60 números (resultado entre 0 e 1)
        model.add(Dense(num_features_base, activation='sigmoid'))

        # Otimizador Adam com taxa de aprendizado baixa (para treinar com mais calma e precisão)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0008)

        # Compila o modelo com:
        # - função de perda: binary_crossentropy (porque o resultado é binário: número foi ou não foi sorteado)
        # - métrica: acurácia binária e AUC (para medir a performance geral)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['binary_accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        # Construir o modelo para que os shapes sejam calculados
        model.build((None, sequence_length, num_features_total))
        
        # Usar uma versão de texto simplificada para o resumo do modelo em vez da versão gráfica
        logger.info("Resumo do Modelo:")
        
        # Exibir camadas em formato simples de texto
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        
        for i, layer in enumerate(model.layers):
            try:
                layer_info = f"Camada {i+1}: {layer.name} ({layer.__class__.__name__})"
                
                # Obter o shape da saída de forma segura
                if hasattr(layer, 'output_shape'):
                    output_shape = str(layer.output_shape)
                else:
                    # Fallback para camadas sem output_shape
                    output_shape = "Desconhecido"
                
                layer_info += f" - Output Shape: {output_shape}"
                
                # Contar parâmetros
                layer_params = layer.count_params()
                layer_info += f" - Parâmetros: {layer_params:,}"
                
                total_params += layer_params
                trainable_layer_params = sum([tf.size(w).numpy() for w in layer.trainable_weights]) if layer.trainable_weights else 0
                trainable_params += trainable_layer_params
                non_trainable_params += layer_params - trainable_layer_params
                
                logger.info(layer_info)
            except Exception as e:
                logger.warning(f"Erro ao obter informações da camada {i+1}: {e}")
        
        # Exibir totais
        logger.info(f"Total de parâmetros: {total_params:,}")
        logger.info(f"Parâmetros treináveis: {trainable_params:,}")
        logger.info(f"Parâmetros não-treináveis: {non_trainable_params:,}")
        
        return model
    except Exception as e:
        logger.error(f"Erro ao construir o modelo GRU: {e}", exc_info=True)
        return None
# ### MODIFICATION END V3 ###

# train_model updated to include TensorBoard callback
def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, log_dir):
    """ Treina o modelo GRU com callbacks otimizados e TensorBoard. """
    logger.info("Iniciando o treinamento do modelo GRU...")
    try:
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1) # Increased patience
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=10, min_lr=0.00001, verbose=1) # Adjusted factor/patience/min_lr

        # TensorBoard Callback
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1) # Log histograms

        # Custom Logger Callback (remains same)
        class TrainingLogger(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                log_str = f"Época {epoch + 1}/{self.params['epochs']} - Loss: {logs.get('loss', -1):.4f}"
                if 'binary_accuracy' in logs: log_str += f" - Acc: {logs.get('binary_accuracy', -1):.4f}"
                if 'auc' in logs: log_str += f" - AUC: {logs.get('auc', -1):.4f}"
                log_str += f" - Val Loss: {logs.get('val_loss', -1):.4f}"
                if 'val_binary_accuracy' in logs: log_str += f" - Val Acc: {logs.get('val_binary_accuracy', -1):.4f}"
                if 'val_auc' in logs: log_str += f" - Val AUC: {logs.get('val_auc', -1):.4f}"
                if hasattr(self.model.optimizer, 'learning_rate'):
                     lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
                     log_str += f" - LR: {lr:.6f}"
                logger.info(log_str)

        # Determine validation data (handle empty validation set)
        validation_data = (X_val, y_val) if X_val.size > 0 and y_val.size > 0 else None
        if validation_data is None:
             logger.warning("Conjunto de validação vazio. EarlyStopping/ReduceLR usarão 'loss' ao invés de 'val_loss'.")
             early_stopping.monitor = 'loss' # Monitor training loss if no validation
             reduce_lr.monitor = 'loss'

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[early_stopping, reduce_lr, tensorboard_callback, TrainingLogger()],
            verbose=0 # Use custom logger
        )
        logger.info("Treinamento concluído.")
        return history
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {e}", exc_info=True)
        return None


# --- Avaliação e Previsão ---

# evaluate_real_hits remains the same as V2
def evaluate_real_hits(model, X_test, y_test, batch_size=32):
    """Evaluates how many of the top 6 predicted numbers were actually drawn."""
    # (Implementation from V2)
    logger.info("Avaliando acertos reais nas previsões (Top 6)...")
    try:
        if model is None or X_test is None or y_test is None or len(X_test) == 0:
             logger.error("Dados inválidos/vazios para avaliação de acertos (X_test, y_test)")
             return None
        y_pred_probs = model.predict(X_test, batch_size=batch_size)
        if y_pred_probs.shape[0] != y_test.shape[0] or y_pred_probs.shape[1] != y_test.shape[1]:
             logger.error(f"Shape mismatch: Preds({y_pred_probs.shape}) vs Test({y_test.shape})")
             return None
        hits_per_draw, detailed_hits = [], []
        for i in range(len(y_pred_probs)):
            top6_pred_indices = np.argsort(y_pred_probs[i])[-6:]
            predicted_numbers = sorted((top6_pred_indices + 1).tolist())
            actual_winning_indices = np.where(y_test[i] == 1)[0]
            actual_numbers = sorted((actual_winning_indices + 1).tolist())
            hits = set(predicted_numbers) & set(actual_numbers)
            num_hits = len(hits)
            detailed_hits.append({ 'sorteio_index_teste': i, 'previstos': predicted_numbers,
                                   'sorteados': actual_numbers, 'acertos': sorted(list(hits)),
                                   'num_acertos': num_hits })
            hits_per_draw.append(num_hits)
        if not hits_per_draw:
             logger.warning("Nenhum sorteio no teste para calcular stats de acertos.")
             avg_hits, max_hits, hits_distribution = 0, 0, {}
        else:
             avg_hits = np.mean(hits_per_draw)
             max_hits = np.max(hits_per_draw)
             hits_distribution = {i: hits_per_draw.count(i) for i in range(max_hits + 1)}
        logger.info("-" * 60)
        logger.info("ANÁLISE DE ACERTOS REAIS (TOP 6 PREVISTOS vs SORTEADOS)")
        # (Logging details from V2)
        logger.info(f"Total de sorteios no teste: {len(hits_per_draw)}")
        if hits_per_draw:
             logger.info(f"Média acertos: {avg_hits:.3f}, Máximo acertos: {max_hits}")
             logger.info("Distribuição:")
             for hits_count, count in hits_distribution.items():
                 if count > 0: logger.info(f"  {hits_count} acerto(s): {count} ({(count / len(hits_per_draw)) * 100:.1f}%)")
             logger.info(f"Exemplo últimos {min(5, len(hits_per_draw))} sorteios teste:")
             for hit_detail in detailed_hits[-min(5, len(hits_per_draw)):]: logger.info(f"  Idx {hit_detail['sorteio_index_teste']}: Prev{hit_detail['previstos']} Real{hit_detail['sorteados']} -> Hits {hit_detail['acertos']} ({hit_detail['num_acertos']})")
        logger.info("-" * 60 + "\nAVISO: Acertos passados NÃO garantem acertos futuros.\n" + "-" * 60)
        return { 'hits_per_draw': hits_per_draw, 'avg_hits': avg_hits, 'max_hits': max_hits,
                 'hits_distribution': hits_distribution, 'detailed_hits': detailed_hits }
    except Exception as e:
        logger.error(f"Erro ao avaliar acertos reais: {e}", exc_info=True)
        return None

# evaluate_model remains the same as V2
def evaluate_model(model, X_test, y_test, batch_size=32):
    """Evaluates the model on the test set using standard metrics and real hits."""
    # (Implementation from V2)
    logger.info("Avaliando o modelo final no conjunto de teste...")
    try:
        if model is None or X_test is None or y_test is None or len(X_test) == 0:
            logger.error("Dados inválidos/vazios para avaliação final (model, X_test, y_test)")
            return None
        logger.info("Calculando métricas básicas (Loss, Accuracy, AUC)...")
        results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        basic_metrics_dict = dict(zip(model.metrics_names, results))
        logger.info("Calculando acertos reais (Top 6)...")
        real_hits_results = evaluate_real_hits(model, X_test, y_test, batch_size)
        if real_hits_results is None: logger.error("Falha na avaliação de acertos reais.")
        logger.info("-" * 60 + "\nResultados da Avaliação no Conjunto de Teste\n" + "-" * 60)
        logger.info("Métricas Padrão:")
        for name, value in basic_metrics_dict.items(): logger.info(f"  - {name}: {value:.4f}")
        logger.info("\nEstatísticas de Acertos Reais (Top 6):")
        if real_hits_results:
             logger.info(f"  - Média acertos: {real_hits_results['avg_hits']:.3f}, Máx acertos: {real_hits_results['max_hits']}")
             logger.info("  Distribuição:")
             total_test_draws = len(real_hits_results.get('hits_per_draw', []))
             if total_test_draws > 0:
                 hits_dist = real_hits_results.get('hits_distribution', {})
                 for hits_count, count in hits_dist.items():
                      if count > 0: logger.info(f"    * {hits_count} acerto(s): {count} ({(count/total_test_draws)*100:.1f}%)")
             else: logger.info("    N/A (sem sorteios)")
        else: logger.info("  N/A (falha no cálculo)")
        logger.info("-" * 60 + "\nLembrete: Métricas refletem desempenho no passado.\n" + "-" * 60)
        return { 'basic_metrics': basic_metrics_dict, 'real_hits': real_hits_results }
    except Exception as e:
        logger.error(f"Erro durante a avaliação final: {e}", exc_info=True)
        return None

# ### MODIFICATION START V3 ###
# predict_next_draw updated to use both scalers
def predict_next_draw(model, last_sequence_labels, last_sequence_time_raw, last_sequence_stat_raw,
                      time_scaler, stat_scaler, mlb, num_predictions=6):
    """
    Prepares the final sequence (scaling features) and predicts the next draw.
    """
    logger.info("Preparando sequência final e fazendo previsão para o PRÓXIMO sorteio...")
    try:
        # Validate inputs
        if model is None: logger.error("Modelo inválido."); return None, None
        if time_scaler is None or stat_scaler is None: logger.error("Scalers inválidos."); return None, None
        seq_len = config['sequence_length']
        if not all(s is not None and len(s) == seq_len for s in [last_sequence_labels, last_sequence_time_raw, last_sequence_stat_raw]):
             logger.error(f"Última sequência inválida ou tamanho incorreto (esperado {seq_len}).")
             return None, None

        # 1. Scale the raw time and statistical features
        logger.debug("Escalando features da última sequência...")
        last_sequence_time_scaled = time_scaler.transform(last_sequence_time_raw)
        last_sequence_stat_scaled = stat_scaler.transform(last_sequence_stat_raw)

        # 2. Combine labels, scaled time, scaled stats
        last_sequence_combined = np.concatenate(
            (last_sequence_labels, last_sequence_time_scaled, last_sequence_stat_scaled),
            axis=-1
        ).astype(np.float32) # Ensure correct dtype

        # 3. Add batch dimension and predict
        last_sequence_batch = np.expand_dims(last_sequence_combined, axis=0)
        logger.info(f"Shape da sequência final para previsão: {last_sequence_batch.shape}")
        predicted_probabilities = model.predict(last_sequence_batch)[0]

        # 4. Process predictions (same as before)
        expected_output_shape = config['num_features_base']
        if predicted_probabilities.shape[0] != expected_output_shape:
            logger.error(f"Shape inesperado da previsão: {predicted_probabilities.shape}. Esperado: ({expected_output_shape},)")
            return None, None
        predicted_indices = np.argsort(predicted_probabilities)[-num_predictions:]
        predicted_numbers = sorted((predicted_indices + 1).tolist())
        confidence_scores = predicted_probabilities[predicted_indices]
        avg_confidence = np.mean(confidence_scores) if confidence_scores.size > 0 else 0.0
        max_confidence = np.max(confidence_scores) if confidence_scores.size > 0 else 0.0
        min_confidence = np.min(confidence_scores) if confidence_scores.size > 0 else 0.0

        # 5. Log results (same as before)
        logger.info("-" * 50)
        logger.info(f"Previsão dos {num_predictions} números mais prováveis: {predicted_numbers}")
        logger.info(f"Confiança Média: {avg_confidence:.4f}, Máx: {max_confidence:.4f}, Mín: {min_confidence:.4f}")
        logger.info("Probabilidades individuais (previstos):")
        sorted_pred_indices = predicted_indices[np.argsort(confidence_scores)[::-1]]
        for num_idx in sorted_pred_indices:
            logger.info(f"  - Número {num_idx + 1}: {predicted_probabilities[num_idx]:.4f}")
        logger.info("-" * 50 + "\nAVISO CRÍTICO: Previsão experimental. NÃO HÁ GARANTIA DE ACERTO.\n" + "-" * 50)

        return predicted_numbers, predicted_probabilities

    except Exception as e:
        logger.error(f"Erro durante a previsão do próximo sorteio: {e}", exc_info=True)
        return None, None
# ### MODIFICATION END V3 ###


# --- Visualização e Exportação ---

# plot_training_history remains the same as V2 (but uses new filename)
def plot_training_history(history, filename=None):
    """ Plots training history (Loss, Accuracy, AUC, LR). """
    # Use default filename if none provided
    if filename is None:
        filename = os.path.join(output_dir, 'training_history_v3.png')
    
    # (Implementation from V2 - robust plotting)
    logger.info(f"Gerando gráficos do histórico de treinamento em {filename}...")
    try:
        if history is None or not hasattr(history, 'history') or not history.history:
            logger.error("Histórico de treinamento inválido/vazio."); return
        plt.figure(figsize=(15, 10)); history_dict = history.history
        metrics = [m for m in ['loss', 'binary_accuracy', 'auc'] if m in history_dict] # Exclude 'lr' initially
        num_plots = len(metrics) + (1 if 'lr' in history_dict else 0) # Add 1 for LR plot
        num_cols = 2; num_rows = (num_plots + num_cols - 1) // num_cols
        plot_index = 1
        for metric in metrics:
            plt.subplot(num_rows, num_cols, plot_index)
            plt.plot(history_dict[metric], label=f'Treino {metric.capitalize()}')
            if f'val_{metric}' in history_dict: plt.plot(history_dict[f'val_{metric}'], label=f'Validação {metric.capitalize()}')
            plt.title(f'{metric.replace("_", " ").capitalize()} por Época'); plt.xlabel('Época'); plt.ylabel(metric.capitalize())
            plt.legend(); plt.grid(True); plot_index += 1
        if 'lr' in history_dict:
             plt.subplot(num_rows, num_cols, plot_index); plt.plot(history_dict['lr'], label='Taxa Aprendizado')
             plt.title('Taxa de Aprendizado'); plt.xlabel('Época'); plt.ylabel('Learning Rate')
             plt.legend(); plt.grid(True); plot_index +=1
        plt.tight_layout(); plt.savefig(filename)
        logger.info(f"Gráficos salvos em '{filename}'"); plt.close()
    except Exception as e: logger.error(f"Erro ao gerar gráficos de treinamento: {e}", exc_info=True)

# plot_prediction_analysis remains the same as V2 (but uses new filename)
def plot_prediction_analysis(predicted_numbers, predicted_probabilities, df_full_valid, sequence_length, filename=None):
    """ Generates visual analysis of predictions vs recent frequency. """
    # Use default filename if none provided
    if filename is None:
        filename = os.path.join(output_dir, 'prediction_analysis_v3.png')
    
    # (Implementation from V2 - plots probabilities and recent frequency)
    logger.info(f"Gerando análise visual das previsões em {filename}...")
    try:
        if predicted_numbers is None or predicted_probabilities is None: logger.error("Dados de previsão inválidos."); return
        if df_full_valid is None or df_full_valid.empty: logger.error("DataFrame histórico inválido."); return
        plt.figure(figsize=(15, 12)); all_numbers = np.arange(1, config['num_features_base'] + 1)
        # Plot 1: All Probabilities
        plt.subplot(2, 2, 1); plt.bar(all_numbers, predicted_probabilities, width=0.8); plt.title(f'Probabilidades Previstas (Todos {config["num_features_base"]})')
        plt.xlabel('Número'); plt.ylabel('Probabilidade'); plt.grid(True, axis='y', alpha=0.7); plt.xticks(np.arange(0, 61, 5)); plt.xlim(0.5, 60.5)
        # Plot 2: Predicted Probabilities
        plt.subplot(2, 2, 2); predicted_numbers_arr = np.array(predicted_numbers); probs_for_predicted = predicted_probabilities[predicted_numbers_arr - 1]
        bars = plt.bar(predicted_numbers_arr, probs_for_predicted, width=0.6, color='red'); plt.title(f'Probabilidades dos {len(predicted_numbers)} Números Previstos')
        plt.xlabel('Número Previsto'); plt.ylabel('Probabilidade'); plt.grid(True, axis='y', alpha=0.7); plt.bar_label(bars, fmt='%.4f', padding=3)
        if probs_for_predicted.size > 0: plt.ylim(0, max(probs_for_predicted) * 1.15)
        plt.xticks(predicted_numbers_arr)
        # Plot 3: Recent Frequency
        plt.subplot(2, 2, 3); last_n_draws_df = df_full_valid.iloc[-sequence_length:]; bola_cols = [f'Bola{i+1}' for i in range(6)]
        last_numbers_flat = pd.concat([last_n_draws_df[col] for col in bola_cols]).dropna().astype(int).values
        number_freq = np.zeros(config['num_features_base']); unique_nums, counts = np.unique(last_numbers_flat, return_counts=True)
        valid_mask = (unique_nums >= 1) & (unique_nums <= config['num_features_base']); valid_unique_nums = unique_nums[valid_mask]; valid_counts = counts[valid_mask]
        if valid_unique_nums.size > 0: number_freq[valid_unique_nums - 1] = valid_counts
        plt.bar(all_numbers, number_freq, width=0.8); plt.title(f'Frequência nos Últimos {sequence_length} Sorteios Históricos')
        plt.xlabel('Número'); plt.ylabel('Frequência'); plt.grid(True, axis='y', alpha=0.7); plt.xticks(np.arange(0, 61, 5)); plt.xlim(0.5, 60.5)
        if number_freq.max() > 0: plt.yticks(np.arange(0, number_freq.max() + 1, 1))
        # Plot 4: Frequency vs Probability
        plt.subplot(2, 2, 4); plt.scatter(number_freq, predicted_probabilities, alpha=0.6); freq_for_predicted = number_freq[predicted_numbers_arr - 1]
        plt.scatter(freq_for_predicted, probs_for_predicted, color='red', s=80, label='Previstos', edgecolors='black'); plt.title('Frequência Recente vs Probabilidade Prevista')
        plt.xlabel(f'Frequência Últimos {sequence_length} Sorteios'); plt.ylabel('Probabilidade Prevista'); plt.grid(True, alpha=0.7); plt.legend()
        for i, num in enumerate(predicted_numbers_arr): plt.text(freq_for_predicted[i] + 0.05 * (number_freq.max() if number_freq.max()>0 else 1), probs_for_predicted[i], str(num), fontsize=9)
        plt.tight_layout(); plt.savefig(filename); logger.info(f"Análise visual salva em '{filename}'"); plt.close()
    except Exception as e: logger.error(f"Erro ao gerar análise visual das previsões: {e}", exc_info=True)

# plot_hits_over_time remains the same as V2 (but uses new filename)
def plot_hits_over_time(model, X_test, y_test, mlb, filename=None):
    """ Plots number of hits (top 6 predicted vs actual) over the test set. """
    # Use default filename if none provided
    if filename is None:
        filename = os.path.join(output_dir, 'hits_over_time_v3.png')
    
    # (Implementation from V2 - calculates and plots hits)
    logger.info(f"Gerando gráfico de acertos ao longo do tempo no teste em {filename}...")
    if X_test is None or y_test is None or X_test.shape[0] == 0: logger.warning("Dados de teste insuficientes para plotar acertos."); return None
    logger.info("Calculando acertos no teste para plotagem...")
    try:
        y_pred_probs_test = model.predict(X_test); hits_per_draw = []
        for i in range(len(y_pred_probs_test)):
            top6_pred_indices = np.argsort(y_pred_probs_test[i])[-6:]
            actual_winning_indices = np.where(y_test[i] == 1)[0]
            num_hits = len(set(top6_pred_indices) & set(actual_winning_indices))
            hits_per_draw.append(num_hits)
        if not hits_per_draw: logger.warning("Nenhum resultado de acerto calculado."); return None
        plt.figure(figsize=(15, 6)); plt.plot(hits_per_draw, marker='o', linestyle='-', markersize=4, label='Nº Acertos (Top 6) / Sorteio Teste')
        if len(hits_per_draw) >= 10:
             rolling_avg = pd.Series(hits_per_draw).rolling(window=10, min_periods=1).mean()
             plt.plot(rolling_avg, linestyle='--', color='red', label='Média Móvel (10)')
        plt.xlabel("Índice Sorteio Teste"); plt.ylabel("Número Acertos (Top 6)"); plt.title("Acertos Modelo no Teste Histórico")
        plt.yticks(np.arange(0, 7, 1)); plt.ylim(bottom=-0.2); plt.grid(True, alpha=0.7); plt.legend()
        plt.savefig(filename); logger.info(f"Gráfico de acertos salvo em '{filename}'"); plt.close()
        return hits_per_draw
    except Exception as e: logger.error(f"Erro ao gerar gráfico de acertos: {e}", exc_info=True); return None

# export_results remains the same as V2 (but uses new filename)
def export_results(df_full_valid, predicted_numbers, predicted_probabilities, evaluation_results, config):
    """ Exports history, predictions, evaluation metrics, and config to Excel. """
    # (Implementation from V2 - detailed Excel export)
    logger.info(f"Exportando resultados para Excel ({config['export_file']})...")
    try:
        if df_full_valid is None: logger.error("DataFrame histórico inválido."); return
        if predicted_numbers is None or predicted_probabilities is None: logger.error("Dados de previsão inválidos."); return
        if evaluation_results is None:
            logger.warning("Resultados da avaliação inválidos, exportando sem eles.")
            evaluation_results = {'basic_metrics': {}, 'real_hits': None}
        predictions_df = pd.DataFrame({'Número': range(1, config['num_features_base'] + 1), 'Probabilidade_Prevista': predicted_probabilities})
        predictions_df['Previsto_Top_6'] = predictions_df['Número'].isin(predicted_numbers)
        predictions_df = predictions_df.sort_values('Probabilidade_Prevista', ascending=False).reset_index(drop=True)
        basic_metrics_dict = evaluation_results.get('basic_metrics', {})
        if not basic_metrics_dict: metrics_df = pd.DataFrame({'Métrica': ['N/A'], 'Valor': ['N/A']})
        else: metrics_df = pd.DataFrame({'Métrica': list(basic_metrics_dict.keys()), 'Valor': [f"{v:.5f}" for v in basic_metrics_dict.values()] })
        real_hits_results = evaluation_results.get('real_hits')
        if real_hits_results and real_hits_results.get('hits_distribution') is not None:
            hits_dist = real_hits_results['hits_distribution']; total_draws = len(real_hits_results.get('hits_per_draw', []))
            hits_summary_df = pd.DataFrame({'Estatística': ['Média Acertos (Top 6)', 'Máx Acertos (Top 6)', 'Total Sorteios Teste'], 'Valor': [f"{real_hits_results.get('avg_hits', 'N/A'):.3f}", f"{real_hits_results.get('max_hits', 'N/A')}", total_draws]})
            hits_dist_df = pd.DataFrame({'Número Acertos': list(hits_dist.keys()), 'Qtd Sorteios': list(hits_dist.values())})
            if total_draws > 0: hits_dist_df['Porcentagem (%)'] = hits_dist_df['Qtd Sorteios'].apply(lambda c: f"{(c / total_draws) * 100:.1f}%")
            else: hits_dist_df['Porcentagem (%)'] = 'N/A'
            hits_dist_df = hits_dist_df.sort_values('Número Acertos').reset_index(drop=True)
            detailed_hits = real_hits_results.get('detailed_hits')
            if detailed_hits:
                 detailed_hits_df = pd.DataFrame(detailed_hits)
                 for col in ['previstos', 'sorteados', 'acertos']:
                     if col in detailed_hits_df.columns: detailed_hits_df[col] = detailed_hits_df[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
            else: detailed_hits_df = pd.DataFrame({'Info': ['Detalhes acertos não disponíveis']})
        else:
            logger.warning("Resultados acertos reais não encontrados/incompletos."); hits_summary_df = pd.DataFrame({'Estatística': ['N/A'], 'Valor': ['N/A']})
            hits_dist_df = pd.DataFrame({'Número Acertos': ['N/A'], 'Qtd Sorteios': ['N/A'], 'Porcentagem (%)': ['N/A']}); detailed_hits_df = pd.DataFrame({'Info': ['Detalhes acertos não disponíveis']})
        logger.info("Escrevendo abas no Excel...")
        with pd.ExcelWriter(config['export_file'], engine='openpyxl') as writer:
            predictions_df.to_excel(writer, sheet_name='Previsao_Probabilidades', index=False); logger.debug("Aba Previsao_Probabilidades OK.")
            metrics_df.to_excel(writer, sheet_name='Metricas_Avaliacao', index=False); logger.debug("Aba Metricas_Avaliacao OK.")
            hits_summary_df.to_excel(writer, sheet_name='Sumario_Acertos_Reais', index=False); logger.debug("Aba Sumario_Acertos_Reais OK.")
            hits_dist_df.to_excel(writer, sheet_name='Distribuicao_Acertos_Reais', index=False); logger.debug("Aba Distribuicao_Acertos_Reais OK.")
            if detailed_hits_df is not None: detailed_hits_df.to_excel(writer, sheet_name='Detalhes_Acertos_Teste', index=False); logger.debug("Aba Detalhes_Acertos_Teste OK.")
            df_full_valid.to_excel(writer, sheet_name='Historico_Utilizado', index=False); logger.debug("Aba Historico_Utilizado OK.")
            config_df = pd.DataFrame(list(config.items()), columns=['Parametro', 'Valor']); config_df.to_excel(writer, sheet_name='Configuracao_Usada', index=False); logger.debug("Aba Configuracao_Usada OK.")
        logger.info(f"Resultados exportados para '{config['export_file']}'")
    except PermissionError: logger.error(f"Erro de permissão ao escrever '{config['export_file']}'. Arquivo aberto?")
    except Exception as e: logger.error(f"Erro ao exportar para Excel: {e}", exc_info=True)

# validate_config updated for V3 settings
def validate_config(config):
    """ Valida as configurações V3. """
    logger.info("Validando configuração V3...")
    is_valid = True
    try:
        required_fields = [ 'data_url', 'data_file', 'export_file', 'sequence_length', 'num_features_base', 'num_features_time',
                            'num_features_statistical', 'num_features_total', 'rolling_freq_windows', 'gru_units', 'use_batch_norm',
                            'dropout_rate', 'epochs', 'batch_size', 'test_size_ratio', 'validation_split_ratio',
                            'cache_duration_hours', 'cache_dir', 'tensorboard_log_dir' ]
        for field in required_fields:
            if field not in config: logger.error(f"Campo obrigatório ausente: {field}"); is_valid = False
        if not is_valid: return False # Early exit

        # Type/Value checks (add more as needed)
        if not isinstance(config['sequence_length'], int) or config['sequence_length'] < 1: logger.error("sequence_length invalido"); is_valid = False
        if not isinstance(config['num_features_base'], int) or config['num_features_base'] <= 0: logger.error("num_features_base invalido"); is_valid = False
        if not isinstance(config['num_features_time'], int) or config['num_features_time'] < 0: logger.error("num_features_time invalido"); is_valid = False
        if not isinstance(config['num_features_statistical'], int) or config['num_features_statistical'] < 0: logger.error("num_features_statistical invalido"); is_valid = False
        if config['num_features_total'] != config['num_features_base'] + config['num_features_time'] + config['num_features_statistical']:
             logger.error("num_features_total != soma das partes"); is_valid = False
        if not isinstance(config['rolling_freq_windows'], list) or not all(isinstance(x, int) and x > 0 for x in config['rolling_freq_windows']):
             logger.error("rolling_freq_windows deve ser lista de inteiros positivos"); is_valid = False
        # Check consistency of statistical features count if windows defined
        expected_stat_count = 1 + 1 + 1 + 4 + len(config['rolling_freq_windows']) * config['num_features_base']
        if config['num_features_statistical'] != expected_stat_count:
            logger.warning(f"num_features_statistical ({config['num_features_statistical']}) "
                           f"não bate com o esperado pelos windows ({expected_stat_count}). Usando valor da config.")
            # Decide if this should be an error or just warning
            # is_valid = False
        if not isinstance(config['gru_units'], int) or config['gru_units'] < 1: logger.error("gru_units invalido"); is_valid = False
        if not isinstance(config['use_batch_norm'], bool): logger.error("use_batch_norm deve ser true/false"); is_valid = False
        if not isinstance(config['dropout_rate'], (int, float)) or not 0 <= config['dropout_rate'] < 1: logger.error("dropout_rate invalido"); is_valid = False
        if not isinstance(config['epochs'], int) or config['epochs'] < 1: logger.error("epochs invalido"); is_valid = False
        if not isinstance(config['batch_size'], int) or config['batch_size'] < 1: logger.error("batch_size invalido"); is_valid = False
        if not isinstance(config['test_size_ratio'], (int, float)) or not 0 < config['test_size_ratio'] < 1: logger.error("test_size_ratio invalido"); is_valid = False
        if not isinstance(config['validation_split_ratio'], (int, float)) or not 0 <= config['validation_split_ratio'] < 1: logger.error("validation_split_ratio invalido"); is_valid = False # Allow 0 validation
        if (config['test_size_ratio'] + config['validation_split_ratio']) >= 1.0: logger.error("Soma test+validation ratio >= 1"); is_valid = False
        if not isinstance(config['cache_duration_hours'], (int, float)) or config['cache_duration_hours'] < 0: logger.error("cache_duration_hours invalido"); is_valid = False
        if config.get('data_url') is None and config.get('data_file') is None: logger.error("É necessário 'data_url' ou 'data_file'."); is_valid = False
        # ... (other checks from V2: url format, export format, cache dir)

        if is_valid: logger.info("Configuração V3 validada com sucesso.")
        else: logger.error("Validação da configuração V3 falhou.")
        return is_valid
    except Exception as e:
        logger.error(f"Erro inesperado na validação da config: {e}", exc_info=True)
        return False


# --- Fluxo Principal (Main) ---
def main():
    """ Função principal do programa V3. """
    run_start_time = datetime.now()
    logger.info(f"Iniciando Mega-Sena V3 em {run_start_time.strftime('%Y-%m-%d %H:%M:%S')}...")
    try:
        global config
        config = load_config()
        if not validate_config(config):
            logger.critical("Configuração inválida. Abortando.")
            return

        # --- Pipeline ---
        logger.info("Etapa 1: Download/Preparação Dados...")
        df_full = download_and_prepare_data(url=config['data_url'], file_path=config['data_file'])
        if df_full is None or df_full.empty: logger.critical("Falha Etapa 1. Abortando."); return

        logger.info("Etapa 2: Pré-processamento Labels...")
        bola_cols = [f'Bola{i+1}' for i in range(6)]
        encoded_labels, mlb, valid_indices = preprocess_data_labels(df_full[bola_cols], config['num_features_base'])
        if encoded_labels is None or mlb is None or valid_indices is None: logger.critical("Falha Etapa 2. Abortando."); return
        df_full_valid = df_full.loc[valid_indices].reset_index(drop=True)
        logger.info(f"Labels processados: {len(df_full_valid)} sorteios válidos.")

        logger.info("Etapa 3a: Cálculo Features Tempo...")
        time_features_raw = add_time_features(df_full_valid[bola_cols], config['num_features_base'])
        if time_features_raw is None or len(time_features_raw) != len(encoded_labels): logger.critical("Falha Etapa 3a. Abortando."); return

        # ### MODIFICATION START V3 ###
        logger.info("Etapa 3b: Cálculo Features Estatísticas...")
        statistical_features_raw = add_statistical_features(df_full_valid[bola_cols], config['num_features_base'], config['rolling_freq_windows'])
        if statistical_features_raw is None or len(statistical_features_raw) != len(encoded_labels): logger.critical("Falha Etapa 3b. Abortando."); return
        # ### MODIFICATION END V3 ###

        # --- Teste de Hiperparâmetros (se ativado) ---
        if config.get('test_hyperparameters', False) and hyperparameter_tuning_available:
            logger.info("-" * 60)
            logger.info("MODO DE TESTE DE HIPERPARÂMETROS ATIVADO")
            logger.info("-" * 60)
            
            # Instanciar o tuner
            tuner = HyperparameterTuner(
                base_config=config,
                encoded_labels=encoded_labels,
                time_features_raw=time_features_raw,
                statistical_features_raw=statistical_features_raw,
                build_model_fn=build_model,
                split_data_fn=split_data,
                validate_config_fn=validate_config,
                output_dir=output_dir
            )
            
            # Executar a busca de hiperparâmetros
            logger.info("Iniciando busca de hiperparâmetros...")
            best_params = tuner.run_search()
            
            if best_params:
                logger.info("Aplicando a melhor configuração de hiperparâmetros encontrada.")
                # Atualizar a configuração com os melhores parâmetros
                for key, value in best_params.items():
                    config[key] = value
                logger.info(f"Configuração atualizada com os melhores hiperparâmetros: {best_params}")
                
                # Recalcular num_features_total (caso sequence_length tenha mudado)
                config['num_features_total'] = (
                    config['num_features_base'] +
                    config['num_features_time'] +
                    config['num_features_statistical']
                )
            else:
                logger.warning("Não foi possível determinar a melhor configuração. Usando configuração original.")
                
            logger.info("-" * 60)
            logger.info("CONCLUÍDO TESTE DE HIPERPARÂMETROS. CONTINUANDO COM TREINAMENTO FINAL.")
            logger.info("-" * 60)
        
        # --- Fluxo Normal (com ou sem otimização de hiperparâmetros) ---
        logger.info("Etapa 4: Divisão/Escalonamento/Sequenciamento...")
        # ### MODIFICATION START V3 ###
        # Pass statistical features and expect two scalers back
        X_train, X_val, X_test, y_train, y_val, y_test, time_scaler, stat_scaler = split_data(
            encoded_labels,
            time_features_raw,
            statistical_features_raw, # Added
            config['test_size_ratio'],
            config['validation_split_ratio'],
            config['sequence_length']
        )
        # Check if all 8 results are valid (9th is stat_scaler)
        if any(data is None for data in [X_train, X_val, X_test, y_train, y_val, y_test, time_scaler, stat_scaler]):
             logger.critical("Falha Etapa 4. Abortando."); return
        if X_train.size == 0 or y_train.size == 0: logger.critical("Conjunto de Treino Vazio! Abortando."); return
        # ### MODIFICATION END V3 ###

        logger.info("Etapa 5: Construção Modelo GRU...")
        model = build_model(
            config['sequence_length'],
            config['num_features_total'], # Use total features
            config['num_features_base'],
            config['gru_units'],
            config['dropout_rate'],
            config['use_batch_norm'] # Added config option
        )
        if model is None: logger.critical("Falha Etapa 5. Abortando."); return

        logger.info("Etapa 6: Treinamento Modelo...")
        # Create unique log dir for TensorBoard for this run
        log_dir = os.path.join(config['tensorboard_log_dir'], datetime.now().strftime("%Y%m%d-%H%M%S"))
        history = train_model( model, X_train, y_train, X_val, y_val,
                               config['epochs'], config['batch_size'], log_dir) # Pass log_dir
        if history is None: logger.critical("Falha Etapa 6. Abortando."); return

        logger.info("Etapa 7: Avaliação Modelo...")
        evaluation_results = None
        if X_test.size > 0 and y_test.size > 0:
             evaluation_results = evaluate_model(model, X_test, y_test, config['batch_size'])
             if evaluation_results is None: logger.warning("Falha Etapa 7. Continuando sem avaliação.")
             evaluation_results = evaluation_results or {'basic_metrics': {}, 'real_hits': None} # Ensure structure exists
        else:
             logger.warning("Conjunto de teste vazio. Pulando avaliação.")
             evaluation_results = {'basic_metrics': {}, 'real_hits': None}

        logger.info("Etapa 8: Previsão Próximo Sorteio...")
        # ### MODIFICATION START V3 ###
        # Prepare last sequence parts: labels, raw time, raw stats
        last_sequence_labels = encoded_labels[-config['sequence_length']:]
        last_sequence_time_raw = time_features_raw[-config['sequence_length']:]
        last_sequence_stat_raw = statistical_features_raw[-config['sequence_length']:]

        predicted_numbers, predicted_probabilities = predict_next_draw(
             model, last_sequence_labels, last_sequence_time_raw, last_sequence_stat_raw,
             time_scaler, stat_scaler, mlb # Pass both scalers
        )
        # ### MODIFICATION END V3 ###
        if predicted_numbers is None or predicted_probabilities is None: logger.critical("Falha Etapa 8. Abortando."); return

        logger.info("Etapa 9: Visualizações...")
        plot_training_history(history)
        plot_prediction_analysis(predicted_numbers, predicted_probabilities, df_full_valid, config['sequence_length'])
        if X_test.size > 0 and y_test.size > 0: plot_hits_over_time(model, X_test, y_test, mlb)
        else: logger.info("Pulando gráfico acertos (teste vazio).")

        logger.info("Etapa 10: Exportação Resultados...")
        export_results(df_full_valid, predicted_numbers, predicted_probabilities, evaluation_results, config)

        # --- Conclusão ---
        run_end_time = datetime.now()
        logger.info("-" * 60 + f"\nProcesso V3 concluído com sucesso em: {run_end_time - run_start_time}\n" + "-" * 60)
        logger.info(f"Log: {os.path.join(output_dir, 'mega_sena_v3.log')} | Excel: {config['export_file']}")
        logger.info(f"Gráficos: {os.path.join(output_dir, 'training_history_v3.png')}, {os.path.join(output_dir, 'prediction_analysis_v3.png')}, {os.path.join(output_dir, 'hits_over_time_v3.png')}")
        logger.info(f"Logs TensorBoard: {config['tensorboard_log_dir']} (use 'tensorboard --logdir {config['tensorboard_log_dir']}' para visualizar)")
        logger.info("-" * 60 + "\nLembre-se: Resultados experimentais. Jogue com responsabilidade.\n" + "-" * 60)

    except Exception as e:
        logger.critical(f"Erro GERAL não tratado no main: {e}", exc_info=True)
        return

if __name__ == "__main__":
    main()