# -*- coding: utf-8 -*-
"""
Script de Exemplo para "Previsão" da Mega-Sena - Versão V4.
MODIFIED: Added Attention Mechanism after GRU layers.
          Implemented CosineDecay LR Scheduler and AdamW optimizer.
          Adjusted hyperparameters.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
# import joblib # Optional: if you want to save/load scalers
import tensorflow as tf
# ### MODIFICATION START V4 ###
# Added Attention, AdditiveAttention, AdamW (from addons or native), CosineDecay
from tensorflow.keras.models import Sequential, Model # Added Model for Functional API
from tensorflow.keras.layers import (
    GRU, Dense, Dropout, Input, BatchNormalization,
    Attention, AdditiveAttention, TimeDistributed # Added Attention layers
)
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, Callback # Added Callback for custom LR logging
from tensorflow.keras.optimizers.schedules import CosineDecay # Added LR Scheduler
import logging


# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s', # Added funcName
    handlers=[
        logging.FileHandler('mega_sena_v3.log', mode='w'), # Changed log file name, overwrite mode
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try importing AdamW from addons, fallback to regular Adam if unavailable
try:
    # Check if TF version is >= 2.11 for native AdamW
    from packaging import version
    if version.parse(tf.__version__) >= version.parse("2.11.0"):
        AdamW_Optimizer = tf.keras.optimizers.AdamW
        logger.info("Using native tf.keras.optimizers.AdamW.")
    else:
        # Try addons if TF version is older
        import tensorflow_addons as tfa
        AdamW_Optimizer = tfa.optimizers.AdamW
        logger.info("Using AdamW optimizer from tensorflow-addons.")
except ImportError:
    logger.warning("TensorFlow >= 2.11 (for native AdamW) or tensorflow-addons not found. Falling back to tf.keras.optimizers.Adam.")
    AdamW_Optimizer = tf.keras.optimizers.Adam # Fallback
# ### MODIFICATION END V4 ###

import requests
from io import StringIO
import os
import warnings
import matplotlib.pyplot as plt
import logging
import json
from datetime import datetime, timedelta
import hashlib
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s', # Added funcName
    handlers=[
        logging.FileHandler('mega_sena_v4.log', mode='w'), # V4 log file
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Environment settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Configuração via Arquivo ---
def load_config(config_file='configv4.json'): # Using configv4.json
    """Carrega configurações V4 de um arquivo JSON."""
    default_config = {
        "data_url": "https://loteriascaixa-api.herokuapp.com/api/megasena",
        "data_file": None,
        "export_file": "historico_e_previsoes_megasena_v4.xlsx", # V4 export file
        "sequence_length": 20,
        "num_features_base": 60,
        "num_features_time": 60,
        "num_features_statistical": 187, # Calculated below
        "rolling_freq_windows": [10, 50, 100],
        "gru_units": 192,
        "use_attention": True,
        "attention_type": "keras_additive", # 'keras_additive', 'keras_multiplicative'
        "use_batch_norm": True,
        "dropout_rate": 0.4,
        "epochs": 250,
        "batch_size": 64,
        "optimizer_type": "AdamW", # AdamW or Adam
        "initial_learning_rate": 0.001,
        "lr_decay_steps": 10000,
        "lr_end_factor": 0.01,
        "adamw_weight_decay": 1e-4,
        "test_size_ratio": 0.15,
        "validation_split_ratio": 0.15,
        "cache_duration_hours": 24,
        "cache_dir": "cache",
        "tensorboard_log_dir": "logs/fit/"
    }
    # Calculate stat features count
    num_stat_features = 1 + 1 + 1 + 4 + len(default_config["rolling_freq_windows"]) * default_config["num_features_base"]
    default_config["num_features_statistical"] = num_stat_features

    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_loaded = json.load(f)
                for key, value in config_loaded.items():
                    if key == "lstm_units" and "gru_units" not in config_loaded: default_config["gru_units"] = value
                    elif key == "num_features" and "num_features_base" not in config_loaded: default_config["num_features_base"] = value
                    elif key == "num_features_statistical" and key in config_loaded: default_config[key] = value
                    elif key != "num_features_statistical" and key in default_config: default_config[key] = value
                    else: logger.warning(f"Ignoring unknown key '{key}' from {config_file}")
                if "rolling_freq_windows" in config_loaded and "num_features_statistical" not in config_loaded:
                     num_stat_features = 1 + 1 + 1 + 4 + len(default_config["rolling_freq_windows"]) * default_config["num_features_base"]
                     default_config["num_features_statistical"] = num_stat_features
                     logger.info(f"Recalculated num_features_statistical to {num_stat_features}")
                logger.info(f"Configurações carregadas e mescladas de {config_file}")
        else:
            logger.warning(f"Arquivo {config_file} não encontrado. Usando padrões V4.")
    except Exception as e:
        logger.error(f"Erro ao carregar config: {e}")

    default_config['num_features_total'] = (
        default_config['num_features_base'] +
        default_config['num_features_time'] +
        default_config['num_features_statistical']
    )
    logger.info(f"Total features calculated: {default_config['num_features_total']}")
    Path(default_config['tensorboard_log_dir']).mkdir(parents=True, exist_ok=True)
    return default_config

# Carrega configurações V4
config = load_config()

# --- Sistema de Cache ---
get_cache_key = lambda url: hashlib.md5(url.encode()).hexdigest()
def is_cache_valid(cache_file, duration_hours):
    if not os.path.exists(cache_file): return False
    return datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file)) < timedelta(hours=duration_hours)
def save_to_cache(data, cache_file):
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w') as f: json.dump(data, f)
        logger.info(f"Dados salvos no cache: {cache_file}")
    except Exception as e: logger.error(f"Erro ao salvar cache: {e}")
def load_from_cache(cache_file):
    try:
        with open(cache_file, 'r') as f: return json.load(f)
    except Exception as e: logger.error(f"Erro ao carregar cache: {e}"); return None

# --- Funções de Dados ---

def download_and_prepare_data(url=None, file_path=None):
    """Downloads/loads data, ensures 'BolaX' columns are present/numeric."""
    logger.info("Iniciando carregamento de dados...")
    df = None
    data = None # Initialize data

    # --- Cache and Download Logic ---
    if url:
        cache_key = get_cache_key(url)
        cache_file = os.path.join(config['cache_dir'], f"{cache_key}.json")
        if is_cache_valid(cache_file, config['cache_duration_hours']):
            logger.info("Carregando dados do cache...")
            data = load_from_cache(cache_file)
            if not data: logger.warning("Cache inválido ou corrompido. Baixando novamente."); data = None
            else: logger.info("Dados carregados com sucesso do cache.")
        if data is None:
             logger.info("Cache expirado/inválido/não encontrado. Baixando dados...")
             try:
                 response = requests.get(url, verify=False, timeout=60)
                 response.raise_for_status()
                 data = response.json()
                 save_to_cache(data, cache_file)
                 logger.info("Dados baixados e salvos no cache com sucesso.")
             except requests.exceptions.RequestException as e:
                 logger.error(f"Erro ao baixar dados: {e}")
                 if file_path and os.path.exists(file_path): logger.info(f"Tentando carregar do arquivo local {file_path}...")
                 else: return None
             except json.JSONDecodeError as json_err:
                 logger.error(f"Erro ao decodificar JSON: {json_err}"); return None

    # --- JSON Processing Logic ---
    if isinstance(data, list) and data:
        results, concursos, datas = [], [], []
        required_keys = {'dezenas', 'concurso', 'data'}
        for i, sorteio in enumerate(data):
            if not isinstance(sorteio, dict): logger.warning(f"Item {i} não é dict: {sorteio}"); continue
            if not required_keys.issubset(sorteio.keys()): logger.warning(f"Sorteio {sorteio.get('concurso', i)} com chaves ausentes ({required_keys - sorteio.keys()})."); continue
            try:
                dezenas_str = sorteio.get('dezenas', []);
                if not isinstance(dezenas_str, list): logger.warning(f"Dezenas inválidas (não lista) no sorteio {sorteio.get('concurso')}."); continue
                dezenas = sorted([int(d) for d in dezenas_str])
                if len(dezenas) == 6 and all(1 <= d <= 60 for d in dezenas):
                    results.append(dezenas); concursos.append(sorteio.get('concurso')); datas.append(sorteio.get('data'))
                else: logger.warning(f"Sorteio {sorteio.get('concurso')} inválido (número/valor dezenas): {dezenas}")
            except (ValueError, TypeError) as e: logger.warning(f"Erro processando sorteio {sorteio.get('concurso', i)}: {e}")
        if not results: logger.error("Nenhum sorteio válido encontrado (API/Cache)."); df = None
        else:
            df = pd.DataFrame(results, columns=[f'Bola{i+1}' for i in range(6)])
            if concursos: df['Concurso'] = concursos
            if datas:
                try: df['Data'] = pd.to_datetime(datas, format='%d/%m/%Y', errors='coerce'); logger.info("Coluna 'Data' convertida.")
                except Exception as e_date: logger.error(f"Erro convertendo 'Data': {e_date}")
            # Sorting
            sort_col = None
            if 'Concurso' in df.columns and pd.api.types.is_numeric_dtype(df['Concurso']): sort_col = 'Concurso'
            elif 'Data' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Data']): sort_col = 'Data'
            if sort_col: df = df.sort_values(by=sort_col).reset_index(drop=True); logger.info(f"Dados ordenados por '{sort_col}'.")
            else: logger.warning("Não foi possível ordenar automaticamente.")
            logger.info(f"Dados processados (API/Cache): {len(df)} sorteios.")
    elif data is not None: logger.error("Formato JSON inválido."); df = None

    # --- Local File Loading Logic ---
    if df is None and file_path and os.path.exists(file_path):
        logger.info(f"Tentando carregar do arquivo local: {file_path}")
        try:
            df_loaded = None
            for sep in [';', ',', '\t', '|']:
                 try:
                     df_try = pd.read_csv(file_path, sep=sep);
                     if df_try.shape[1] >= 6: logger.info(f"CSV lido (sep='{sep}')."); df_loaded = df_try; break
                 except Exception: continue
            if df_loaded is None:
                 try: df_loaded = pd.read_csv(file_path, sep=None, engine='python'); logger.info("CSV lido (detecção automática).")
                 except Exception as e_auto: logger.error(f"Falha na leitura CSV (auto): {e_auto}")
            if df_loaded is not None: df = df_loaded; logger.info(f"Dados carregados de {file_path}.")
            else: logger.error(f"Não foi possível ler {file_path}."); return None
        except Exception as e_file: logger.error(f"Erro ao carregar {file_path}: {e_file}"); return None

    # --- Final Check and Column Processing ---
    if df is None: logger.critical("Nenhuma fonte de dados funcionou."); return None
    bola_cols_found = [] # Find 'BolaX' columns (robust logic from V2/V3 reused)
    potential_patterns = [[f'Bola{i+1}' for i in range(6)], [f'bola{i+1}' for i in range(6)], [f'Dezena{i+1}' for i in range(6)], [f'dezena{i+1}' for i in range(6)], [f'N{i+1}' for i in range(6)], [f'n{i+1}' for i in range(6)]]
    df_cols_lower = {c.lower(): c for c in df.columns}
    for pattern_list in potential_patterns:
        if all(col in df.columns for col in pattern_list): bola_cols_found = pattern_list; logger.info(f"Colunas encontradas: {pattern_list}"); break
        elif all(col.lower() in df_cols_lower for col in pattern_list): bola_cols_found = [df_cols_lower[col.lower()] for col in pattern_list]; logger.info(f"Colunas encontradas (case-insensitive): {bola_cols_found}"); break
    if not bola_cols_found: # Heuristic if patterns fail
        logger.warning("Nenhum padrão de coluna. Tentando heurística..."); numeric_cols = df.select_dtypes(include=np.number).columns; potential_bola_cols = []
        for c in numeric_cols:
            try: numeric_col = pd.to_numeric(df[c], errors='coerce'); is_likely = numeric_col.between(1, 60, inclusive='both').all() and numeric_col.notna().all() and (numeric_col.fillna(-1) == numeric_col.fillna(-1).astype(int)).all();
            except: is_likely = False
            if is_likely: potential_bola_cols.append(c)
        if len(potential_bola_cols) >= 6: bola_cols_found = potential_bola_cols[:6]; logger.warning(f"Colunas identificadas heuristicamente: {bola_cols_found}. VERIFIQUE!")
        else: logger.error(f"Não foi possível identificar 6 colunas válidas (1-60). Colunas: {list(df.columns)}"); return None
    rename_map = {found_col: f'Bola{i+1}' for i, found_col in enumerate(bola_cols_found)}; df.rename(columns=rename_map, inplace=True)
    bola_cols = [f'Bola{i+1}' for i in range(6)]
    try: # Ensure numeric and handle NaNs
        for col in bola_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any(): logger.warning(f"Removendo {df[col].isnull().sum()} linhas com NaNs em '{col}'."); df.dropna(subset=[col], inplace=True)
            df[col] = df[col].astype(int)
        logger.info("Colunas das bolas verificadas/convertidas.")
    except Exception as e_num: logger.error(f"Erro convertendo colunas: {e_num}"); return None
    cols_to_keep = bola_cols + [col for col in ['Concurso', 'Data'] if col in df.columns]; final_df = df[cols_to_keep].copy()
    # Final sort check
    sort_col = None;
    if 'Concurso' in final_df.columns and pd.api.types.is_numeric_dtype(final_df['Concurso']): sort_col = 'Concurso'
    elif 'Data' in final_df.columns and pd.api.types.is_datetime64_any_dtype(final_df['Data']): sort_col = 'Data'
    if sort_col: final_df = final_df.sort_values(by=sort_col).reset_index(drop=True)
    else: logger.warning("Não foi possível garantir ordem cronológica final.")
    logger.info(f"Processamento final: {len(final_df)} sorteios.")
    if len(final_df) < config['sequence_length'] * 3: logger.critical(f"Dados insuficientes ({len(final_df)}) para seq={config['sequence_length']}."); return None
    return final_df

def preprocess_data_labels(df_balls_only, num_features_base):
    """Transforms winning numbers into MultiLabelBinarizer format (labels y)."""
    logger.info("Pré-processando labels (MultiLabelBinarizer)...")
    try:
        if df_balls_only.empty: logger.error("DataFrame vazio para labels."); return None, None, None
        required_cols = [f'Bola{i+1}' for i in range(6)]
        if not all(col in df_balls_only.columns for col in required_cols): logger.error(f"Colunas ausentes para labels: {[c for c in required_cols if c not in df_balls_only.columns]}"); return None, None, None
        balls_df = df_balls_only[required_cols].copy()
        invalid_rows_mask = ~balls_df.apply(lambda x: all(1 <= val <= num_features_base for val in x if pd.notna(val)), axis=1)
        if invalid_rows_mask.any(): logger.warning(f"Removendo {invalid_rows_mask.sum()} linhas inválidas."); balls_df = balls_df[~invalid_rows_mask]
        if balls_df.empty: logger.error("Nenhuma linha válida restante."); return None, None, None
        mlb = MultiLabelBinarizer(classes=list(range(1, num_features_base + 1)))
        encoded_data = mlb.fit_transform(balls_df.values.tolist())
        valid_indices = df_balls_only.index[~invalid_rows_mask]
        logger.info(f"Labels transformados: {encoded_data.shape}.")
        return encoded_data, mlb, valid_indices
    except Exception as e: logger.error(f"Erro pré-processando labels: {e}", exc_info=True); return None, None, None

def add_time_features(df_balls_only, num_features_base):
    """Calculates 'draws since last seen' for each number."""
    logger.info("Calculando features de tempo...")
    try:
        bola_cols = [f'Bola{i+1}' for i in range(6)]; draws = df_balls_only[bola_cols].values; num_draws = len(draws)
        time_features_list = []; last_seen_draw = {num: -1 for num in range(1, num_features_base + 1)}
        for i in range(num_draws):
            current_features = np.zeros(num_features_base); numbers_in_current_draw = set(draws[i])
            for num in range(1, num_features_base + 1): current_features[num - 1] = i + 1 if last_seen_draw[num] == -1 else i - last_seen_draw[num]
            time_features_list.append(current_features)
            for drawn_num in numbers_in_current_draw:
                if 1 <= drawn_num <= num_features_base: last_seen_draw[drawn_num] = i
        time_features_raw = np.array(time_features_list).astype(np.float32) # Ensure float32
        logger.info(f"Features de tempo calculadas. Shape: {time_features_raw.shape}")
        return time_features_raw
    except Exception as e: logger.error(f"Erro calculando features de tempo: {e}", exc_info=True); return None

def add_statistical_features(df_balls_only, num_features_base, rolling_windows):
    """Calculates statistical features (Parity, Sum, Range, Zones, Rolling Freq)."""
    logger.info(f"Calculando features estatísticas (Freq. {rolling_windows})...")
    try:
        bola_cols = [f'Bola{i+1}' for i in range(6)]; draws = df_balls_only[bola_cols].values; num_draws = len(draws)
        odd_counts, sums, ranges, zone_counts_list = [], [], [], []
        zone_defs = [(1, 15), (16, 30), (31, 45), (46, 60)]
        num_freq_features = len(rolling_windows) * num_features_base
        rolling_freq_features = np.zeros((num_draws, num_freq_features), dtype=np.float32)
        mlb_freq = MultiLabelBinarizer(classes=list(range(1, num_features_base + 1)))
        encoded_draws_freq = mlb_freq.fit_transform(draws.tolist())
        encoded_draws_df = pd.DataFrame(encoded_draws_freq, columns=mlb_freq.classes_)
        logger.debug("Calculando frequências rolantes...")
        freq_col_offset = 0
        for window in rolling_windows:
            rolling_sum_shifted = encoded_draws_df.rolling(window=window, min_periods=1).sum().shift(1).fillna(0)
            rolling_freq_features[:, freq_col_offset : freq_col_offset + num_features_base] = rolling_sum_shifted.values
            freq_col_offset += num_features_base
        logger.debug("Calculando estatísticas por sorteio...")
        for i in range(num_draws):
            current_numbers = draws[i]; odd_counts.append(np.sum(current_numbers % 2 != 0)); sums.append(np.sum(current_numbers))
            ranges.append(np.max(current_numbers) - np.min(current_numbers)); counts_in_zones = []
            for z_min, z_max in zone_defs: counts_in_zones.append(np.sum((current_numbers >= z_min) & (current_numbers <= z_max)))
            zone_counts_list.append(counts_in_zones)
        statistical_features_raw = np.concatenate([
            np.array(odd_counts).reshape(-1, 1), np.array(sums).reshape(-1, 1),
            np.array(ranges).reshape(-1, 1), np.array(zone_counts_list), rolling_freq_features
        ], axis=1).astype(np.float32)
        expected_cols = 1 + 1 + 1 + len(zone_defs) + num_freq_features
        logger.info(f"Features estatísticas calculadas. Shape: {statistical_features_raw.shape}")
        if statistical_features_raw.shape[1] != expected_cols: logger.error(f"Erro shape features estatísticas! Esperado {expected_cols}, obtido {statistical_features_raw.shape[1]}"); return None
        return statistical_features_raw
    except Exception as e: logger.error(f"Erro calculando features estatísticas: {e}", exc_info=True); return None

def split_data(encoded_labels, time_features_raw, statistical_features_raw,
               test_size_ratio, validation_split_ratio, sequence_length):
    """Splits, scales (separately), and creates sequences for train/val/test."""
    logger.info("Dividindo dados, escalando features (Tempo/Estatísticas)...")
    try:
        n_samples = len(encoded_labels)
        if not (n_samples == len(time_features_raw) == len(statistical_features_raw)): logger.error("Disparidade tamanho labels/features."); return [None] * 8
        test_split_index = int(n_samples * (1 - test_size_ratio))
        val_split_index = int(test_split_index * (1 - validation_split_ratio))
        train_indices, val_indices, test_indices = np.arange(val_split_index), np.arange(val_split_index, test_split_index), np.arange(test_split_index, n_samples)
        min_len_for_seq = sequence_length + 1
        if len(train_indices) < min_len_for_seq: logger.error(f"Treino ({len(train_indices)}) pequeno para seq={sequence_length}."); return [None] * 8
        if len(val_indices) < min_len_for_seq: logger.warning(f"Validação ({len(val_indices)}) pequena. Validação será vazia."); train_indices = np.arange(test_split_index); val_indices = np.array([], dtype=int)
        if len(test_indices) < min_len_for_seq: logger.warning(f"Teste ({len(test_indices)}) pequeno. Teste será vazio."); test_indices = np.array([], dtype=int)
        logger.info(f"Índices Brutos - Treino: {len(train_indices)}, Val: {len(val_indices)}, Teste: {len(test_indices)}")
        train_labels, val_labels, test_labels = encoded_labels[train_indices], encoded_labels[val_indices] if len(val_indices) > 0 else np.array([]), encoded_labels[test_indices] if len(test_indices) > 0 else np.array([])
        train_time_raw, val_time_raw, test_time_raw = time_features_raw[train_indices], time_features_raw[val_indices] if len(val_indices) > 0 else np.array([]), time_features_raw[test_indices] if len(test_indices) > 0 else np.array([])
        train_stat_raw, val_stat_raw, test_stat_raw = statistical_features_raw[train_indices], statistical_features_raw[val_indices] if len(val_indices) > 0 else np.array([]), statistical_features_raw[test_indices] if len(test_indices) > 0 else np.array([])
        logger.info("Ajustando Scalers (StandardScaler) no treino...")
        time_scaler, stat_scaler = StandardScaler(), StandardScaler()
        train_time_scaled = time_scaler.fit_transform(train_time_raw)
        train_stat_scaled = stat_scaler.fit_transform(train_stat_raw)
        logger.info("Escalando features de validação/teste...")
        val_time_scaled = time_scaler.transform(val_time_raw) if len(val_indices) > 0 else np.array([])
        test_time_scaled = time_scaler.transform(test_time_raw) if len(test_indices) > 0 else np.array([])
        val_stat_scaled = stat_scaler.transform(val_stat_raw) if len(val_indices) > 0 else np.array([])
        test_stat_scaled = stat_scaler.transform(test_stat_raw) if len(test_indices) > 0 else np.array([])
        logger.info("Criando sequências (Treino)...")
        X_train, y_train = create_sequences(train_labels, train_time_scaled, train_stat_scaled, sequence_length)
        logger.info("Criando sequências (Validação)...")
        X_val, y_val = create_sequences(val_labels, val_time_scaled, val_stat_scaled, sequence_length)
        logger.info("Criando sequências (Teste)...")
        X_test, y_test = create_sequences(test_labels, test_time_scaled, test_stat_scaled, sequence_length)
        logger.info(f"Shapes Finais: Treino X={X_train.shape if X_train.size>0 else 'Vazio'}, Val X={X_val.shape if X_val.size>0 else 'Vazio'}, Teste X={X_test.shape if X_test.size>0 else 'Vazio'}")
        return X_train, X_val, X_test, y_train, y_val, y_test, time_scaler, stat_scaler
    except Exception as e: logger.error(f"Erro split/scale/sequence: {e}", exc_info=True); return [None] * 8

def create_sequences(encoded_labels, time_features_scaled, statistical_features_scaled, sequence_length):
    """Creates sequences combining labels and scaled features."""
    if not all(a is not None and a.size > 0 for a in [encoded_labels, time_features_scaled, statistical_features_scaled]): return np.array([]), np.array([])
    n_samples_total = len(encoded_labels)
    if not (n_samples_total == len(time_features_scaled) == len(statistical_features_scaled)): logger.error("Inconsistência tamanho create_sequences."); return np.array([]), np.array([])
    if n_samples_total <= sequence_length: return np.array([]), np.array([])
    logger.debug(f"Criando sequências de {sequence_length} a partir de {n_samples_total} amostras...")
    try:
        num_sequences = n_samples_total - sequence_length
        num_features_base, num_features_time, num_features_stat = encoded_labels.shape[1], time_features_scaled.shape[1], statistical_features_scaled.shape[1]
        num_features_total = num_features_base + num_features_time + num_features_stat
        X = np.zeros((num_sequences, sequence_length, num_features_total), dtype=np.float32)
        y = np.zeros((num_sequences, num_features_base), dtype=encoded_labels.dtype)
        for i in range(num_sequences):
            seq_labels = encoded_labels[i : i + sequence_length]
            seq_time = time_features_scaled[i : i + sequence_length]
            seq_stat = statistical_features_scaled[i : i + sequence_length]
            X[i] = np.concatenate((seq_labels, seq_time, seq_stat), axis=-1)
            y[i] = encoded_labels[i + sequence_length]
        logger.debug(f"{len(X)} sequências criadas. Shape X: {X.shape}, y: {y.shape}")
        return X, y
    except Exception as e: logger.error(f"Erro criando sequências: {e}", exc_info=True); return np.array([]), np.array([])


# --- Modelo ---

def build_model(sequence_length, num_features_total, num_features_base, gru_units,
                dropout_rate, use_batch_norm, use_attention, attention_type):
    """ Constrói o modelo GRU V4 com Attention e Batch Normalization opcionais. """
    logger.info(f"Construindo modelo GRU V4: units={gru_units}, dropout={dropout_rate}, "
                f"batch_norm={use_batch_norm}, attention={use_attention} ({attention_type})")
    try:
        input_layer = Input(shape=(sequence_length, num_features_total), name="input_layer")
        x = input_layer
        if use_batch_norm: x = BatchNormalization(name="input_bn")(x)

        # GRU Layer 1
        x = GRU(gru_units, return_sequences=True, kernel_initializer='he_normal',
                recurrent_initializer='orthogonal', name="gru_1")(x)
        if use_batch_norm: x = BatchNormalization(name="bn_gru_1")(x)
        x = Dropout(dropout_rate, name="dropout_gru_1")(x)

        # GRU Layer 2
        last_gru_return_sequences = use_attention # Attention needs sequences
        gru_2_output = GRU(gru_units // 2, return_sequences=last_gru_return_sequences, kernel_initializer='he_normal',
                           recurrent_initializer='orthogonal', name="gru_2")(x)
        x = gru_2_output # Keep track of the output tensor
        if use_batch_norm: x = BatchNormalization(name="bn_gru_2")(x)
        x = Dropout(dropout_rate, name="dropout_gru_2")(x)

        # Attention Layer (Optional)
        if use_attention:
            logger.info(f"Adding {attention_type} Attention layer.")
            # gru_output already holds the output of the last dropout layer (x)
            if attention_type == 'keras_additive':
                attention_layer = AdditiveAttention(name="attention_layer")
                # Use gru_2_output (before BN/Dropout) as value? Or x (after)? Let's use x.
                context_vector = attention_layer([x, x]) # Query and Value are the sequence output
            elif attention_type == 'keras_multiplicative':
                attention_layer = Attention(name="attention_layer") # Use default multiplicative
                context_vector = attention_layer([x, x])
            else:
                logger.warning(f"Unknown attention_type '{attention_type}'. Skipping attention.")
                # If attention is skipped or fails, ensure x is the correct tensor
                # If last GRU had return_sequences=False (because no attention), x would be (batch, units)
                # If last GRU had return_sequences=True, x would be (batch, seq_len, units) - need reduction (e.g., GlobalAvgPool1D)
                if last_gru_return_sequences: # If GRU returned sequences but attention failed
                     logger.warning("Attention skipped, but GRU returned sequences. Applying GlobalAveragePooling1D.")
                     from tensorflow.keras.layers import GlobalAveragePooling1D
                     x = GlobalAveragePooling1D()(x)
                # If last_gru_return_sequences was already False, x is fine.
                context_vector = x # Use the output of the last Dropout/GRU
            current_output = context_vector # The context vector is the input for Dense layers
        else:
            # If no attention, the output of the last GRU (with return_sequences=False) is used
             current_output = x # x already holds the correct output tensor

        # Dense Layers
        dense_output = Dense(gru_units // 2, activation='relu', name="dense_1")(current_output)
        if use_batch_norm: dense_output = BatchNormalization(name="bn_dense_1")(dense_output)
        dense_output = Dropout(dropout_rate, name="dropout_dense_1")(dense_output)

        # Output Layer
        final_output = Dense(num_features_base, activation='sigmoid', name="output_layer")(dense_output)

        # Define the Model using Functional API
        model = Model(inputs=input_layer, outputs=final_output, name="Modelo_GRU_MegaSena_V4")

        # --- Optimizer and LR Schedule ---
        logger.info(f"Using Optimizer: {config['optimizer_type']}, Initial LR: {config['initial_learning_rate']}")
        lr_schedule = CosineDecay(
            initial_learning_rate=config['initial_learning_rate'],
            decay_steps=config['lr_decay_steps'],
            alpha=config['lr_end_factor']
        )
        logger.info(f"LR Schedule: CosineDecay over {config['lr_decay_steps']} steps, end factor {config['lr_end_factor']}.")

        if config['optimizer_type'].lower() == 'adamw':
             optimizer = AdamW_Optimizer(learning_rate=lr_schedule, weight_decay=config['adamw_weight_decay'])
             logger.info(f"AdamW Weight Decay: {config['adamw_weight_decay']}")
        else:
             if config['optimizer_type'].lower() != 'adam': logger.warning(f"Optimizer '{config['optimizer_type']}' não reconhecido. Usando Adam.")
             optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['binary_accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        model.summary(print_fn=logger.info)
        return model

    except Exception as e:
        logger.error(f"Erro ao construir o modelo GRU V4: {e}", exc_info=True)
        return None

# Custom Callback for LR Logging
class LearningRateLogger(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self.model.optimizer, 'learning_rate'):
            try:
                # Get current LR value
                lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
                # If it's a schedule, calculate based on current step
                if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                    current_step = tf.keras.backend.get_value(self.model.optimizer.iterations)
                    lr = lr(current_step)
                logger.info(f"Época {epoch + 1} Início - LR Atual: {lr:.7f}")
            except Exception as e:
                logger.warning(f"Não foi possível obter LR no início da época: {e}")

    def on_epoch_end(self, epoch, logs=None):
         logs = logs or {}
         log_str = f"Época {epoch + 1} Fim - Loss: {logs.get('loss', -1):.4f}"
         if 'binary_accuracy' in logs: log_str += f" - Acc: {logs.get('binary_accuracy', -1):.4f}"
         if 'auc' in logs: log_str += f" - AUC: {logs.get('auc', -1):.4f}"
         if 'val_loss' in logs:
             log_str += f" - Val Loss: {logs.get('val_loss', -1):.4f}"
             if 'val_binary_accuracy' in logs: log_str += f" - Val Acc: {logs.get('val_binary_accuracy', -1):.4f}"
             if 'val_auc' in logs: log_str += f" - Val AUC: {logs.get('val_auc', -1):.4f}"
         else: log_str += " (Sem dados de validação)"
         logger.info(log_str)


def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, log_dir):
    """ Treina o modelo GRU V4 com callbacks (incl TensorBoard) e LR Scheduler. """
    logger.info("Iniciando o treinamento do modelo GRU V4...")
    try:
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True, verbose=1, mode='min')
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)
        lr_logger = LearningRateLogger() # Custom LR logger

        validation_data = (X_val, y_val) if X_val.size > 0 and y_val.size > 0 else None
        if validation_data is None: logger.warning("Conjunto de validação vazio. EarlyStopping monitorará 'loss'."); early_stopping.monitor = 'loss'

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[early_stopping, tensorboard_callback, lr_logger], # Use custom LR logger
            verbose=0 # Suppress default Keras progress bar
        )
        logger.info("Treinamento concluído.")
        return history
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {e}", exc_info=True)
        return None


# --- Avaliação e Previsão ---

def evaluate_real_hits(model, X_test, y_test, batch_size=32):
    """Evaluates how many of the top 6 predicted numbers were actually drawn."""
    logger.info("Avaliando acertos reais (Top 6)...")
    try:
        if model is None or X_test is None or y_test is None or len(X_test) == 0: logger.error("Dados inválidos/vazios para avaliação hits."); return None
        y_pred_probs = model.predict(X_test, batch_size=batch_size)
        if y_pred_probs.shape[0]!= y_test.shape[0] or y_pred_probs.shape[1]!= y_test.shape[1]: logger.error(f"Shape mismatch: Preds({y_pred_probs.shape}) vs Test({y_test.shape})"); return None
        hits_per_draw, detailed_hits = [], []
        for i in range(len(y_pred_probs)):
            top6_pred_indices = np.argsort(y_pred_probs[i])[-6:]; predicted_numbers = sorted((top6_pred_indices + 1).tolist())
            actual_winning_indices = np.where(y_test[i] == 1)[0]; actual_numbers = sorted((actual_winning_indices + 1).tolist())
            hits = set(predicted_numbers) & set(actual_numbers); num_hits = len(hits)
            detailed_hits.append({ 'sorteio_index_teste': i, 'previstos': predicted_numbers, 'sorteados': actual_numbers, 'acertos': sorted(list(hits)), 'num_acertos': num_hits })
            hits_per_draw.append(num_hits)
        if not hits_per_draw: avg_hits, max_hits, hits_distribution = 0, 0, {}; logger.warning("Nenhum sorteio no teste para stats hits.")
        else: avg_hits = np.mean(hits_per_draw); max_hits = np.max(hits_per_draw); hits_distribution = {i: hits_per_draw.count(i) for i in range(max_hits + 1)}
        logger.info("-" * 60 + "\nANÁLISE DE ACERTOS REAIS (TOP 6 PREVISTOS vs SORTEADOS)\n" + "-" * 60)
        logger.info(f"Total sorteios teste: {len(hits_per_draw)}")
        if hits_per_draw:
             logger.info(f"Média acertos: {avg_hits:.3f}, Máximo acertos: {max_hits}")
             logger.info("Distribuição:")
             for hits_count, count in hits_distribution.items():
                 if count > 0: logger.info(f"  {hits_count} acerto(s): {count} ({(count / len(hits_per_draw)) * 100:.1f}%)")
             logger.info(f"Exemplo últimos {min(5, len(hits_per_draw))} sorteios teste:")
             for hit_detail in detailed_hits[-min(5, len(hits_per_draw)):]: logger.info(f"  Idx {hit_detail['sorteio_index_teste']}: Prev{hit_detail['previstos']} Real{hit_detail['sorteados']} -> Hits {hit_detail['acertos']} ({hit_detail['num_acertos']})")
        logger.info("-" * 60 + "\nAVISO: Acertos passados NÃO garantem acertos futuros.\n" + "-" * 60)
        return { 'hits_per_draw': hits_per_draw, 'avg_hits': avg_hits, 'max_hits': max_hits, 'hits_distribution': hits_distribution, 'detailed_hits': detailed_hits }
    except Exception as e: logger.error(f"Erro avaliando acertos reais: {e}", exc_info=True); return None

def evaluate_model(model, X_test, y_test, batch_size=32):
    """Evaluates the model on the test set using standard metrics and real hits."""
    logger.info("Avaliando modelo final no teste...")
    try:
        if model is None or X_test is None or y_test is None or len(X_test) == 0: logger.error("Dados inválidos/vazios para avaliação final."); return None
        logger.info("Calculando métricas básicas...")
        results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        basic_metrics_dict = dict(zip(model.metrics_names, results))
        logger.info("Calculando acertos reais (Top 6)...")
        real_hits_results = evaluate_real_hits(model, X_test, y_test, batch_size)
        if real_hits_results is None: logger.error("Falha avaliação acertos reais.")
        logger.info("-" * 60 + "\nResultados Avaliação Teste\n" + "-" * 60)
        logger.info("Métricas Padrão:"); [logger.info(f"  - {name}: {value:.4f}") for name, value in basic_metrics_dict.items()]
        logger.info("\nStats Acertos Reais (Top 6):")
        if real_hits_results:
             logger.info(f"  - Média: {real_hits_results['avg_hits']:.3f}, Máx: {real_hits_results['max_hits']}")
             logger.info("  Distribuição:"); total_draws = len(real_hits_results.get('hits_per_draw', []))
             if total_draws > 0:
                 hits_dist = real_hits_results.get('hits_distribution', {})
                 for hc, count in hits_dist.items():
                      if count > 0: logger.info(f"    * {hc} acerto(s): {count} ({(count/total_draws)*100:.1f}%)")
             else: logger.info("    N/A (sem sorteios)")
        else: logger.info("  N/A (falha)")
        logger.info("-" * 60 + "\nLembrete: Métricas refletem passado.\n" + "-" * 60)
        return { 'basic_metrics': basic_metrics_dict, 'real_hits': real_hits_results }
    except Exception as e: logger.error(f"Erro avaliação final: {e}", exc_info=True); return None

def predict_next_draw(model, last_sequence_labels, last_sequence_time_raw, last_sequence_stat_raw,
                      time_scaler, stat_scaler, mlb, num_predictions=6):
    """Prepares final sequence (scaling features) and predicts next draw."""
    logger.info("Preparando sequência e prevendo PRÓXIMO sorteio...")
    try:
        if model is None: logger.error("Modelo inválido."); return None, None
        if time_scaler is None or stat_scaler is None: logger.error("Scalers inválidos."); return None, None
        seq_len = config['sequence_length']
        if not all(s is not None and len(s) == seq_len for s in [last_sequence_labels, last_sequence_time_raw, last_sequence_stat_raw]): logger.error(f"Sequência inválida/tamanho incorreto (esperado {seq_len})."); return None, None
        logger.debug("Escalando features última sequência...")
        last_sequence_time_scaled = time_scaler.transform(last_sequence_time_raw)
        last_sequence_stat_scaled = stat_scaler.transform(last_sequence_stat_raw)
        last_sequence_combined = np.concatenate((last_sequence_labels, last_sequence_time_scaled, last_sequence_stat_scaled), axis=-1).astype(np.float32)
        last_sequence_batch = np.expand_dims(last_sequence_combined, axis=0)
        logger.info(f"Shape sequência final previsão: {last_sequence_batch.shape}")
        predicted_probabilities = model.predict(last_sequence_batch)[0]
        expected_output_shape = config['num_features_base']
        if predicted_probabilities.shape[0] != expected_output_shape: logger.error(f"Shape previsão inesperado: {predicted_probabilities.shape}. Esperado: ({expected_output_shape},)"); return None, None
        predicted_indices = np.argsort(predicted_probabilities)[-num_predictions:]
        predicted_numbers = sorted((predicted_indices + 1).tolist())
        confidence_scores = predicted_probabilities[predicted_indices]
        avg_conf, max_conf, min_conf = (np.mean(confidence_scores), np.max(confidence_scores), np.min(confidence_scores)) if confidence_scores.size > 0 else (0,0,0)
        logger.info("-" * 50 + f"\nPrevisão {num_predictions} números: {predicted_numbers}")
        logger.info(f"Confiança Média: {avg_conf:.4f}, Máx: {max_conf:.4f}, Mín: {min_conf:.4f}")
        logger.info("Probabilidades individuais (previstos):")
        sorted_pred_indices = predicted_indices[np.argsort(confidence_scores)[::-1]]
        for num_idx in sorted_pred_indices: logger.info(f"  - Número {num_idx + 1}: {predicted_probabilities[num_idx]:.4f}")
        logger.info("-" * 50 + "\nAVISO CRÍTICO: Previsão experimental. NÃO HÁ GARANTIA.\n" + "-" * 50)
        return predicted_numbers, predicted_probabilities
    except Exception as e: logger.error(f"Erro previsão próximo sorteio: {e}", exc_info=True); return None, None


# --- Visualização e Exportação ---

def plot_training_history(history, filename='training_history_v4.png'):
    """ Plots training history (Loss, Accuracy, AUC, LR). """
    logger.info(f"Gerando gráficos histórico treino em {filename}...")
    try:
        if history is None or not history.history: logger.error("Histórico treino inválido."); return
        plt.figure(figsize=(15, 10)); history_dict = history.history
        metrics = [m for m in ['loss', 'binary_accuracy', 'auc'] if m in history_dict]
        num_plots = len(metrics) + (1 if 'lr' in history_dict else 0); num_cols = 2; num_rows = (num_plots + num_cols - 1) // num_cols
        plot_index = 1
        for metric in metrics:
            plt.subplot(num_rows, num_cols, plot_index); plt.plot(history_dict[metric], label=f'Treino {metric.capitalize()}')
            if f'val_{metric}' in history_dict: plt.plot(history_dict[f'val_{metric}'], label=f'Validação {metric.capitalize()}')
            plt.title(f'{metric.replace("_", " ").capitalize()} por Época'); plt.xlabel('Época'); plt.ylabel(metric.capitalize()); plt.legend(); plt.grid(True); plot_index += 1
        if 'lr' in history_dict: # Check if LR was logged (might not be if training stopped early)
             plt.subplot(num_rows, num_cols, plot_index); plt.plot(history_dict['lr'], label='Taxa Aprendizado')
             plt.title('Taxa Aprendizado'); plt.xlabel('Época'); plt.ylabel('Learning Rate'); plt.legend(); plt.grid(True);
        plt.tight_layout(); plt.savefig(filename); logger.info(f"Gráficos salvos: '{filename}'"); plt.close()
    except Exception as e: logger.error(f"Erro plotando treino: {e}", exc_info=True)

def plot_prediction_analysis(predicted_numbers, predicted_probabilities, df_full_valid, sequence_length, filename='prediction_analysis_v4.png'):
    """ Generates visual analysis of predictions vs recent frequency. """
    logger.info(f"Gerando análise visual previsões em {filename}...")
    try:
        if predicted_numbers is None or predicted_probabilities is None: logger.error("Dados previsão inválidos."); return
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
        if number_freq.max() > 0: plt.yticks(np.arange(0, number_freq.max() + 1, max(1, int(number_freq.max()//5)))) # Adjust yticks
        # Plot 4: Frequency vs Probability
        plt.subplot(2, 2, 4); plt.scatter(number_freq, predicted_probabilities, alpha=0.6); freq_for_predicted = number_freq[predicted_numbers_arr - 1]
        plt.scatter(freq_for_predicted, probs_for_predicted, color='red', s=80, label='Previstos', edgecolors='black'); plt.title('Frequência Recente vs Probabilidade Prevista')
        plt.xlabel(f'Frequência Últimos {sequence_length} Sorteios'); plt.ylabel('Probabilidade Prevista'); plt.grid(True, alpha=0.7); plt.legend()
        for i, num in enumerate(predicted_numbers_arr): plt.text(freq_for_predicted[i] + 0.05 * (number_freq.max() if number_freq.max()>0 else 1), probs_for_predicted[i], str(num), fontsize=9)
        plt.tight_layout(); plt.savefig(filename); logger.info(f"Análise visual salva: '{filename}'"); plt.close()
    except Exception as e: logger.error(f"Erro plotando previsão: {e}", exc_info=True)

def plot_hits_over_time(model, X_test, y_test, mlb, filename='hits_over_time_v4.png'):
    """ Plots number of hits (top 6 predicted vs actual) over the test set. """
    logger.info(f"Gerando gráfico acertos no teste em {filename}...")
    if X_test is None or y_test is None or X_test.shape[0] == 0: logger.warning("Dados teste insuficientes p/ plot acertos."); return None
    logger.info("Calculando acertos no teste para plotagem...")
    try:
        y_pred_probs_test = model.predict(X_test, batch_size=config['batch_size']); hits_per_draw = []
        for i in range(len(y_pred_probs_test)):
            top6_pred_indices = np.argsort(y_pred_probs_test[i])[-6:]
            actual_winning_indices = np.where(y_test[i] == 1)[0]
            num_hits = len(set(top6_pred_indices) & set(actual_winning_indices))
            hits_per_draw.append(num_hits)
        if not hits_per_draw: logger.warning("Nenhum resultado acerto calculado."); return None
        plt.figure(figsize=(15, 6)); plt.plot(hits_per_draw, marker='o', linestyle='-', markersize=4, label='Nº Acertos (Top 6) / Sorteio Teste')
        if len(hits_per_draw) >= 10:
             rolling_avg = pd.Series(hits_per_draw).rolling(window=10, min_periods=1).mean()
             plt.plot(rolling_avg, linestyle='--', color='red', label='Média Móvel (10)')
        plt.xlabel("Índice Sorteio Teste"); plt.ylabel("Número Acertos (Top 6)"); plt.title("Acertos Modelo no Teste Histórico")
        plt.yticks(np.arange(0, 7, 1)); plt.ylim(bottom=-0.2); plt.grid(True, alpha=0.7); plt.legend()
        plt.savefig(filename); logger.info(f"Gráfico acertos salvo: '{filename}'"); plt.close()
        return hits_per_draw
    except Exception as e: logger.error(f"Erro plotando acertos: {e}", exc_info=True); return None

def export_results(df_full_valid, predicted_numbers, predicted_probabilities, evaluation_results, config):
    """ Exports history, predictions, evaluation metrics, and config to Excel. """
    logger.info(f"Exportando resultados para Excel ({config['export_file']})...")
    try:
        if df_full_valid is None: logger.error("DataFrame histórico inválido."); return
        if predicted_numbers is None or predicted_probabilities is None: logger.error("Dados previsão inválidos."); return
        evaluation_results = evaluation_results or {'basic_metrics': {}, 'real_hits': None} # Ensure structure
        # Prepare DataFrames for Excel sheets (reuse V3 logic)
        predictions_df = pd.DataFrame({'Número': range(1, config['num_features_base'] + 1), 'Probabilidade_Prevista': predicted_probabilities})
        predictions_df['Previsto_Top_6'] = predictions_df['Número'].isin(predicted_numbers); predictions_df = predictions_df.sort_values('Probabilidade_Prevista', ascending=False).reset_index(drop=True)
        basic_metrics_dict = evaluation_results.get('basic_metrics', {}); metrics_df = pd.DataFrame({'Métrica': list(basic_metrics_dict.keys()) if basic_metrics_dict else ['N/A'], 'Valor': [f"{v:.5f}" for v in basic_metrics_dict.values()] if basic_metrics_dict else ['N/A'] })
        real_hits_results = evaluation_results.get('real_hits'); hits_summary_df, hits_dist_df, detailed_hits_df = None, None, None # Init
        if real_hits_results and 'hits_distribution' in real_hits_results:
            hits_dist = real_hits_results['hits_distribution']; total_draws = len(real_hits_results.get('hits_per_draw', [])); avg_hits = real_hits_results.get('avg_hits', 'N/A'); max_hits = real_hits_results.get('max_hits', 'N/A')
            hits_summary_df = pd.DataFrame({'Estatística': ['Média Acertos (Top 6)', 'Máx Acertos (Top 6)', 'Total Sorteios Teste'], 'Valor': [f"{avg_hits:.3f}" if isinstance(avg_hits, float) else avg_hits, f"{max_hits}", total_draws]})
            hits_dist_df = pd.DataFrame({'Número Acertos': list(hits_dist.keys()), 'Qtd Sorteios': list(hits_dist.values())})
            hits_dist_df['Porcentagem (%)'] = hits_dist_df['Qtd Sorteios'].apply(lambda c: f"{(c / total_draws) * 100:.1f}%" if total_draws > 0 else 'N/A'); hits_dist_df = hits_dist_df.sort_values('Número Acertos').reset_index(drop=True)
            detailed_hits = real_hits_results.get('detailed_hits')
            if detailed_hits: detailed_hits_df = pd.DataFrame(detailed_hits); [detailed_hits_df[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x) for col in ['previstos', 'sorteados', 'acertos'] if col in detailed_hits_df.columns]
            else: detailed_hits_df = pd.DataFrame({'Info': ['Detalhes acertos não disponíveis']})
        else: logger.warning("Resultados acertos reais não disponíveis/incompletos."); hits_summary_df = pd.DataFrame({'Estatística': ['N/A'], 'Valor': ['N/A']}); hits_dist_df = pd.DataFrame({'Número Acertos': ['N/A'], 'Qtd Sorteios': ['N/A'], 'Porcentagem (%)': ['N/A']}); detailed_hits_df = pd.DataFrame({'Info': ['Detalhes acertos não disponíveis']})
        # Write to Excel
        logger.info("Escrevendo abas Excel...")
        with pd.ExcelWriter(config['export_file'], engine='openpyxl') as writer:
            predictions_df.to_excel(writer, sheet_name='Previsao_Probs', index=False)
            metrics_df.to_excel(writer, sheet_name='Metricas_Avaliacao', index=False)
            if hits_summary_df is not None: hits_summary_df.to_excel(writer, sheet_name='Sumario_Acertos', index=False)
            if hits_dist_df is not None: hits_dist_df.to_excel(writer, sheet_name='Distr_Acertos', index=False)
            if detailed_hits_df is not None: detailed_hits_df.to_excel(writer, sheet_name='Detalhes_Acertos', index=False)
            df_full_valid.to_excel(writer, sheet_name='Historico_Utilizado', index=False)
            config_df = pd.DataFrame(list(config.items()), columns=['Parametro', 'Valor']); config_df.to_excel(writer, sheet_name='Configuracao_Usada', index=False)
        logger.info(f"Resultados exportados: '{config['export_file']}'")
    except PermissionError: logger.error(f"Erro permissão escrevendo '{config['export_file']}'.")
    except Exception as e: logger.error(f"Erro exportando Excel: {e}", exc_info=True)

def validate_config(config):
    """ Valida as configurações V4. """
    logger.info("Validando configuração V4...")
    is_valid = True; required = [ 'data_url', 'data_file', 'export_file', 'sequence_length', 'num_features_base', 'num_features_time','num_features_statistical', 'num_features_total', 'rolling_freq_windows', 'gru_units', 'use_attention','attention_type', 'use_batch_norm', 'dropout_rate', 'epochs', 'batch_size', 'optimizer_type','initial_learning_rate', 'lr_decay_steps', 'lr_end_factor', 'adamw_weight_decay', 'test_size_ratio','validation_split_ratio', 'cache_duration_hours', 'cache_dir', 'tensorboard_log_dir' ]
    try:
        for field in required:
            if field not in config: logger.error(f"Campo obrigatório ausente: {field}"); is_valid = False
        if not is_valid: return False
        # V4 Checks
        if not isinstance(config['use_attention'], bool): logger.error("use_attention invalido"); is_valid = False
        if config['use_attention'] and config['attention_type'] not in ['keras_additive', 'keras_multiplicative']: logger.error("attention_type invalido"); is_valid = False
        if not isinstance(config['initial_learning_rate'], (int, float)) or config['initial_learning_rate'] <= 0: logger.error("initial_learning_rate invalido"); is_valid = False
        if not isinstance(config['lr_decay_steps'], int) or config['lr_decay_steps'] <= 0: logger.error("lr_decay_steps invalido"); is_valid = False
        if not isinstance(config['lr_end_factor'], (int, float)) or not 0 < config['lr_end_factor'] <= 1: logger.error("lr_end_factor invalido"); is_valid = False
        if not isinstance(config['adamw_weight_decay'], (int, float)) or config['adamw_weight_decay'] < 0: logger.error("adamw_weight_decay invalido"); is_valid = False
        # Inherited Checks (simplified for brevity, assume V3 checks were comprehensive)
        if not isinstance(config['sequence_length'], int) or config['sequence_length'] < 1: logger.error("sequence_length invalido"); is_valid = False
        if config['num_features_total'] != config['num_features_base'] + config['num_features_time'] + config['num_features_statistical']: logger.error("num_features_total != soma"); is_valid = False
        if (config['test_size_ratio'] + config['validation_split_ratio']) >= 1.0: logger.error("Soma test+validation ratio >= 1"); is_valid = False
        if config.get('data_url') is None and config.get('data_file') is None: logger.error("Necessário 'data_url' ou 'data_file'."); is_valid = False
        # ... add other critical checks if needed ...
        if is_valid: logger.info("Configuração V4 validada.")
        else: logger.error("Validação V4 falhou.")
        return is_valid
    except Exception as e: logger.error(f"Erro validação config V4: {e}", exc_info=True); return False


# --- Fluxo Principal (Main) ---
def main():
    """ Função principal do programa V4. """
    run_start_time = datetime.now()
    logger.info(f"Iniciando Mega-Sena V4 em {run_start_time.strftime('%Y-%m-%d %H:%M:%S')}...")
    try:
        global config; config = load_config()
        if not validate_config(config): logger.critical("Configuração inválida. Abortando."); return

        # --- Pipeline ---
        logger.info("Etapa 1: Download/Preparação Dados...")
        df_full = download_and_prepare_data(url=config['data_url'], file_path=config['data_file'])
        if df_full is None: logger.critical("Falha Etapa 1. Abortando."); return

        logger.info("Etapa 2: Pré-processamento Labels...")
        bola_cols = [f'Bola{i+1}' for i in range(6)]
        encoded_labels, mlb, valid_indices = preprocess_data_labels(df_full[bola_cols], config['num_features_base'])
        if encoded_labels is None: logger.critical("Falha Etapa 2. Abortando."); return
        df_full_valid = df_full.loc[valid_indices].reset_index(drop=True)

        logger.info("Etapa 3a: Cálculo Features Tempo...")
        time_features_raw = add_time_features(df_full_valid[bola_cols], config['num_features_base'])
        if time_features_raw is None: logger.critical("Falha Etapa 3a. Abortando."); return

        logger.info("Etapa 3b: Cálculo Features Estatísticas...")
        statistical_features_raw = add_statistical_features(df_full_valid[bola_cols], config['num_features_base'], config['rolling_freq_windows'])
        if statistical_features_raw is None: logger.critical("Falha Etapa 3b. Abortando."); return

        logger.info("Etapa 4: Divisão/Escalonamento/Sequenciamento...")
        X_train, X_val, X_test, y_train, y_val, y_test, time_scaler, stat_scaler = split_data(
            encoded_labels, time_features_raw, statistical_features_raw,
            config['test_size_ratio'], config['validation_split_ratio'], config['sequence_length'] )
        if X_train is None or time_scaler is None or stat_scaler is None: logger.critical("Falha Etapa 4. Abortando."); return
        if X_train.size == 0: logger.critical("Conjunto Treino Vazio! Abortando."); return

        logger.info("Etapa 5: Construção Modelo GRU V4...")
        model = build_model( # V4 build
            config['sequence_length'], config['num_features_total'], config['num_features_base'],
            config['gru_units'], config['dropout_rate'], config['use_batch_norm'],
            config['use_attention'], config['attention_type'] )
        if model is None: logger.critical("Falha Etapa 5. Abortando."); return

        logger.info("Etapa 6: Treinamento Modelo V4...")
        log_dir = os.path.join(config['tensorboard_log_dir'], datetime.now().strftime("%Y%m%d-%H%M%S"))
        history = train_model( model, X_train, y_train, X_val, y_val, # V4 train
                               config['epochs'], config['batch_size'], log_dir)
        if history is None: logger.critical("Falha Etapa 6. Abortando."); return

        logger.info("Etapa 7: Avaliação Modelo...")
        evaluation_results = {'basic_metrics': {}, 'real_hits': None} # Default empty results
        if X_test.size > 0 and y_test.size > 0:
             eval_res = evaluate_model(model, X_test, y_test, config['batch_size'])
             if eval_res is None: logger.warning("Falha Etapa 7. Continuando sem avaliação.")
             else: evaluation_results = eval_res
        else: logger.warning("Conjunto teste vazio. Pulando avaliação.")

        logger.info("Etapa 8: Previsão Próximo Sorteio...")
        last_seq_labels = encoded_labels[-config['sequence_length']:]
        last_seq_time_raw = time_features_raw[-config['sequence_length']:]
        last_seq_stat_raw = statistical_features_raw[-config['sequence_length']:]
        predicted_numbers, predicted_probabilities = predict_next_draw( # V4 predict (uses scalers)
             model, last_seq_labels, last_seq_time_raw, last_seq_stat_raw,
             time_scaler, stat_scaler, mlb )
        if predicted_numbers is None: logger.critical("Falha Etapa 8. Abortando."); return

        logger.info("Etapa 9: Visualizações...")
        plot_training_history(history)
        plot_prediction_analysis(predicted_numbers, predicted_probabilities, df_full_valid, config['sequence_length'])
        if X_test.size > 0: plot_hits_over_time(model, X_test, y_test, mlb)
        else: logger.info("Pulando gráfico acertos (teste vazio).")

        logger.info("Etapa 10: Exportação Resultados...")
        export_results(df_full_valid, predicted_numbers, predicted_probabilities, evaluation_results, config)

        run_end_time = datetime.now(); logger.info("-" * 60 + f"\nProcesso V4 Concluído: {run_end_time - run_start_time}\n" + "-" * 60)
        logger.info(f"Log: mega_sena_v4.log | Excel: {config['export_file']}")
        logger.info("Gráficos: training_history_v4.png, prediction_analysis_v4.png, hits_over_time_v4.png")
        logger.info(f"Logs TensorBoard: {log_dir}")
        logger.info("-" * 60 + "\nLembre-se: Resultados experimentais. Jogue com responsabilidade.\n" + "-" * 60)

    except Exception as e:
        logger.critical(f"Erro GERAL não tratado no main V4: {e}", exc_info=True)
        return

if __name__ == "__main__":
    # Check dependencies: pip install tensorflow pandas numpy requests matplotlib openpyxl scikit-learn packaging tensorflow-addons (optional, for older TF)
    main()