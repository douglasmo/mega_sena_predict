# -*- coding: utf-8 -*-
"""
Script de Exemplo para "Previsão" da Mega-Sena - Versão V3.1
MODIFIED: Added Statistical Features (Parity, Sum, Range, Zones, Rolling Freq).
          Uses separate scalers for time and statistical features.
MODIFIED: Implemented Complex Time Features (Interval, Mean Interval, Std Dev Interval).
MODIFIED: Corrected dynamic feature count updates and validation flow, especially around hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler, StandardScaler
# import joblib # Optional: if you want to save/load scalers
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
    # Assume hyperparameter_tuning.py exists in the same directory
    from hyperparameter_tuning import HyperparameterTuner
    hyperparameter_tuning_available = True
except ImportError:
    hyperparameter_tuning_available = False
    warnings.warn("Módulo de otimização de hiperparâmetros (hyperparameter_tuning.py) não encontrado. A funcionalidade de teste de hiperparâmetros não estará disponível.")

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
                # Try writing with default encoding (hopefully UTF-8)
                self.stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                # Fallback to ASCII with replacement if UTF-8 fails on stdout
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
    # ### MODIFICATION START V3.1 ###
    # Removed num_features_time from defaults, it's calculated dynamically
    default_config = {
        "data_url": "https://loteriascaixa-api.herokuapp.com/api/megasena",
        "data_file": None,
        "export_file": os.path.join(output_dir, "historico_e_previsoes_megasena_v3.xlsx"),
        "sequence_length": 15,
        "num_features_base": 60,
        # "num_features_time": ..., # Removed: Calculated dynamically later
        "num_features_statistical": 187, # Calculated below (or loaded)
        "rolling_freq_windows": [10, 50, 100],
        "gru_units": 192,
        "use_batch_norm": True,
        "dropout_rate": 0.4,
        "epochs": 200,
        "batch_size": 64,
        "test_size_ratio": 0.15,
        "validation_split_ratio": 0.15,
        "cache_duration_hours": 24,
        "cache_dir": os.path.join(output_dir, "cache"),
        "tensorboard_log_dir": os.path.join(output_dir, "logs/fit/"),
        "test_hyperparameters": False, # Default to False
        "hyperparameter_search": None # Default placeholder
    }
    # ### MODIFICATION END V3.1 ###

    # Calculate statistical features count based on windows
    num_stat_features = 1 + 1 + 1 + 4 + len(default_config["rolling_freq_windows"]) * default_config["num_features_base"]
    default_config["num_features_statistical"] = num_stat_features

    try:
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f: # Added encoding
                config_loaded = json.load(f)
                logger.info(f"Configurações carregadas de {config_file}")

                # Update default config with loaded values, handling potential renames/new keys
                # Use loop over loaded keys to handle additions/overrides
                for key, value in config_loaded.items():
                    if key == "lstm_units" and "gru_units" not in config_loaded:
                        default_config["gru_units"] = value
                        logger.info(f"Mapped loaded 'lstm_units' to 'gru_units'.")
                    elif key == "num_features" and "num_features_base" not in config_loaded:
                         default_config["num_features_base"] = value
                         logger.info(f"Mapped loaded 'num_features' to 'num_features_base'.")
                    # ### MODIFICATION START V3.1 ###
                    # Don't load num_features_time or num_features_total from file
                    elif key in ["num_features_time", "num_features_total"]:
                         logger.warning(f"Ignoring '{key}' from {config_file}. It will be calculated dynamically.")
                    # ### MODIFICATION END V3.1 ###
                    # Don't allow overriding calculated stat features count unless explicitly in file
                    elif key == "num_features_statistical" and key in config_loaded:
                         default_config[key] = value # Allow explicit override
                         logger.info(f"Using explicit 'num_features_statistical' from config file: {value}")
                    # Handle paths and other known keys
                    elif key in default_config:
                         # If it's a file path, make it relative to output_dir if not absolute
                         if key in ['export_file', 'cache_dir', 'tensorboard_log_dir']:
                             if not os.path.isabs(value):
                                 value = os.path.join(output_dir, value)
                         default_config[key] = value
                    # Handle hyperparameter search dict
                    elif key == "hyperparameter_search" and isinstance(value, dict):
                        default_config[key] = value # Load the whole dict
                    else:
                        logger.warning(f"Ignoring unknown or deprecated key '{key}' from {config_file}")

                # Recalculate stat features if windows changed AND stat count wasn't explicit in file
                if ("rolling_freq_windows" in config_loaded and
                    "num_features_statistical" not in config_loaded and # Only if not explicitly set
                    "num_features_base" in default_config): # Need base features count
                     num_stat_features = 1 + 1 + 1 + 4 + len(default_config["rolling_freq_windows"]) * default_config["num_features_base"]
                     if default_config["num_features_statistical"] != num_stat_features:
                         logger.info(f"Recalculating num_features_statistical to {num_stat_features} based on loaded rolling_freq_windows and num_features_base.")
                         default_config["num_features_statistical"] = num_stat_features

                logger.info("Configurações padrão mescladas com valores do arquivo.")
        else:
            logger.warning(f"Arquivo de configuração {config_file} não encontrado. Usando configurações padrão.")

    except json.JSONDecodeError as e:
        logger.error(f"Erro ao decodificar JSON de {config_file}: {e}")
    except Exception as e:
        logger.error(f"Erro ao carregar ou mesclar configurações: {e}")

    # Ensure dependent directories exist
    Path(default_config['tensorboard_log_dir']).mkdir(parents=True, exist_ok=True)
    Path(default_config['cache_dir']).mkdir(parents=True, exist_ok=True)
    if default_config.get('export_file'):
        Path(os.path.dirname(default_config['export_file'])).mkdir(parents=True, exist_ok=True)

    # num_features_total will be calculated later in main() after features are generated
    logger.info("Configuração carregada. 'num_features_time' e 'num_features_total' serão calculados dinamicamente.")

    return default_config

# --- Sistema de Cache ---
def get_cache_key(url):
    return hashlib.md5(url.encode()).hexdigest()

def is_cache_valid(cache_file, duration_hours):
    if not os.path.exists(cache_file): return False
    try:
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        return datetime.now() - cache_time < timedelta(hours=duration_hours)
    except Exception as e:
        logger.warning(f"Erro ao verificar tempo do cache {cache_file}: {e}")
        return False

def save_to_cache(data, cache_file):
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f: # Added encoding
            json.dump(data, f, ensure_ascii=False) # ensure_ascii=False for potentially non-ASCII chars
        logger.info(f"Dados salvos no cache: {cache_file}")
    except Exception as e: logger.error(f"Erro ao salvar cache: {e}")

def load_from_cache(cache_file):
    try:
        with open(cache_file, 'r', encoding='utf-8') as f: # Added encoding
            return json.load(f)
    except FileNotFoundError:
        logger.info(f"Arquivo de cache não encontrado: {cache_file}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Erro ao decodificar JSON do cache {cache_file}: {e}")
        return None
    except Exception as e:
        logger.error(f"Erro ao carregar cache {cache_file}: {e}")
        return None

# --- Funções de Dados ---

def download_and_prepare_data(url=None, file_path=None, cache_dir=None, cache_duration_hours=24):
    """Downloads/loads data, ensures 'BolaX' columns are present/numeric."""
    logger.info("Iniciando carregamento e preparação de dados...")
    df = None
    data = None # Initialize data

    # --- Cache and Download Logic ---
    if url and cache_dir:
        cache_key = get_cache_key(url)
        # Ensure cache_file path is constructed correctly even if cache_dir is relative
        safe_cache_dir = os.path.abspath(cache_dir)
        Path(safe_cache_dir).mkdir(parents=True, exist_ok=True) # Ensure it exists
        cache_file = os.path.join(safe_cache_dir, f"{cache_key}.json")

        if is_cache_valid(cache_file, cache_duration_hours):
            logger.info(f"Tentando carregar dados do cache: {cache_file}...")
            data = load_from_cache(cache_file)
            if data:
                logger.info("Dados carregados com sucesso do cache.")
            else:
                logger.warning("Cache inválido ou corrompido. Tentando baixar dados novamente.")
                data = None # Ensure data is None if cache load failed

        if data is None:
             logger.info("Cache expirado, não encontrado ou inválido. Baixando dados da URL...")
             try:
                 # Added headers to mimic browser request slightly
                 headers = {'User-Agent': 'Mozilla/5.0'}
                 # Using verify=False is generally discouraged, consider alternatives if possible
                 response = requests.get(url, headers=headers, verify=False, timeout=60)
                 response.raise_for_status() # Check for HTTP errors (4xx, 5xx)
                 # Try decoding explicitly as UTF-8 first
                 try:
                    data = response.json()
                 except UnicodeDecodeError:
                    logger.warning("Falha na decodificação UTF-8 inicial, tentando com a codificação detectada.")
                    response.encoding = response.apparent_encoding # Use detected encoding
                    data = response.json() # Retry json decoding

                 save_to_cache(data, cache_file)
                 logger.info("Dados baixados e salvos no cache com sucesso.")
             except requests.exceptions.Timeout:
                 logger.error(f"Timeout ao baixar dados de {url}")
                 data = None # Ensure data is None on error
             except requests.exceptions.SSLError as ssl_err:
                 logger.error(f"Erro SSL ao conectar a {url}. Verifique a URL ou certificados. Detalhes: {ssl_err}")
                 data = None
             except requests.exceptions.HTTPError as http_err:
                 logger.error(f"Erro HTTP {http_err.response.status_code} ao baixar dados: {http_err}")
                 data = None
             except requests.exceptions.RequestException as e:
                 logger.error(f"Erro de rede/conexão ao baixar dados: {e}")
                 data = None
             except json.JSONDecodeError as json_err:
                 logger.error(f"Erro ao decodificar JSON da resposta: {json_err}")
                 try: logger.error(f"Resposta recebida (início): {response.text[:500]}...")
                 except Exception: pass
                 data = None

    # --- Fallback to local file if URL/Cache failed or wasn't primary ---
    if data is None and file_path and os.path.exists(file_path):
        logger.info(f"Tentando carregar do arquivo local: {file_path} como fonte primária ou fallback...")
        # (Robust CSV reading logic from V2 remains good)
        try:
            df_loaded = None
            common_seps = [';', ',', '\t', '|']
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

            for enc in encodings_to_try:
                logger.debug(f"Tentando ler CSV com encoding '{enc}'...")
                try:
                    # Try common separators first
                    for sep in common_seps:
                         try:
                             df_try = pd.read_csv(file_path, sep=sep, encoding=enc)
                             if df_try.shape[1] >= 6:
                                 logger.info(f"CSV lido com sucesso (sep='{sep}', encoding='{enc}').")
                                 df_loaded = df_try
                                 break # Break inner loop (separator)
                         except Exception as e_sep:
                             logger.debug(f"Falha ao ler CSV com sep='{sep}', enc='{enc}'. Erro: {e_sep}")
                             continue # Try next separator
                    if df_loaded is not None: break # Break outer loop (encoding)

                    # If common separators failed, try auto-detect
                    if df_loaded is None:
                         logger.debug(f"Tentando detecção automática de separador (encoding='{enc}')...")
                         try:
                             df_auto = pd.read_csv(file_path, sep=None, engine='python', encoding=enc)
                             if df_auto.shape[1] >= 6:
                                 logger.info(f"Detecção automática de separador funcionou (encoding='{enc}').")
                                 df_loaded = df_auto
                                 break # Break outer loop (encoding)
                             else:
                                 logger.debug(f"Detecção automática resultou em < 6 colunas ({df_auto.shape[1]}) para enc='{enc}'.")
                         except Exception as e_auto:
                             logger.debug(f"Falha na detecção automática (encoding='{enc}'). Erro: {e_auto}")

                except UnicodeDecodeError:
                     logger.debug(f"Encoding '{enc}' falhou para o arquivo.")
                     continue # Try next encoding
                except Exception as e_enc:
                     logger.warning(f"Erro inesperado ao tentar ler com encoding '{enc}': {e_enc}")
                     continue # Try next encoding

            if df_loaded is not None:
                df = df_loaded
                logger.info(f"Dados carregados com sucesso de {file_path}")
            else:
                logger.error(f"Não foi possível ler o arquivo local {file_path} com separadores/encodings comuns.")
                return None # Critical failure if file was the only option or fallback failed
        except Exception as e_file:
            logger.error(f"Erro crítico ao tentar carregar arquivo local {file_path}: {e_file}")
            return None

    # --- Process data if loaded from URL/Cache (needs conversion to DataFrame) ---
    elif data is not None:
        logger.info("Processando dados carregados da API/Cache para DataFrame...")
        if isinstance(data, list) and data:
            results, concursos, datas = [], [], []
            required_keys = {'dezenas', 'concurso', 'data'} # Adjust if API format changes

            for i, sorteio in enumerate(data):
                if not isinstance(sorteio, dict):
                    logger.warning(f"Item {i} nos dados não é um dicionário, pulando: {sorteio}")
                    continue
                if not required_keys.issubset(sorteio.keys()):
                     missing = required_keys - sorteio.keys()
                     logger.warning(f"Sorteio {sorteio.get('concurso', f'index {i}')} com chaves ausentes ({missing}), pulando.")
                     continue
                try:
                    # Ensure 'dezenas' is a list of strings before converting
                    dezenas_raw = sorteio.get('dezenas', [])
                    if not isinstance(dezenas_raw, list):
                        logger.warning(f"Dezenas no sorteio {sorteio.get('concurso')} não é uma lista ({type(dezenas_raw)}), pulando.")
                        continue

                    # Convert to int, handling potential errors within the list
                    dezenas_int = []
                    valid_dezenas = True
                    for d_str in dezenas_raw:
                        try:
                            dezenas_int.append(int(d_str))
                        except (ValueError, TypeError):
                            logger.warning(f"Valor não numérico encontrado nas dezenas do sorteio {sorteio.get('concurso')}: '{d_str}', pulando sorteio.")
                            valid_dezenas = False
                            break
                    if not valid_dezenas: continue

                    dezenas = sorted(dezenas_int)

                    # Validate content
                    if len(dezenas) == 6 and all(1 <= d <= 60 for d in dezenas):
                        results.append(dezenas)
                        concursos.append(sorteio.get('concurso')) # Allow potential None, handle later
                        datas.append(sorteio.get('data')) # Allow potential None, handle later
                    else:
                        logger.warning(f"Sorteio {sorteio.get('concurso')} inválido (número/valor de dezenas): {dezenas}, pulando.")
                except Exception as e_proc: # Catch broader errors during processing
                    logger.warning(f"Erro inesperado ao processar sorteio {sorteio.get('concurso', f'index {i}')}: {e_proc} - Sorteio: {sorteio}")
                    continue # Skip this draw

            if not results:
                logger.error("Nenhum sorteio válido encontrado após processar dados da API/Cache.")
                df = None # Ensure df is None
            else:
                # Create DataFrame from valid results
                df = pd.DataFrame(results, columns=[f'Bola{i+1}' for i in range(6)])
                # Add Concurso and Data if available
                if concursos and len(concursos) == len(df):
                    # Attempt conversion to numeric, coerce errors to NaN for checking
                    df['Concurso'] = pd.to_numeric(concursos, errors='coerce')
                    if df['Concurso'].isnull().any():
                        logger.warning(f"Coluna 'Concurso' contém valores não numéricos ({df['Concurso'].isnull().sum()} ocorrências).")
                if datas and len(datas) == len(df):
                    # Attempt date conversion, coerce errors, check format consistency
                    df['Data'] = pd.to_datetime(datas, format='%d/%m/%Y', errors='coerce')
                    if df['Data'].isnull().any():
                         logger.warning(f"Coluna 'Data' contém valores não convertidos ({df['Data'].isnull().sum()} ocorrências). Verifique o formato ('dd/mm/yyyy').")

                logger.info(f"Dados processados com sucesso da API/Cache ({len(df)} sorteios válidos).")
        elif data is not None: # Handle cases where data was loaded but wasn't a list or was empty
            logger.error("Formato de dados da API/Cache não reconhecido ou vazio (esperado: lista de dicionários).")
            df = None

    # --- Final DataFrame Check and Column Processing (Common path for both sources) ---
    if df is None:
        logger.critical("Nenhuma fonte de dados (URL, Cache ou Arquivo Local) resultou em um DataFrame válido.")
        return None

    # --- Column Identification and Renaming (Improved Robustness) ---
    bola_cols_found = []
    potential_patterns = [
        [f'Bola{i+1}' for i in range(6)], [f'bola{i+1}' for i in range(6)],
        [f'Dezena{i+1}' for i in range(6)], [f'dezena{i+1}' for i in range(6)],
        [f'N{i+1}' for i in range(6)], [f'n{i+1}' for i in range(6)]
    ]
    df_cols_lower = {c.lower().strip(): c for c in df.columns} # Handle leading/trailing spaces and case

    for pattern_list in potential_patterns:
        pattern_lower = [p.lower() for p in pattern_list]
        # Check if all lowercase patterns exist in the lowercased column names
        if all(pat_low in df_cols_lower for pat_low in pattern_lower):
             # Retrieve the original column names using the mapping
             bola_cols_found = [df_cols_lower[pat_low] for pat_low in pattern_lower]
             logger.info(f"Colunas das bolas encontradas usando padrão (case-insensitive/strip): {bola_cols_found}")
             break # Found a pattern

    if not bola_cols_found:
        logger.warning("Nenhum padrão de coluna conhecido encontrado. Tentando heurística de colunas numéricas...")
        numeric_cols = df.select_dtypes(include=np.number).columns
        potential_bola_cols = []
        for c in numeric_cols:
             try:
                 # Coerce to numeric, check range 1-60, ensure they are integers (no decimals)
                 numeric_col = pd.to_numeric(df[c], errors='coerce')
                 # Check if all non-NaN values are between 1 and 60 (inclusive)
                 # and if all non-NaN values are whole numbers
                 is_likely_bola = (numeric_col.dropna().between(1, 60, inclusive='both').all() and
                                   numeric_col.dropna().apply(lambda x: x == int(x)).all())
                 if is_likely_bola:
                     potential_bola_cols.append(c)
             except Exception as e_heur:
                 logger.warning(f"Erro avaliando coluna '{c}' para heurística das bolas: {e_heur}")

        if len(potential_bola_cols) >= 6:
            # Try to be smarter: prefer columns named like 'bola', 'dezena', 'n' if available
            preferred_cols = [c for c in potential_bola_cols if any(kw in c.lower() for kw in ['bola', 'dezena', 'n'])]
            if len(preferred_cols) >= 6:
                 bola_cols_found = preferred_cols[:6]
                 logger.warning(f"Colunas identificadas heuristicamente (com preferência): {bola_cols_found}. VERIFIQUE!")
            else:
                 # Fallback to just the first 6 numeric likely columns
                 bola_cols_found = potential_bola_cols[:6]
                 logger.warning(f"Colunas identificadas heuristicamente (primeiras 6): {bola_cols_found}. VERIFIQUE!")
        else:
            logger.error(f"Erro: Não foi possível identificar 6 colunas numéricas válidas (1-60, inteiros). Colunas encontradas: {list(df.columns)}")
            return None

    # Rename found columns to standard 'Bola1', 'Bola2', ...
    rename_map = {found_col: f'Bola{i+1}' for i, found_col in enumerate(bola_cols_found)}
    try:
        df.rename(columns=rename_map, inplace=True)
        bola_cols = [f'Bola{i+1}' for i in range(6)] # Standard names from now on
        logger.info(f"Colunas renomeadas para: {bola_cols}")
    except Exception as e_rename:
        logger.error(f"Erro ao renomear colunas: {e_rename}")
        return None

    # Convert ball columns to numeric, coerce errors, drop invalid rows
    try:
        initial_rows = len(df)
        for col in bola_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Check for NaNs introduced by coercion or already present
            if df[col].isnull().any():
                nan_count = df[col].isnull().sum()
                logger.warning(f"Encontrados {nan_count} valores não numéricos/NaN em '{col}'. Removendo linhas correspondentes.")
                df.dropna(subset=[col], inplace=True)
        # Convert to integer type *after* handling NaNs
        for col in bola_cols:
             df[col] = df[col].astype(int)

        rows_after_cleaning = len(df)
        if rows_after_cleaning < initial_rows:
            logger.info(f"{initial_rows - rows_after_cleaning} linhas removidas devido a valores inválidos nas colunas das bolas.")
        if rows_after_cleaning == 0:
            logger.error("Nenhuma linha válida restante após limpeza das colunas das bolas.")
            return None
        logger.info("Colunas das bolas verificadas, convertidas para inteiro e linhas inválidas removidas.")

    except Exception as e_num:
        logger.error(f"Erro ao converter/limpar colunas de bolas: {e_num}", exc_info=True)
        return None

    # Select final columns and ensure chronological order
    cols_to_keep = bola_cols + [col for col in ['Concurso', 'Data'] if col in df.columns]
    final_df = df[cols_to_keep].copy()

    sort_col = None
    # Prioritize sorting by 'Concurso' if numeric, then by 'Data' if datetime
    if 'Concurso' in final_df.columns and pd.api.types.is_numeric_dtype(final_df['Concurso']) and not final_df['Concurso'].isnull().all():
        sort_col = 'Concurso'
    elif 'Data' in final_df.columns and pd.api.types.is_datetime64_any_dtype(final_df['Data']) and not final_df['Data'].isnull().all():
        sort_col = 'Data'

    if sort_col:
        try:
            final_df = final_df.sort_values(by=sort_col).reset_index(drop=True)
            logger.info(f"Dados finais ordenados por '{sort_col}'.")
        except Exception as e_sort:
            logger.error(f"Erro ao tentar ordenar por '{sort_col}': {e_sort}. Continuando sem ordenação garantida.")
    else:
        logger.warning("Não foi possível identificar uma coluna confiável ('Concurso' numérico ou 'Data' datetime) para garantir a ordem cronológica final.")

    logger.info(f"Processamento final: {len(final_df)} sorteios válidos e limpos carregados.")

    # Check for sufficient data *after* all cleaning and sorting
    min_data_needed = config.get('sequence_length', 15) * 3 # Use config value if available
    if len(final_df) < min_data_needed:
        logger.error(f"Dados insuficientes ({len(final_df)} sorteios) após limpeza para criar sequências/divisões. Mínimo recomendado ~{min_data_needed}")
        return None

    return final_df

# preprocess_data_labels remains the same as V2/V3
def preprocess_data_labels(df_balls_only, num_features_base):
    """Transforms winning numbers into MultiLabelBinarizer format (labels y)."""
    logger.info("Iniciando pré-processamento dos labels (MultiLabelBinarizer)...")
    try:
        if df_balls_only is None or df_balls_only.empty:
             logger.error("DataFrame vazio fornecido para pré-processar labels.")
             return None, None, None
        required_cols = [f'Bola{i+1}' for i in range(6)]
        if not all(col in df_balls_only.columns for col in required_cols):
            missing_cols = [c for c in required_cols if c not in df_balls_only.columns]
            logger.error(f"Colunas das bolas ausentes no DataFrame para labels: {missing_cols}")
            return None, None, None

        # Select only the ball columns for processing
        balls_df = df_balls_only[required_cols].copy()

        # Validate data types and ranges before creating list
        initial_rows = len(balls_df)
        rows_to_drop = []
        for index, row in balls_df.iterrows():
            valid_row = True
            for col in required_cols:
                val = row[col]
                # Check if value is integer and within the expected range
                if not (isinstance(val, (int, np.integer)) and 1 <= val <= num_features_base):
                    logger.debug(f"Valor inválido ({val}, tipo {type(val)}) na linha {index}, coluna {col}. Marcando para remoção.")
                    valid_row = False
                    break # No need to check other columns in this row
            if not valid_row:
                rows_to_drop.append(index)

        if rows_to_drop:
            original_indices = balls_df.index # Store original indices before dropping
            balls_df.drop(rows_to_drop, inplace=True)
            logger.warning(f"Removidas {len(rows_to_drop)} linhas com valores inválidos/fora do range [1, {num_features_base}] nas bolas durante pré-processamento de labels.")
            if balls_df.empty:
                logger.error("Nenhuma linha válida restante após validação dos valores das bolas.")
                return None, None, None
            # Get the original indices that were kept
            valid_original_indices = original_indices.difference(rows_to_drop)
        else:
            valid_original_indices = balls_df.index # All original indices are valid

        # Convert the validated DataFrame rows to a list of lists
        draws_list = balls_df.values.tolist()

        # Fit the MultiLabelBinarizer
        mlb = MultiLabelBinarizer(classes=list(range(1, num_features_base + 1)))
        encoded_data = mlb.fit_transform(draws_list)

        logger.info(f"Labels transformados com sucesso: {encoded_data.shape[0]} amostras, {encoded_data.shape[1]} features base (números 1-{num_features_base}).")
        # Return the original indices corresponding to the encoded_data
        return encoded_data, mlb, valid_original_indices

    except Exception as e:
        logger.error(f"Erro durante pré-processamento dos labels: {e}", exc_info=True)
        return None, None, None


# ### MODIFICATION START V3.1 - Complex Time Features ###
# Replaced the simple time features function with this complex version
def add_complex_time_features(df_balls_only, num_features_base):
    """
    Calculates complex time features for each number:
    1. Time since last seen (current interval).
    2. Mean of past intervals between sightings.
    3. Standard deviation of past intervals between sightings.
    Returns a NumPy array (n_draws, n_features_base * 3).
    """
    logger.info("Calculando features de tempo COMPLEXAS (intervalo atual, média intervalos, desv padrão intervalos)...")
    try:
        if df_balls_only is None or df_balls_only.empty:
             logger.error("DataFrame vazio fornecido para calcular features de tempo complexas.")
             return None

        bola_cols = [f'Bola{i+1}' for i in range(6)]
        # Ensure columns exist before accessing .values
        if not all(col in df_balls_only.columns for col in bola_cols):
            missing_cols = [c for c in bola_cols if c not in df_balls_only.columns]
            logger.error(f"Colunas das bolas ausentes no DataFrame para features de tempo: {missing_cols}")
            return None

        draws = df_balls_only[bola_cols].values
        num_draws = len(draws)
        num_time_feat_per_num = 3 # interval, mean, std
        total_time_features = num_features_base * num_time_feat_per_num

        # Output array initialized with zeros (float32 for TF compatibility)
        time_features_complex = np.zeros((num_draws, total_time_features), dtype=np.float32)

        # Store lists of draw indices where each number appeared
        # Key: number (1-60), Value: list of draw indices [0, 5, 12, ...]
        seen_history = {num: [] for num in range(1, num_features_base + 1)}

        for i in range(num_draws): # Iterate through each draw (index i)
            current_numbers_in_draw = set(draws[i])

            for num in range(1, num_features_base + 1): # Iterate through each number (1-60)
                history = seen_history[num]
                num_sightings = len(history)

                # --- Feature 1: Current Interval (Time since last seen) ---
                if num_sightings == 0:
                    # Never seen before this draw 'i'
                    current_interval = float(i + 1) # Interval is number of draws so far + 1
                else:
                    # Seen before, calculate interval from last sighting
                    last_sighting_index = history[-1]
                    current_interval = float(i - last_sighting_index)

                # --- Feature 2 & 3: Mean and Std Dev of *past* intervals ---
                mean_interval = 0.0
                std_dev_interval = 0.0 # Default to 0 if not enough data

                if num_sightings >= 2: # Need at least 2 sightings to have 1 past interval
                    # Calculate intervals *between* past sightings
                    # Example: history = [2, 8, 15] -> diff -> [6, 7]
                    past_intervals = np.diff(np.array(history))

                    if len(past_intervals) > 0: # Should always be true if num_sightings >= 2
                        mean_interval = np.mean(past_intervals)

                        # Need at least 2 *intervals* (i.e., 3 sightings) for a meaningful standard deviation
                        if len(past_intervals) >= 2:
                            std_dev_interval = np.std(past_intervals)
                        # If only 1 interval (2 sightings), std dev is technically undefined or 0.
                        # We keep it as 0.0 here for simplicity in the feature vector.

                # --- Store the results in the output array ---
                # Index mapping:
                # Number 1 -> cols 0 (interval), 1 (mean), 2 (std)
                # Number 2 -> cols 3 (interval), 4 (mean), 5 (std)
                # ...
                # Number k -> cols 3*(k-1), 3*(k-1)+1, 3*(k-1)+2
                base_col_index = (num - 1) * num_time_feat_per_num
                time_features_complex[i, base_col_index]     = current_interval
                time_features_complex[i, base_col_index + 1] = mean_interval
                time_features_complex[i, base_col_index + 2] = std_dev_interval

            # --- Update history for numbers present in the *current* draw ---
            # This happens *after* calculating features for draw 'i', so history reflects state *before* this draw
            for drawn_num in current_numbers_in_draw:
                if 1 <= drawn_num <= num_features_base:
                    seen_history[drawn_num].append(i) # Append the index of the current draw

        logger.info(f"Features de tempo complexas calculadas. Shape: {time_features_complex.shape}")
        # Add validation for NaNs or Infs if necessary, though np.zeros and calculations should avoid them
        if np.isnan(time_features_complex).any() or np.isinf(time_features_complex).any():
            logger.error("Features de tempo complexas contêm NaN ou Inf!")
            # Handle or return None, e.g., replace with 0 or investigate source
            # time_features_complex = np.nan_to_num(time_features_complex, nan=0.0, posinf=0.0, neginf=0.0)
            return None # Indicate failure

        return time_features_complex

    except Exception as e:
        logger.error(f"Erro ao calcular features de tempo complexas: {e}", exc_info=True)
        return None
# ### MODIFICATION END V3.1 - Complex Time Features ###


# add_statistical_features remains the same as V3
def add_statistical_features(df_balls_only, num_features_base, rolling_windows):
    """
    Calculates statistical features for each draw:
    - Parity (Odd Count)
    - Sum
    - Range (Max - Min)
    - Zone Counts (4 zones: 1-15, 16-30, 31-45, 46-60)
    - Rolling Frequencies (for specified windows)
    Returns a NumPy array (n_draws, n_stat_features).
    """
    logger.info(f"Calculando features estatísticas (Par/Ímpar, Soma, Range, Zonas, Freq. Janelas: {rolling_windows})...")
    try:
        if df_balls_only is None or df_balls_only.empty:
             logger.error("DataFrame vazio fornecido para calcular features estatísticas.")
             return None

        bola_cols = [f'Bola{i+1}' for i in range(6)]
        if not all(col in df_balls_only.columns for col in bola_cols):
            missing_cols = [c for c in bola_cols if c not in df_balls_only.columns]
            logger.error(f"Colunas das bolas ausentes no DataFrame para features estatísticas: {missing_cols}")
            return None

        draws = df_balls_only[bola_cols].values
        num_draws = len(draws)

        # --- Calculate Rolling Frequencies First ---
        num_freq_features = len(rolling_windows) * num_features_base
        rolling_freq_features = np.zeros((num_draws, num_freq_features), dtype=np.float32)

        # Need one-hot encoded draws for efficient rolling sum/frequency count
        try:
            mlb_freq = MultiLabelBinarizer(classes=list(range(1, num_features_base + 1)))
            # Ensure draws list passed to fit_transform is valid
            draws_list_for_mlb = [[int(n) for n in row] for row in draws] # Explicit int conversion might be redundant but safe
            encoded_draws_freq = mlb_freq.fit_transform(draws_list_for_mlb) # Shape (num_draws, num_features_base)
            encoded_draws_df = pd.DataFrame(encoded_draws_freq, columns=mlb_freq.classes_, index=df_balls_only.index) # Keep original index if needed
        except Exception as e_mlb:
            logger.error(f"Erro ao criar representação MultiLabelBinarizer para cálculo de frequência: {e_mlb}", exc_info=True)
            return None

        logger.info("Calculando frequências rolantes...")
        freq_col_offset = 0
        for window in rolling_windows:
            if window <= 0:
                logger.warning(f"Janela rolante inválida ({window}) encontrada, pulando.")
                continue
            logger.debug(f"  Calculando frequência para janela: {window}")
            try:
                # Rolling sum on the one-hot encoded data gives frequency count in the window
                # min_periods=1 ensures output starts from the first draw, window size grows until full
                rolling_sum = encoded_draws_df.rolling(window=window, min_periods=1).sum()

                # Shift by 1 so the frequency for draw 'i' reflects draws *before* 'i' (up to i-1)
                # fillna(0) handles the first row which has no preceding data after shifting.
                rolling_sum_shifted = rolling_sum.shift(1).fillna(0)

                # Store in the correct slice of the main frequency array
                start_idx = freq_col_offset
                end_idx = freq_col_offset + num_features_base
                rolling_freq_features[:, start_idx : end_idx] = rolling_sum_shifted.values
                freq_col_offset += num_features_base
            except Exception as e_roll:
                 logger.error(f"Erro ao calcular frequência rolante para janela {window}: {e_roll}", exc_info=True)
                 # Decide if you want to continue or abort. Aborting might be safer.
                 return None
        logger.info("Frequências rolantes calculadas.")

        # --- Calculate Draw-Specific Stats (Parity, Sum, Range, Zones) ---
        logger.info("Calculando estatísticas por sorteio (Par/Ímpar, Soma, Range, Zonas)...")
        odd_counts = []
        sums = []
        ranges = []
        # Define zones clearly
        zone_defs = [(1, 15), (16, 30), (31, 45), (46, 60)]
        num_zones = len(zone_defs)
        zone_counts_arr = np.zeros((num_draws, num_zones), dtype=np.int32) # Pre-allocate zone counts

        try:
            for i in range(num_draws):
                current_numbers = draws[i] # This is already validated as int[6] from earlier steps

                # Parity (Count of Odd Numbers)
                odd_counts.append(np.sum(current_numbers % 2 != 0))
                # Sum of Numbers
                sums.append(np.sum(current_numbers))
                # Range (Max - Min)
                ranges.append(np.max(current_numbers) - np.min(current_numbers))
                # Zone Counts
                for zone_idx, (z_min, z_max) in enumerate(zone_defs):
                    zone_counts_arr[i, zone_idx] = np.sum((current_numbers >= z_min) & (current_numbers <= z_max))

        except Exception as e_stat:
             logger.error(f"Erro ao calcular estatísticas básicas no sorteio índice {i}: {e_stat}", exc_info=True)
             return None

        # Combine all statistical features into a single array
        # Reshape 1D lists to 2D arrays (n_draws, 1) for concatenation
        odd_counts_arr = np.array(odd_counts).reshape(-1, 1)
        sums_arr = np.array(sums).reshape(-1, 1)
        ranges_arr = np.array(ranges).reshape(-1, 1)
        # zone_counts_arr is already (n_draws, num_zones)

        # Concatenate along axis=1 (columns)
        statistical_features_raw = np.concatenate([
            odd_counts_arr,
            sums_arr,
            ranges_arr,
            zone_counts_arr, # Already (n_draws, 4)
            rolling_freq_features # Already (n_draws, num_freq_features)
        ], axis=1).astype(np.float32) # Ensure float32 for TF

        # --- Final Validation ---
        expected_cols = 1 + 1 + 1 + num_zones + num_freq_features
        actual_cols = statistical_features_raw.shape[1]
        logger.info(f"Features estatísticas combinadas. Shape: {statistical_features_raw.shape}")

        if actual_cols != expected_cols:
             logger.error(f"Erro de shape final nas features estatísticas! Esperado {expected_cols} colunas, obtido {actual_cols}")
             # Log details of components for debugging
             logger.error(f"  Component shapes: Odd({odd_counts_arr.shape}), Sum({sums_arr.shape}), Range({ranges_arr.shape}), Zones({zone_counts_arr.shape}), Freq({rolling_freq_features.shape})")
             return None
        if np.isnan(statistical_features_raw).any() or np.isinf(statistical_features_raw).any():
            logger.error("Features estatísticas contêm NaN ou Inf!")
            return None

        return statistical_features_raw

    except Exception as e:
        logger.error(f"Erro geral ao calcular features estatísticas: {e}", exc_info=True)
        return None


# split_data remains the same as V3 (handles separate time/stat features and scaling)
def split_data(encoded_labels, time_features_raw, statistical_features_raw,
               test_size_ratio, validation_split_ratio, sequence_length):
    """
    Divides labels and features (time, statistical) into train/val/test.
    Scales time and statistical features separately using StandardScaler fit on train data.
    Creates sequences for each set.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test, time_scaler, stat_scaler
    """
    logger.info("Dividindo dados, escalando features (Tempo e Estatísticas separadamente) e criando sequências...")
    try:
        # --- Input Validation ---
        if not isinstance(encoded_labels, np.ndarray) or encoded_labels.ndim != 2:
            logger.error(f"encoded_labels inválido (tipo {type(encoded_labels)}, ndim {encoded_labels.ndim}). Esperado ndarray 2D.")
            return [None] * 8 # Return 8 Nones (X_tr, X_v, X_te, y_tr, y_v, y_te, time_scaler, stat_scaler)
        if not isinstance(time_features_raw, np.ndarray) or time_features_raw.ndim != 2:
            logger.error(f"time_features_raw inválido (tipo {type(time_features_raw)}, ndim {time_features_raw.ndim}). Esperado ndarray 2D.")
            return [None] * 8
        if not isinstance(statistical_features_raw, np.ndarray) or statistical_features_raw.ndim != 2:
            logger.error(f"statistical_features_raw inválido (tipo {type(statistical_features_raw)}, ndim {statistical_features_raw.ndim}). Esperado ndarray 2D.")
            return [None] * 8

        n_samples = len(encoded_labels)
        if not (n_samples == len(time_features_raw) == len(statistical_features_raw)):
             logger.error(f"Disparidade no número de amostras entre labels ({n_samples}), "
                          f"time features ({len(time_features_raw)}), e "
                          f"statistical features ({len(statistical_features_raw)}) antes da divisão.")
             return [None] * 8

        if n_samples <= sequence_length:
             logger.error(f"Número total de amostras ({n_samples}) é menor ou igual ao tamanho da sequência ({sequence_length}). Não é possível criar sequências.")
             return [None] * 8

        # --- 1. Split Indices Chronologically ---
        if not (0 < test_size_ratio < 1):
            logger.error(f"test_size_ratio ({test_size_ratio}) inválido. Deve estar entre 0 e 1.")
            return [None] * 8
        if not (0 <= validation_split_ratio < 1):
            logger.error(f"validation_split_ratio ({validation_split_ratio}) inválido. Deve estar entre 0 (inclusive) e 1.")
            return [None] * 8
        if test_size_ratio + validation_split_ratio >= 1.0:
             logger.error(f"Soma de test_size_ratio ({test_size_ratio}) e validation_split_ratio ({validation_split_ratio}) deve ser menor que 1.")
             return [None] * 8

        # Calculate split points based on indices
        test_split_index = int(n_samples * (1 - test_size_ratio))
        # Validation split is relative to the remaining non-test data
        val_split_index = int(test_split_index * (1 - validation_split_ratio))

        # Generate index arrays
        train_indices = np.arange(val_split_index)
        val_indices = np.arange(val_split_index, test_split_index)
        test_indices = np.arange(test_split_index, n_samples)

        logger.info(f"Índices Brutos - Treino: {len(train_indices)} (0 a {val_split_index-1}), "
                    f"Val: {len(val_indices)} ({val_split_index} a {test_split_index-1}), "
                    f"Teste: {len(test_indices)} ({test_split_index} a {n_samples-1})")

        # --- Check for Sufficient Data in Each Split for Sequence Creation ---
        min_len_for_seq = sequence_length + 1 # Need sequence_length samples for input + 1 for target

        # Validate Train set
        if len(train_indices) < min_len_for_seq:
             logger.error(f"Conjunto de treino resultante ({len(train_indices)} amostras) é muito pequeno para criar sequências de tamanho {sequence_length}. Mínimo necessário: {min_len_for_seq}.")
             return [None] * 8

        # Validate Validation set (allow it to be empty if validation_split_ratio is 0 or too small)
        if validation_split_ratio > 0 and len(val_indices) < min_len_for_seq:
             logger.warning(f"Conjunto de validação ({len(val_indices)} amostras) pequeno demais para criar sequências de tamanho {sequence_length} (mínimo {min_len_for_seq}). Validação será efetivamente vazia.")
             # Keep indices as calculated, create_sequences will handle empty result later
        elif validation_split_ratio == 0:
             logger.info("Conjunto de validação terá 0 amostras pois validation_split_ratio=0.")

        # Validate Test set
        if len(test_indices) < min_len_for_seq:
             logger.warning(f"Conjunto de teste ({len(test_indices)} amostras) pequeno demais para criar sequências de tamanho {sequence_length} (mínimo {min_len_for_seq}). Teste será efetivamente vazio.")
             # Keep indices, create_sequences handles empty result


        # --- 2. Slice Data Based on Indices ---
        # Slice labels
        train_labels = encoded_labels[train_indices]
        val_labels = encoded_labels[val_indices] if len(val_indices) > 0 else np.array([])
        test_labels = encoded_labels[test_indices] if len(test_indices) > 0 else np.array([])

        # Slice raw time features
        train_time_raw = time_features_raw[train_indices]
        val_time_raw = time_features_raw[val_indices] if len(val_indices) > 0 else np.array([])
        test_time_raw = time_features_raw[test_indices] if len(test_indices) > 0 else np.array([])

        # Slice raw statistical features
        train_stat_raw = statistical_features_raw[train_indices]
        val_stat_raw = statistical_features_raw[val_indices] if len(val_indices) > 0 else np.array([])
        test_stat_raw = statistical_features_raw[test_indices] if len(test_indices) > 0 else np.array([])

        logger.info(f"Dados brutos divididos: Treino({train_labels.shape[0]}), Val({val_labels.shape[0]}), Teste({test_labels.shape[0]})")

        # --- 3. Scale Features (Fit ONLY on Training Data) ---
        logger.info("Ajustando Scalers (StandardScaler) SEPARADAMENTE nas features de TEMPO e ESTATÍSTICAS do treino...")
        time_scaler = StandardScaler()
        stat_scaler = StandardScaler()

        # Fit and transform training data
        # Handle case where train data might be empty (though checked earlier)
        if train_time_raw.size > 0:
            train_time_scaled = time_scaler.fit_transform(train_time_raw)
        else:
            train_time_scaled = np.array([]) # Should not happen based on earlier checks
            logger.error("train_time_raw está vazio antes do scaling. Abortando.")
            return [None] * 8

        if train_stat_raw.size > 0:
            train_stat_scaled = stat_scaler.fit_transform(train_stat_raw)
        else:
            train_stat_scaled = np.array([]) # Should not happen
            logger.error("train_stat_raw está vazio antes do scaling. Abortando.")
            return [None] * 8

        logger.info("Scalers ajustados. Escalando features de validação e teste...")
        # Transform validation and test sets using the *fitted* scalers
        val_time_scaled = time_scaler.transform(val_time_raw) if val_time_raw.size > 0 else np.array([])
        test_time_scaled = time_scaler.transform(test_time_raw) if test_time_raw.size > 0 else np.array([])

        val_stat_scaled = stat_scaler.transform(val_stat_raw) if val_stat_raw.size > 0 else np.array([])
        test_stat_scaled = stat_scaler.transform(test_stat_raw) if test_stat_raw.size > 0 else np.array([])

        # Optional: Save scalers (uncomment if needed)
        # try:
        #     scaler_dir = os.path.join(output_dir, "scalers")
        #     os.makedirs(scaler_dir, exist_ok=True)
        #     joblib.dump(time_scaler, os.path.join(scaler_dir,'time_scaler.gz'))
        #     joblib.dump(stat_scaler, os.path.join(scaler_dir,'stat_scaler.gz'))
        #     logger.info(f"Scalers salvos em {scaler_dir}")
        # except Exception as e_save_scaler:
        #      logger.error(f"Erro ao salvar scalers: {e_save_scaler}")

        # --- 4. Create Sequences for each set ---
        # The create_sequences function handles potentially empty inputs based on the sliced data
        logger.info("Criando sequências para treino...")
        X_train, y_train = create_sequences(train_labels, train_time_scaled, train_stat_scaled, sequence_length)

        logger.info("Criando sequências para validação...")
        X_val, y_val = create_sequences(val_labels, val_time_scaled, val_stat_scaled, sequence_length)

        logger.info("Criando sequências para teste...")
        X_test, y_test = create_sequences(test_labels, test_time_scaled, test_stat_scaled, sequence_length)

        # Log final shapes after sequence creation
        logger.info(f"Tamanho final dos conjuntos de sequências:")
        logger.info(f"- Treino:    X={X_train.shape if X_train.size>0 else 'Vazio'}, y={y_train.shape if y_train.size>0 else 'Vazio'}")
        logger.info(f"- Validação: X={X_val.shape if X_val.size>0 else 'Vazio'}, y={y_val.shape if y_val.size>0 else 'Vazio'}")
        logger.info(f"- Teste:     X={X_test.shape if X_test.size>0 else 'Vazio'}, y={y_test.shape if y_test.size>0 else 'Vazio'}")

        # Final check: ensure training set is not empty after sequencing
        if X_train.size == 0 or y_train.size == 0:
            logger.error("Conjunto de Treino resultou VAZIO após criação das sequências. Verifique sequence_length e tamanho dos dados.")
            return [None] * 8

        # Return all sets and the fitted scalers
        return X_train, X_val, X_test, y_train, y_val, y_test, time_scaler, stat_scaler

    except Exception as e:
        logger.error(f"Erro inesperado ao dividir/escalar dados ou criar sequências: {e}", exc_info=True)
        return [None] * 8 # Return 8 Nones


# create_sequences remains the same as V3 (handles combined features)
def create_sequences(encoded_labels, time_features_scaled, statistical_features_scaled, sequence_length):
    """
    Creates sequences by combining labels, scaled time features, and scaled statistical features.
    X: Combined sequence data (num_sequences, sequence_length, num_total_features).
    y: Next draw's labels (num_sequences, num_features_base).
    Handles potentially empty inputs gracefully (e.g., for val/test sets).
    """
    # --- Input Validation ---
    # Check if inputs are ndarrays and have the expected dimensions
    # Allow empty arrays for val/test, but log error if labels are present but features aren't (or vice versa)
    if ((encoded_labels is not None and encoded_labels.size > 0) and
        (time_features_scaled is None or time_features_scaled.size == 0 or
         statistical_features_scaled is None or statistical_features_scaled.size == 0)):
        logger.error("Inconsistência: Labels presentes, mas features de tempo/estatísticas ausentes/vazias em create_sequences.")
        return np.array([]), np.array([]) # Return empty

    # If any critical input is None or empty, return empty arrays immediately
    if (encoded_labels is None or encoded_labels.size == 0 or
        time_features_scaled is None or time_features_scaled.size == 0 or
        statistical_features_scaled is None or statistical_features_scaled.size == 0):
        logger.debug("Input vazio ou ausente para create_sequences (normal para val/test vazios), retornando vazio.")
        return np.array([]), np.array([])

    # Check for consistent number of samples (rows) across inputs
    n_samples_total = len(encoded_labels)
    if not (n_samples_total == len(time_features_scaled) == len(statistical_features_scaled)):
        logger.error(f"Inconsistência no número de amostras em create_sequences: "
                     f"Labels({n_samples_total}), Time({len(time_features_scaled)}), Stat({len(statistical_features_scaled)})")
        return np.array([]), np.array([])

    # Check if there are enough samples to create at least one sequence
    if n_samples_total <= sequence_length:
        logger.debug(f"Dados insuficientes ({n_samples_total} amostras) para criar sequências de tamanho {sequence_length} em create_sequences.")
        return np.array([]), np.array([])

    # --- Proceed with Sequence Creation ---
    logger.debug(f"Criando sequências de tamanho {sequence_length} a partir de {n_samples_total} amostras...")
    try:
        # Determine dimensions
        num_sequences = n_samples_total - sequence_length
        num_features_base = encoded_labels.shape[1]
        num_features_time = time_features_scaled.shape[1]
        num_features_stat = statistical_features_scaled.shape[1]
        num_features_total = num_features_base + num_features_time + num_features_stat

        # Pre-allocate NumPy arrays for efficiency
        # Use float32 for X (model input), keep y's dtype (likely int or bool)
        X = np.zeros((num_sequences, sequence_length, num_features_total), dtype=np.float32)
        y = np.zeros((num_sequences, num_features_base), dtype=encoded_labels.dtype)

        # Loop through the data to create sequences
        for i in range(num_sequences):
            # Extract sequences for each feature type
            # End index is i + sequence_length (exclusive)
            seq_labels = encoded_labels[i : i + sequence_length]
            seq_time = time_features_scaled[i : i + sequence_length]
            seq_stat = statistical_features_scaled[i : i + sequence_length]

            # Combine features along the last axis (feature axis)
            # Ensure concatenation happens correctly: (seq_len, f_base), (seq_len, f_time), (seq_len, f_stat) -> (seq_len, f_total)
            X[i] = np.concatenate((seq_labels, seq_time, seq_stat), axis=-1)

            # Target is the label of the draw immediately following the sequence
            y[i] = encoded_labels[i + sequence_length]

        logger.debug(f"{len(X)} sequências combinadas criadas. Shape X: {X.shape}, Shape y: {y.shape}")
        return X, y

    except Exception as e:
        logger.error(f"Erro inesperado ao criar sequências combinadas: {e}", exc_info=True)
        return np.array([]), np.array([])


# --- Modelo ---

# build_model remains the same as V3 (uses dynamically calculated total features)
def build_model(sequence_length, num_features_total, num_features_base, gru_units, dropout_rate, use_batch_norm):
    """ Constrói o modelo GRU com opção de Batch Normalization. """
    logger.info(f"Construindo modelo GRU: SeqLen={sequence_length}, TotalFeat={num_features_total}, BaseFeat={num_features_base}, "
                f"GRUUnits={gru_units}, Dropout={dropout_rate}, BatchNorm={use_batch_norm}")
    try:
        if not all(isinstance(arg, int) and arg > 0 for arg in [sequence_length, num_features_total, num_features_base, gru_units]):
            logger.error("Argumentos numéricos inválidos para build_model (devem ser inteiros > 0).")
            return None
        if not isinstance(dropout_rate, (float, int)) or not 0 <= dropout_rate < 1:
            logger.error(f"dropout_rate inválido: {dropout_rate}. Deve estar em [0, 1).")
            return None
        if not isinstance(use_batch_norm, bool):
            logger.error(f"use_batch_norm inválido: {use_batch_norm}. Deve ser True ou False.")
            return None

        model = Sequential(name="Modelo_GRU_MegaSena_V3")

        # Input Layer specifying the shape
        model.add(Input(shape=(sequence_length, num_features_total), name="Input_Layer"))

        # Optional Batch Norm after Input
        if use_batch_norm:
            model.add(BatchNormalization(name="BN_Input"))

        # First GRU Layer
        model.add(GRU(
            gru_units,
            return_sequences=True, # Return full sequence for the next GRU layer
            kernel_initializer='he_normal',
            recurrent_initializer='orthogonal',
            name="GRU_1"
        ))

        # Optional Batch Norm after GRU 1
        if use_batch_norm:
            model.add(BatchNormalization(name="BN_GRU_1"))

        # Dropout after GRU 1
        model.add(Dropout(dropout_rate, name="Dropout_GRU_1"))

        # Second GRU Layer (Optional - consider removing if model is too complex or slow)
        # Takes the sequence output from the first GRU
        model.add(GRU(
            gru_units // 2, # Typically reduce units in deeper layers
            return_sequences=False, # Only return the output of the last time step
            kernel_initializer='he_normal',
            recurrent_initializer='orthogonal',
            name="GRU_2"
        ))

        # Optional Batch Norm after GRU 2
        if use_batch_norm:
            model.add(BatchNormalization(name="BN_GRU_2"))

        # Dropout after GRU 2
        model.add(Dropout(dropout_rate, name="Dropout_GRU_2"))

        # Dense Layer (Optional - helps learn combinations of GRU output features)
        model.add(Dense(gru_units // 2, activation='relu', name="Dense_Hidden")) # ReLU activation

        # Optional Batch Norm before final Dense layer (often placed before activation, but can be after)
        if use_batch_norm:
             model.add(BatchNormalization(name="BN_Dense_Hidden"))

        # Dropout before Output Layer
        model.add(Dropout(dropout_rate, name="Dropout_Output"))

        # Output Layer: Predicts probability for each of the base features (numbers 1-60)
        # Sigmoid activation for multi-label binary classification (each number is independent)
        model.add(Dense(num_features_base, activation='sigmoid', name="Output_Layer"))

        # Optimizer (Adam is a good default)
        # Consider adjusting learning rate based on tuning or ReduceLROnPlateau results
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0008) # Starting LR, may be reduced by callback

        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy', # Suitable for multi-label binary classification
            metrics=[
                'binary_accuracy', # Accuracy for each label prediction
                tf.keras.metrics.AUC(name='auc') # Area Under Curve - good for ranking/imbalanced data
                ]
        )

        # Build the model with a sample batch shape to finalize weights shapes
        # Use None for batch size to allow variable batch sizes
        model.build((None, sequence_length, num_features_total))

        # Log the model summary using Keras' built-in method (more structured)
        logger.info("Resumo do Modelo (Keras):")
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        for line in model_summary.split('\n'):
            logger.info(line) # Log each line of the summary

        return model

    except Exception as e:
        logger.error(f"Erro ao construir o modelo GRU: {e}", exc_info=True)
        return None


# train_model remains the same as V3 (uses TensorBoard, custom logger)
def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, log_dir):
    """ Treina o modelo GRU com callbacks otimizados e TensorBoard. """
    logger.info("Iniciando o treinamento do modelo GRU...")
    try:
        if model is None:
            logger.error("Modelo inválido fornecido para treinamento.")
            return None
        if X_train is None or X_train.size == 0 or y_train is None or y_train.size == 0:
            logger.error("Dados de treinamento inválidos ou vazios.")
            return None

        # --- Callbacks ---
        # Early Stopping: Stop training if validation loss doesn't improve
        early_stopping = EarlyStopping(
            monitor='val_loss',     # Monitor validation loss
            patience=25,            # Number of epochs with no improvement to wait
            restore_best_weights=True, # Restore model weights from the epoch with the best val_loss
            verbose=1
        )
        # Reduce Learning Rate on Plateau: Decrease LR if val_loss stagnates
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.25,            # Factor by which LR is reduced (new_lr = lr * factor)
            patience=10,            # Number of epochs with no improvement to wait before reducing LR
            min_lr=0.00001,         # Lower bound for the learning rate
            verbose=1
        )
        # TensorBoard: Log metrics and graph for visualization
        tensorboard_callback = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,       # Log activation and weight histograms every epoch
            write_graph=True,       # Write model graph
            update_freq='epoch'     # Log metrics after each epoch
        )
        logger.info(f"Logs do TensorBoard serão salvos em: {log_dir}")

        # Custom Logger Callback (for console output per epoch)
        class TrainingLogger(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                log_str = f"Época {epoch + 1}/{self.params['epochs']}"
                # Training Metrics
                log_str += f" - Loss: {logs.get('loss', -1):.4f}"
                if 'binary_accuracy' in logs: log_str += f" - Acc: {logs.get('binary_accuracy', -1):.4f}"
                if 'auc' in logs: log_str += f" - AUC: {logs.get('auc', -1):.4f}"
                # Validation Metrics (if available)
                if 'val_loss' in logs: log_str += f" - Val Loss: {logs.get('val_loss', -1):.4f}"
                if 'val_binary_accuracy' in logs: log_str += f" - Val Acc: {logs.get('val_binary_accuracy', -1):.4f}"
                if 'val_auc' in logs: log_str += f" - Val AUC: {logs.get('val_auc', -1):.4f}"
                # Learning Rate
                if hasattr(self.model.optimizer, 'learning_rate'):
                     # Get current LR value safely
                     try:
                         lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
                         log_str += f" - LR: {lr:.6f}"
                     except Exception: # Handle cases where LR might not be directly accessible
                         log_str += " - LR: N/A"
                logger.info(log_str)

        # --- Determine Validation Data ---
        # Use validation set if provided and not empty
        validation_data = None
        if X_val is not None and X_val.size > 0 and y_val is not None and y_val.size > 0:
            # Ensure validation data shape matches expected input shape
            if X_val.shape[1:] == model.input_shape[1:] and y_val.shape[1:] == model.output_shape[1:]:
                 validation_data = (X_val, y_val)
                 logger.info(f"Usando conjunto de validação com {len(X_val)} amostras.")
            else:
                 logger.error(f"Shape do conjunto de validação X_val={X_val.shape} ou y_val={y_val.shape} "
                              f"incompatível com o modelo (Input: {model.input_shape}, Output: {model.output_shape}). "
                              "Validação será desativada.")
                 # Adjust callbacks if validation is unusable
                 early_stopping.monitor = 'loss'
                 reduce_lr.monitor = 'loss'
        else:
             logger.warning("Conjunto de validação vazio ou ausente. EarlyStopping/ReduceLR usarão 'loss' (treino) ao invés de 'val_loss'.")
             # Adjust callbacks to monitor training loss instead
             early_stopping.monitor = 'loss'
             reduce_lr.monitor = 'loss'

        # --- Train the Model ---
        logger.info(f"Iniciando treinamento por até {epochs} épocas com batch_size={batch_size}...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data, # Pass validation data if available
            callbacks=[early_stopping, reduce_lr, tensorboard_callback, TrainingLogger()],
            verbose=0 # Use custom logger for epoch output, disable Keras default verbose levels
        )
        logger.info("Treinamento concluído.")
        return history

    except tf.errors.ResourceExhaustedError as e:
         logger.error(f"Erro de Recurso Exaurido (OOM - Out of Memory) durante o treinamento: {e}. "
                      "Tente reduzir batch_size, sequence_length ou gru_units.")
         return None
    except Exception as e:
        logger.error(f"Erro inesperado durante o treinamento: {e}", exc_info=True)
        return None


# --- Avaliação e Previsão ---

# evaluate_real_hits remains the same as V2/V3
def evaluate_real_hits(model, X_test, y_test, batch_size=32):
    """Evaluates how many of the top 6 predicted numbers were actually drawn in the test set."""
    logger.info("Avaliando acertos reais nas previsões do conjunto de teste (Top 6)...")
    try:
        # --- Input Validation ---
        if model is None:
             logger.error("Modelo inválido para avaliação de acertos.")
             return None
        if X_test is None or X_test.size == 0 or y_test is None or y_test.size == 0:
             logger.warning("Dados de teste (X_test ou y_test) vazios ou ausentes. Pulando avaliação de acertos.")
             # Return a structure indicating no results
             return { 'hits_per_draw': [], 'avg_hits': 0, 'max_hits': 0,
                      'hits_distribution': {}, 'detailed_hits': [] }
        if X_test.shape[0] != y_test.shape[0]:
             logger.error(f"Inconsistência no número de amostras entre X_test ({X_test.shape[0]}) e y_test ({y_test.shape[0]}).")
             return None
        # Check if model output shape matches y_test features
        if model.output_shape[-1] != y_test.shape[-1]:
             logger.error(f"Inconsistência no número de features entre output do modelo ({model.output_shape[-1]}) e y_test ({y_test.shape[-1]}).")
             return None

        # --- Predict Probabilities ---
        logger.info(f"Realizando previsões no conjunto de teste ({len(X_test)} amostras)...")
        y_pred_probs = model.predict(X_test, batch_size=batch_size)

        # --- Calculate Hits for Each Draw ---
        hits_per_draw = [] # Store number of hits (0-6) for each test draw
        detailed_hits = [] # Store more details for analysis/export

        for i in range(len(y_pred_probs)):
            # Get indices of the top 6 probabilities
            top6_pred_indices = np.argsort(y_pred_probs[i])[-6:]
            # Convert indices (0-59) to numbers (1-60)
            predicted_numbers = sorted((top6_pred_indices + 1).tolist())

            # Get indices where the actual label is 1
            actual_winning_indices = np.where(y_test[i] == 1)[0]
            # Convert indices to numbers
            actual_numbers = sorted((actual_winning_indices + 1).tolist())

            # Find the intersection (common numbers)
            hits = set(predicted_numbers) & set(actual_numbers)
            num_hits = len(hits)

            # Store results
            hits_per_draw.append(num_hits)
            detailed_hits.append({
                'sorteio_index_teste': i, # Index within the test set
                'previstos': predicted_numbers,
                'sorteados': actual_numbers,
                'acertos': sorted(list(hits)), # Store the actual numbers hit
                'num_acertos': num_hits
            })

        # --- Calculate Summary Statistics ---
        num_test_draws = len(hits_per_draw)
        if num_test_draws == 0:
             logger.warning("Nenhum sorteio no teste para calcular estatísticas de acertos.")
             avg_hits, max_hits, hits_distribution = 0, 0, {}
        else:
             avg_hits = np.mean(hits_per_draw)
             max_hits = np.max(hits_per_draw) if hits_per_draw else 0
             # Count occurrences of each hit count (0, 1, 2, ... max_hits)
             hits_distribution = {i: hits_per_draw.count(i) for i in range(max_hits + 1)}

        # --- Log Summary ---
        logger.info("-" * 60)
        logger.info("ANÁLISE DE ACERTOS REAIS (TOP 6 PREVISTOS vs SORTEADOS NO TESTE)")
        logger.info(f"Total de sorteios no teste avaliados: {num_test_draws}")
        if num_test_draws > 0:
             logger.info(f"Média de acertos por sorteio: {avg_hits:.3f}")
             logger.info(f"Máximo de acertos em um único sorteio: {max_hits}")
             logger.info("Distribuição de Acertos:")
             # Sort distribution by number of hits for clarity
             for hits_count in sorted(hits_distribution.keys()):
                 count = hits_distribution[hits_count]
                 if count > 0: # Only log if there were occurrences
                      percentage = (count / num_test_draws) * 100
                      logger.info(f"  - {hits_count} acerto(s): {count} sorteios ({percentage:.1f}%)")

             # Log examples of last few predictions vs actual
             logger.info(f"Exemplo dos últimos {min(5, num_test_draws)} sorteios do teste:")
             for hit_detail in detailed_hits[-min(5, num_test_draws):]:
                 logger.info(f"  - Idx {hit_detail['sorteio_index_teste']}: Prev{hit_detail['previstos']} | Real{hit_detail['sorteados']} -> Hits: {hit_detail['acertos']} ({hit_detail['num_acertos']})")
        logger.info("-" * 60 + "\nAVISO: Acertos passados NÃO garantem acertos futuros.\n" + "-" * 60)

        # Return dictionary with results
        return {
            'hits_per_draw': hits_per_draw,
            'avg_hits': avg_hits,
            'max_hits': max_hits,
            'hits_distribution': hits_distribution,
            'detailed_hits': detailed_hits
        }
    except Exception as e:
        logger.error(f"Erro inesperado ao avaliar acertos reais: {e}", exc_info=True)
        return None

# evaluate_model remains the same as V2/V3
def evaluate_model(model, X_test, y_test, batch_size=32):
    """Evaluates the model on the test set using standard Keras metrics and custom real hits evaluation."""
    logger.info("Avaliando o modelo final no conjunto de teste...")
    evaluation_summary = {'basic_metrics': {}, 'real_hits': None} # Initialize results dict
    try:
        # --- Input Validation ---
        if model is None:
            logger.error("Modelo inválido para avaliação.")
            return None # Cannot proceed without model
        if X_test is None or X_test.size == 0 or y_test is None or y_test.size == 0:
            logger.warning("Dados de teste (X_test ou y_test) vazios ou ausentes. Pulando avaliação completa.")
            # Return default summary indicating no evaluation done
            return evaluation_summary

        # --- 1. Keras Standard Evaluation ---
        logger.info("Calculando métricas padrão Keras (Loss, Accuracy, AUC)...")
        try:
            # Use model.evaluate to get loss and metrics defined during compile
            results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0) # verbose=0 to avoid double logging
            # Create a dictionary mapping metric names to results
            basic_metrics_dict = dict(zip(model.metrics_names, results))
            evaluation_summary['basic_metrics'] = basic_metrics_dict
            logger.info("Métricas padrão Keras calculadas:")
            for name, value in basic_metrics_dict.items():
                logger.info(f"  - {name}: {value:.4f}")
        except Exception as e_eval:
             logger.error(f"Erro durante model.evaluate: {e_eval}", exc_info=True)
             # Keep basic_metrics empty in summary if evaluation failed

        # --- 2. Real Hits Evaluation (Top 6) ---
        logger.info("Calculando acertos reais (Top 6)...")
        real_hits_results = evaluate_real_hits(model, X_test, y_test, batch_size)
        if real_hits_results is None:
            logger.error("Falha na avaliação de acertos reais. Resultados de acertos não estarão disponíveis.")
        evaluation_summary['real_hits'] = real_hits_results # Store results (or None if failed)

        # --- Log Combined Summary ---
        logger.info("-" * 60 + "\nResumo da Avaliação no Conjunto de Teste\n" + "-" * 60)
        logger.info("1. Métricas Padrão Keras:")
        if evaluation_summary['basic_metrics']:
            for name, value in evaluation_summary['basic_metrics'].items():
                logger.info(f"  - {name}: {value:.4f}")
        else:
            logger.info("  N/A (falha no cálculo)")

        logger.info("\n2. Estatísticas de Acertos Reais (Top 6 Previstos vs Sorteados):")
        if real_hits_results: # Check if the evaluation succeeded
             logger.info(f"  - Média acertos: {real_hits_results.get('avg_hits', 'N/A'):.3f}")
             logger.info(f"  - Máx acertos: {real_hits_results.get('max_hits', 'N/A')}")
             logger.info("  - Distribuição:")
             total_test_draws = len(real_hits_results.get('hits_per_draw', []))
             if total_test_draws > 0:
                 hits_dist = real_hits_results.get('hits_distribution', {})
                 if hits_dist:
                     for hits_count in sorted(hits_dist.keys()):
                          count = hits_dist[hits_count]
                          if count > 0:
                              percentage = (count / total_test_draws) * 100
                              logger.info(f"    * {hits_count} acerto(s): {count} ({percentage:.1f}%)")
                 else:
                      logger.info("    Distribuição não disponível.")
             else:
                 logger.info("    N/A (sem sorteios no teste para calcular distribuição)")
        else:
             logger.info("  N/A (falha no cálculo de acertos reais)")
        logger.info("-" * 60 + "\nLembrete: Métricas refletem desempenho no passado e em dados não vistos durante treino.\n" + "-" * 60)

        return evaluation_summary

    except Exception as e:
        logger.error(f"Erro inesperado durante a avaliação final do modelo: {e}", exc_info=True)
        # Return the summary which might contain partial results
        return evaluation_summary


# predict_next_draw remains the same as V3 (uses both scalers)
def predict_next_draw(model, last_sequence_labels, last_sequence_time_raw, last_sequence_stat_raw,
                      time_scaler, stat_scaler, mlb, num_predictions=6):
    """
    Prepares the final sequence using the last N available draws (labels and raw features),
    scales the features using the provided fitted scalers, and predicts the next draw probabilities.
    Returns the top N predicted numbers and the full probability vector.
    """
    logger.info(f"Preparando última sequência e fazendo previsão para o PRÓXIMO sorteio (Top {num_predictions})...")
    try:
        # --- Input Validation ---
        if model is None: logger.error("Modelo inválido para previsão."); return None, None
        if time_scaler is None or stat_scaler is None: logger.error("Scalers (time ou stat) inválidos para previsão."); return None, None
        if mlb is None: logger.error("MultiLabelBinarizer (mlb) inválido."); return None, None # Needed for number mapping if used

        # Get expected sequence length from model input shape
        try:
            seq_len = model.input_shape[1]
            if seq_len is None: raise ValueError("Model input sequence length is None.")
        except Exception as e_shape:
             logger.error(f"Não foi possível determinar sequence_length da input_shape do modelo ({model.input_shape}): {e_shape}")
             # Fallback to config if available, otherwise fail
             if 'sequence_length' in config:
                 seq_len = config['sequence_length']
                 logger.warning(f"Usando sequence_length ({seq_len}) da configuração como fallback.")
             else:
                 logger.error("sequence_length não encontrado na config. Abortando previsão.")
                 return None, None

        # Validate shapes of the last sequence parts
        error_msg = ""
        if last_sequence_labels is None or last_sequence_labels.shape[0] != seq_len:
            error_msg += f" last_sequence_labels shape ({last_sequence_labels.shape if last_sequence_labels is not None else 'None'}) != {seq_len}."
        if last_sequence_time_raw is None or last_sequence_time_raw.shape[0] != seq_len:
            error_msg += f" last_sequence_time_raw shape ({last_sequence_time_raw.shape if last_sequence_time_raw is not None else 'None'}) != {seq_len}."
        if last_sequence_stat_raw is None or last_sequence_stat_raw.shape[0] != seq_len:
            error_msg += f" last_sequence_stat_raw shape ({last_sequence_stat_raw.shape if last_sequence_stat_raw is not None else 'None'}) != {seq_len}."

        if error_msg:
             logger.error(f"Última sequência inválida ou tamanho incorreto (esperado {seq_len} amostras). Erros:{error_msg}")
             return None, None
        # Further check feature dimensions match scalers / model input
        try:
             if last_sequence_labels.shape[1] != config.get('num_features_base'): # Use config as source of truth
                  error_msg += f" Dimensão features base ({last_sequence_labels.shape[1]}) != config ({config.get('num_features_base')})."
             if last_sequence_time_raw.shape[1] != time_scaler.n_features_in_:
                  error_msg += f" Dimensão features tempo raw ({last_sequence_time_raw.shape[1]}) != scaler ({time_scaler.n_features_in_})."
             if last_sequence_stat_raw.shape[1] != stat_scaler.n_features_in_:
                  error_msg += f" Dimensão features stat raw ({last_sequence_stat_raw.shape[1]}) != scaler ({stat_scaler.n_features_in_})."
             if error_msg:
                  logger.error(f"Inconsistência nas dimensões das features da última sequência:{error_msg}")
                  return None, None
        except AttributeError:
             logger.error("Erro ao acessar atributos dos scalers (n_features_in_). Scalers foram ajustados?")
             return None, None
        except Exception as e_dim:
             logger.error(f"Erro ao validar dimensões da última sequência: {e_dim}")
             return None, None


        # --- 1. Scale the Raw Time and Statistical Features ---
        logger.debug("Escalando features raw de tempo e estatísticas da última sequência...")
        try:
            last_sequence_time_scaled = time_scaler.transform(last_sequence_time_raw)
            last_sequence_stat_scaled = stat_scaler.transform(last_sequence_stat_raw)
        except Exception as e_scale:
             logger.error(f"Erro ao escalar features da última sequência: {e_scale}", exc_info=True)
             return None, None

        # --- 2. Combine Labels, Scaled Time, Scaled Stats ---
        logger.debug("Combinando labels e features escaladas para a sequência final...")
        try:
            last_sequence_combined = np.concatenate(
                (last_sequence_labels, last_sequence_time_scaled, last_sequence_stat_scaled),
                axis=-1 # Concatenate along the feature dimension
            ).astype(np.float32) # Ensure correct dtype for the model
        except ValueError as e_concat:
            logger.error(f"Erro ao concatenar features da última sequência (verifique shapes): {e_concat}", exc_info=True)
            logger.error(f"  Shapes: Labels({last_sequence_labels.shape}), Time({last_sequence_time_scaled.shape}), Stat({last_sequence_stat_scaled.shape})")
            return None, None
        except Exception as e_comb:
            logger.error(f"Erro inesperado ao combinar features da última sequência: {e_comb}", exc_info=True)
            return None, None

        # Check if the combined feature dimension matches the model's expected input feature dimension
        expected_total_features = model.input_shape[-1]
        if last_sequence_combined.shape[-1] != expected_total_features:
            logger.error(f"Número total de features combinadas ({last_sequence_combined.shape[-1]}) "
                         f"não corresponde à entrada esperada pelo modelo ({expected_total_features}).")
            return None, None

        # --- 3. Add Batch Dimension and Predict ---
        # Model expects input shape (batch_size, sequence_length, num_features_total)
        last_sequence_batch = np.expand_dims(last_sequence_combined, axis=0) # Add batch dimension
        logger.info(f"Shape da sequência final formatada para previsão: {last_sequence_batch.shape}")

        try:
            predicted_probabilities_batch = model.predict(last_sequence_batch)
            # Result is shape (1, num_features_base), extract the first element
            predicted_probabilities = predicted_probabilities_batch[0]
            logger.debug(f"Probabilidades brutas previstas (shape {predicted_probabilities.shape}).")
        except Exception as e_pred:
             logger.error(f"Erro durante a predição do modelo: {e_pred}", exc_info=True)
             return None, None

        # --- 4. Process Predictions ---
        # Validate output shape
        expected_output_shape = model.output_shape[-1] # Should be num_features_base
        if predicted_probabilities.shape[0] != expected_output_shape:
            logger.error(f"Shape inesperado da saída da previsão: {predicted_probabilities.shape}. Esperado: ({expected_output_shape},)")
            return None, None

        # Get indices of the top N predictions
        # Argsort sorts in ascending order, take the last N for highest probabilities
        predicted_indices = np.argsort(predicted_probabilities)[-num_predictions:]
        # Convert indices (0 to num_features_base-1) to actual numbers (1 to num_features_base)
        # Adding 1 maps the index to the number
        predicted_numbers = sorted((predicted_indices + 1).tolist())

        # Get confidence scores (probabilities) for the predicted numbers
        confidence_scores = predicted_probabilities[predicted_indices]

        # Calculate stats for predicted numbers' confidence
        avg_confidence = np.mean(confidence_scores) if confidence_scores.size > 0 else 0.0
        max_confidence = np.max(confidence_scores) if confidence_scores.size > 0 else 0.0
        min_confidence = np.min(confidence_scores) if confidence_scores.size > 0 else 0.0

        # --- 5. Log Results ---
        logger.info("-" * 50)
        logger.info(f"PREVISÃO PARA O PRÓXIMO SORTEIO")
        logger.info(f"Números mais prováveis ({num_predictions}): {predicted_numbers}")
        logger.info(f"Confiança (Probabilidade Média/Máx/Mín): {avg_confidence:.4f} / {max_confidence:.4f} / {min_confidence:.4f}")
        logger.info("Probabilidades individuais dos números previstos (ordenado por prob):")
        # Sort predicted indices by their probabilities in descending order for logging
        sorted_pred_indices_by_prob = predicted_indices[np.argsort(confidence_scores)[::-1]]
        for num_idx in sorted_pred_indices_by_prob:
            number = num_idx + 1
            probability = predicted_probabilities[num_idx]
            logger.info(f"  - Número {number:02d}: {probability:.4f}") # {:02d} formats as two digits (e.g., 03)
        logger.info("-" * 50 + "\nAVISO CRÍTICO: Esta é uma previsão estatística experimental. NÃO HÁ GARANTIA DE ACERTO. Jogue com responsabilidade.\n" + "-" * 50)

        # Return the list of predicted numbers and the full probability vector
        return predicted_numbers, predicted_probabilities

    except Exception as e:
        logger.error(f"Erro inesperado durante a previsão do próximo sorteio: {e}", exc_info=True)
        return None, None


# --- Visualização e Exportação ---

# plot_training_history remains the same as V3
def plot_training_history(history, filename=None):
    """ Plots training history (Loss, Accuracy, AUC, LR) saved from model.fit. """
    # Use default filename if none provided
    if filename is None:
        # Ensure filename uses the output directory
        filename = os.path.join(output_dir, 'training_history_v3.png')
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    logger.info(f"Gerando gráficos do histórico de treinamento em {filename}...")
    try:
        # Validate history object
        if history is None or not hasattr(history, 'history') or not history.history:
            logger.error("Histórico de treinamento inválido ou vazio fornecido para plotagem.")
            return
        history_dict = history.history

        # Identify available metrics (handle cases where some metrics might be missing)
        metrics_to_plot = []
        val_metrics_to_plot = []
        if 'loss' in history_dict: metrics_to_plot.append('loss')
        if 'binary_accuracy' in history_dict: metrics_to_plot.append('binary_accuracy')
        if 'auc' in history_dict: metrics_to_plot.append('auc') # Assumes AUC metric name is 'auc'

        # Check for corresponding validation metrics
        for m in metrics_to_plot:
            if f'val_{m}' in history_dict:
                val_metrics_to_plot.append(f'val_{m}')

        # Check for learning rate
        has_lr = 'lr' in history_dict

        num_plots = len(metrics_to_plot) + (1 if has_lr else 0)
        if num_plots == 0:
             logger.error("Nenhuma métrica ('loss', 'binary_accuracy', 'auc', 'lr') encontrada no histórico para plotar.")
             return

        # Determine subplot layout (e.g., 2 columns)
        num_cols = 2
        num_rows = (num_plots + num_cols - 1) // num_cols # Ceiling division

        plt.figure(figsize=(max(12, num_cols * 6), num_rows * 5)) # Adjust size based on layout

        plot_index = 1
        # Plot Loss, Accuracy, AUC
        for metric in metrics_to_plot:
            plt.subplot(num_rows, num_cols, plot_index)
            plt.plot(history_dict[metric], label=f'Treino {metric.capitalize().replace("_", " ")}')
            # Plot validation metric if available
            if f'val_{metric}' in history_dict:
                plt.plot(history_dict[f'val_{metric}'], label=f'Validação {metric.capitalize().replace("_", " ")}')

            # Find best epoch based on validation metric if available, else training metric
            best_epoch = -1
            best_value = float('inf') if 'loss' in metric else float('-inf')
            monitor_metric = f'val_{metric}' if f'val_{metric}' in history_dict else metric

            try:
                 if 'loss' in monitor_metric:
                      best_epoch = np.argmin(history_dict[monitor_metric])
                      best_value = history_dict[monitor_metric][best_epoch]
                 else: # Accuracy, AUC - maximize
                      best_epoch = np.argmax(history_dict[monitor_metric])
                      best_value = history_dict[monitor_metric][best_epoch]

                 # Add marker for the best epoch
                 if best_epoch != -1:
                      plt.scatter(best_epoch, best_value, color='red', s=100, zorder=5,
                                  label=f'Melhor {monitor_metric.split("_")[-1].capitalize()} (Época {best_epoch+1}): {best_value:.4f}')
            except Exception as e_best:
                 logger.warning(f"Não foi possível determinar a melhor época para {monitor_metric}: {e_best}")


            plt.title(f'{metric.replace("_", " ").capitalize()} por Época')
            plt.xlabel('Época')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plot_index += 1

        # Plot Learning Rate if available
        if has_lr:
             plt.subplot(num_rows, num_cols, plot_index)
             plt.plot(history_dict['lr'], label='Taxa Aprendizado (LR)')
             plt.title('Taxa de Aprendizado por Época')
             plt.xlabel('Época')
             plt.ylabel('Learning Rate')
             plt.legend()
             plt.grid(True, linestyle='--', alpha=0.6)
             # Use scientific notation if LR gets very small
             plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
             plot_index +=1

        plt.tight_layout() # Adjust spacing between plots
        plt.savefig(filename)
        logger.info(f"Gráficos do histórico de treinamento salvos em '{filename}'")
        plt.close() # Close the plot figure to free memory

    except Exception as e:
        logger.error(f"Erro ao gerar gráficos de treinamento: {e}", exc_info=True)


# plot_prediction_analysis remains the same as V3
def plot_prediction_analysis(predicted_numbers, predicted_probabilities, df_full_valid, sequence_length, filename=None):
    """ Generates visual analysis comparing predictions, probabilities, and recent historical frequency. """
    if filename is None:
        filename = os.path.join(output_dir, 'prediction_analysis_v3.png')
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    logger.info(f"Gerando análise visual das previsões em {filename}...")
    try:
        # --- Input Validation ---
        if predicted_numbers is None or predicted_probabilities is None:
            logger.error("Dados de previsão inválidos (números ou probabilidades) para análise visual.")
            return
        if not isinstance(predicted_probabilities, np.ndarray) or predicted_probabilities.ndim != 1:
            logger.error("predicted_probabilities deve ser um array NumPy 1D.")
            return
        num_features_base = len(predicted_probabilities)
        if num_features_base == 0:
            logger.error("Vetor de probabilidades previsto está vazio.")
            return
        if df_full_valid is None or df_full_valid.empty:
            logger.error("DataFrame histórico ('df_full_valid') inválido ou vazio para análise.")
            return
        if sequence_length <= 0:
             logger.error(f"sequence_length ({sequence_length}) inválido para análise de frequência recente.")
             return
        if len(df_full_valid) < sequence_length:
             logger.warning(f"Histórico ({len(df_full_valid)}) mais curto que sequence_length ({sequence_length}). Frequência será baseada em menos sorteios.")
             sequence_length = len(df_full_valid) # Adjust sequence length for frequency calc


        # --- Prepare Data ---
        all_numbers = np.arange(1, num_features_base + 1) # Numbers 1 to 60 (or base)
        predicted_numbers_arr = np.array(predicted_numbers) # Ensure it's an array
        # Get probabilities corresponding to the predicted numbers (indices are number-1)
        probs_for_predicted = predicted_probabilities[predicted_numbers_arr - 1]

        # Calculate Recent Frequency
        bola_cols = [f'Bola{i+1}' for i in range(6)] # Assuming 6 balls
        # Check if bola_cols exist
        if not all(col in df_full_valid.columns for col in bola_cols):
             logger.error(f"Colunas das bolas ({bola_cols}) não encontradas no df_full_valid para cálculo de frequência.")
             return

        # Take the last 'sequence_length' draws
        last_n_draws_df = df_full_valid.iloc[-sequence_length:]
        # Flatten numbers from these draws
        try:
            # Concatenate columns, drop NaNs just in case, convert to int
            last_numbers_flat = pd.concat([last_n_draws_df[col] for col in bola_cols]).dropna().astype(int).values
        except Exception as e_flat:
            logger.error(f"Erro ao achatar números para cálculo de frequência: {e_flat}")
            return

        # Calculate frequency of each number (1 to num_features_base)
        number_freq = np.zeros(num_features_base) # Initialize frequency array
        if last_numbers_flat.size > 0:
            unique_nums, counts = np.unique(last_numbers_flat, return_counts=True)
            # Filter out any numbers outside the valid range [1, num_features_base] that might sneak in
            valid_mask = (unique_nums >= 1) & (unique_nums <= num_features_base)
            valid_unique_nums = unique_nums[valid_mask]
            valid_counts = counts[valid_mask]
            # Update frequency array using valid numbers (indices are number-1)
            if valid_unique_nums.size > 0:
                number_freq[valid_unique_nums - 1] = valid_counts

        # Get recent frequency for the predicted numbers
        freq_for_predicted = number_freq[predicted_numbers_arr - 1]

        # --- Create Plots ---
        plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style
        fig, axes = plt.subplots(2, 2, figsize=(16, 12)) # Create 2x2 grid of subplots
        fig.suptitle('Análise da Previsão vs Histórico Recente', fontsize=16, y=1.02)

        # Plot 1: All Predicted Probabilities
        ax1 = axes[0, 0]
        ax1.bar(all_numbers, predicted_probabilities, width=0.8, color='skyblue', edgecolor='black', linewidth=0.5)
        ax1.set_title(f'Probabilidades Previstas (Todos os {num_features_base} Números)')
        ax1.set_xlabel('Número')
        ax1.set_ylabel('Probabilidade Prevista')
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
        # Set x-axis ticks to be clearer (e.g., every 5 numbers)
        ax1.set_xticks(np.arange(0, num_features_base + 1, 5))
        ax1.set_xlim(0.5, num_features_base + 0.5)

        # Plot 2: Probabilities of the Top N Predicted Numbers
        ax2 = axes[0, 1]
        bars = ax2.bar(predicted_numbers_arr, probs_for_predicted, width=0.6, color='coral', edgecolor='black', linewidth=0.5)
        ax2.set_title(f'Probabilidades dos {len(predicted_numbers)} Números Previstos')
        ax2.set_xlabel('Número Previsto')
        ax2.set_ylabel('Probabilidade Prevista')
        ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
        # Add probability values on top of bars
        ax2.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
        # Set y-limit to give space for labels
        if probs_for_predicted.size > 0:
            ax2.set_ylim(0, max(probs_for_predicted) * 1.20) # Increase padding
        # Ensure only predicted numbers are shown as ticks
        ax2.set_xticks(predicted_numbers_arr)
        ax2.set_xticklabels(predicted_numbers_arr) # Explicitly set labels

        # Plot 3: Recent Frequency (Last N Draws)
        ax3 = axes[1, 0]
        ax3.bar(all_numbers, number_freq, width=0.8, color='lightgreen', edgecolor='black', linewidth=0.5)
        ax3.set_title(f'Frequência nos Últimos {sequence_length} Sorteios Históricos')
        ax3.set_xlabel('Número')
        ax3.set_ylabel('Frequência (Contagem)')
        ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax3.set_xticks(np.arange(0, num_features_base + 1, 5))
        ax3.set_xlim(0.5, num_features_base + 0.5)
        # Set y-ticks to be integers
        if number_freq.max() > 0:
            ax3.set_yticks(np.arange(0, int(number_freq.max()) + 2, 1)) # +2 to ensure top tick is visible

        # Plot 4: Scatter Plot - Recent Frequency vs Predicted Probability
        ax4 = axes[1, 1]
        # Scatter all numbers
        ax4.scatter(number_freq, predicted_probabilities, alpha=0.5, label='Outros Números', s=30)
        # Highlight predicted numbers
        ax4.scatter(freq_for_predicted, probs_for_predicted, color='red', s=100, label='Previstos', edgecolors='black', zorder=5)
        ax4.set_title('Frequência Recente vs Probabilidade Prevista')
        ax4.set_xlabel(f'Frequência nos Últimos {sequence_length} Sorteios')
        ax4.set_ylabel('Probabilidade Prevista')
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend()
        # Add text labels for predicted numbers
        for i, num in enumerate(predicted_numbers_arr):
            # Add slight offset to text position for better visibility
            text_x = freq_for_predicted[i] + 0.03 * (number_freq.max() if number_freq.max() > 0 else 1)
            text_y = probs_for_predicted[i]
            ax4.text(text_x, text_y, str(num), fontsize=9, verticalalignment='center')

        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent title overlap
        plt.savefig(filename)
        logger.info(f"Análise visual das previsões salva em '{filename}'")
        plt.close(fig) # Close the figure object explicitly

    except Exception as e:
        logger.error(f"Erro ao gerar análise visual das previsões: {e}", exc_info=True)


# plot_hits_over_time remains the same as V3
def plot_hits_over_time(model, X_test, y_test, mlb, filename=None):
    """ Plots the number of hits (top 6 predicted vs actual) for each draw in the test set. """
    if filename is None:
        filename = os.path.join(output_dir, 'hits_over_time_v3.png')
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    logger.info(f"Gerando gráfico de acertos ao longo do tempo no teste em {filename}...")
    try:
        # --- Input Validation ---
        if model is None: logger.error("Modelo inválido para plotar acertos."); return None
        if X_test is None or X_test.size == 0 or y_test is None or y_test.size == 0:
            logger.warning("Dados de teste (X_test ou y_test) vazios ou ausentes. Pulando gráfico de acertos.")
            return None
        if X_test.shape[0] != y_test.shape[0]:
             logger.error(f"Inconsistência no número de amostras entre X_test ({X_test.shape[0]}) e y_test ({y_test.shape[0]}) para plotar acertos.")
             return None
        # Check if model output shape matches y_test features
        if model.output_shape[-1] != y_test.shape[-1]:
             logger.error(f"Inconsistência no número de features entre output do modelo ({model.output_shape[-1]}) e y_test ({y_test.shape[-1]}) para plotar acertos.")
             return None
        # MLB might not be strictly necessary here if we only compare indices, but good practice to have it if needed later
        if mlb is None: logger.warning("MLB não fornecido para plot_hits_over_time (não essencial para contagem de hits).")

        # --- Predict and Calculate Hits ---
        logger.info("Calculando acertos no conjunto de teste para plotagem...")
        y_pred_probs_test = model.predict(X_test)
        hits_per_draw = []
        num_test_draws = len(y_pred_probs_test)

        for i in range(num_test_draws):
            top6_pred_indices = np.argsort(y_pred_probs_test[i])[-6:] # Indices 0-59
            actual_winning_indices = np.where(y_test[i] == 1)[0] # Indices 0-59
            # Intersection of index sets gives the count of hits directly
            num_hits = len(set(top6_pred_indices) & set(actual_winning_indices))
            hits_per_draw.append(num_hits)

        if not hits_per_draw:
            logger.warning("Nenhum resultado de acerto calculado no teste para plotagem.")
            return None

        # --- Create Plot ---
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(15, 6))
        # Plot individual hits per draw
        plt.plot(range(num_test_draws), hits_per_draw, marker='o', linestyle='-', markersize=4,
                 label='Nº Acertos (Top 6) / Sorteio Teste', alpha=0.7)

        # Calculate and plot rolling average if enough data points
        if num_test_draws >= 10:
             # Use pandas for easy rolling mean calculation
             rolling_avg = pd.Series(hits_per_draw).rolling(window=10, min_periods=1).mean()
             plt.plot(range(num_test_draws), rolling_avg, linestyle='--', color='red', linewidth=2,
                      label=f'Média Móvel ({10} Sorteios)')

        plt.xlabel("Índice do Sorteio no Conjunto de Teste")
        plt.ylabel("Número de Acertos (0 a 6)")
        plt.title("Acertos do Modelo (Top 6) ao Longo do Conjunto de Teste Histórico")
        # Set y-axis ticks to show integers 0-6 clearly
        plt.yticks(np.arange(0, 7, 1))
        plt.ylim(bottom=-0.2, top=6.2) # Give slight margin
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()

        plt.savefig(filename)
        logger.info(f"Gráfico de acertos ao longo do tempo salvo em '{filename}'")
        plt.close() # Close plot figure

        return hits_per_draw # Return the list of hits per draw

    except Exception as e:
        logger.error(f"Erro ao gerar gráfico de acertos ao longo do tempo: {e}", exc_info=True)
        return None


# export_results remains the same as V3
def export_results(df_full_valid, predicted_numbers, predicted_probabilities, evaluation_results, config):
    """ Exports historical data, predictions, evaluation metrics, and configuration to an Excel file. """
    export_file = config.get('export_file')
    if not export_file:
        logger.error("Nome do arquivo de exportação ('export_file') não definido na configuração.")
        return

    logger.info(f"Exportando resultados para Excel: {export_file}...")
    try:
        # --- Input Validation ---
        if df_full_valid is None or df_full_valid.empty:
            logger.error("DataFrame histórico ('df_full_valid') inválido ou vazio. Não é possível exportar.")
            return
        if predicted_numbers is None or predicted_probabilities is None:
            logger.error("Dados de previsão (números ou probabilidades) inválidos. Não é possível exportar.")
            return
        if not isinstance(predicted_probabilities, np.ndarray) or predicted_probabilities.ndim != 1:
            logger.error("predicted_probabilities deve ser um array NumPy 1D para exportação.")
            return
        num_features_base = len(predicted_probabilities)
        if num_features_base == 0:
            logger.error("Vetor de probabilidades previsto está vazio.")
            return
        # Handle potentially missing evaluation results gracefully
        if evaluation_results is None:
            logger.warning("Resultados da avaliação ('evaluation_results') não fornecidos. Exportando sem eles.")
            evaluation_results = {'basic_metrics': {}, 'real_hits': None} # Default structure


        # --- Prepare DataFrames for Export ---

        # 1. Predictions and Probabilities Sheet
        logger.debug("Preparando aba 'Previsao_Probabilidades'...")
        predictions_df = pd.DataFrame({
            'Número': range(1, num_features_base + 1),
            'Probabilidade_Prevista': predicted_probabilities
        })
        # Mark which numbers were in the top 6 prediction
        predictions_df['Previsto_Top_6'] = predictions_df['Número'].isin(predicted_numbers)
        # Sort by probability descending
        predictions_df = predictions_df.sort_values('Probabilidade_Prevista', ascending=False).reset_index(drop=True)

        # 2. Basic Evaluation Metrics Sheet
        logger.debug("Preparando aba 'Metricas_Avaliacao'...")
        basic_metrics_dict = evaluation_results.get('basic_metrics', {})
        if not basic_metrics_dict:
            metrics_df = pd.DataFrame({'Métrica': ['N/A - Avaliação não realizada ou falhou'], 'Valor': ['N/A']})
        else:
            metrics_df = pd.DataFrame({
                'Métrica': list(basic_metrics_dict.keys()),
                'Valor': [f"{v:.5f}" if isinstance(v, (float, int)) else str(v) for v in basic_metrics_dict.values()]
            })

        # 3. Real Hits Evaluation Sheets
        logger.debug("Preparando abas de 'Acertos_Reais'...")
        real_hits_results = evaluation_results.get('real_hits')
        hits_summary_df = pd.DataFrame({'Estatística': ['N/A'], 'Valor': ['N/A']}) # Default
        hits_dist_df = pd.DataFrame({'Número Acertos': ['N/A'], 'Qtd Sorteios': ['N/A'], 'Porcentagem (%)': ['N/A']}) # Default
        detailed_hits_df = pd.DataFrame({'Info': ['Detalhes de acertos não disponíveis ou avaliação não realizada']}) # Default

        if real_hits_results and isinstance(real_hits_results, dict):
            total_draws = len(real_hits_results.get('hits_per_draw', []))
            if total_draws > 0:
                # Summary Sheet
                hits_summary_df = pd.DataFrame({
                    'Estatística': ['Média Acertos (Top 6)', 'Máx Acertos (Top 6)', 'Total Sorteios Teste'],
                    'Valor': [
                        f"{real_hits_results.get('avg_hits', 'N/A'):.3f}",
                        f"{real_hits_results.get('max_hits', 'N/A')}",
                        total_draws
                    ]
                })
                # Distribution Sheet
                hits_dist = real_hits_results.get('hits_distribution')
                if hits_dist and isinstance(hits_dist, dict):
                    hits_dist_df = pd.DataFrame({
                        'Número Acertos': list(hits_dist.keys()),
                        'Qtd Sorteios': list(hits_dist.values())
                    })
                    # Calculate percentage if possible
                    hits_dist_df['Porcentagem (%)'] = hits_dist_df['Qtd Sorteios'].apply(
                        lambda c: f"{(c / total_draws) * 100:.1f}%" if total_draws > 0 else 'N/A'
                    )
                    hits_dist_df = hits_dist_df.sort_values('Número Acertos').reset_index(drop=True)
                else:
                     logger.warning("Distribuição de acertos ausente ou inválida nos resultados.")
                # Detailed Hits Sheet
                detailed_hits = real_hits_results.get('detailed_hits')
                if detailed_hits and isinstance(detailed_hits, list) and len(detailed_hits) > 0:
                     try:
                        detailed_hits_df = pd.DataFrame(detailed_hits)
                        # Format list columns for better readability in Excel
                        for col in ['previstos', 'sorteados', 'acertos']:
                            if col in detailed_hits_df.columns:
                                detailed_hits_df[col] = detailed_hits_df[col].apply(
                                    lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x
                                )
                     except Exception as e_det:
                        logger.error(f"Erro ao formatar DataFrame de detalhes de acertos: {e_det}")
                        detailed_hits_df = pd.DataFrame({'Info': ['Erro ao formatar detalhes de acertos']})
                else:
                     logger.warning("Detalhes de acertos ausentes ou vazios nos resultados.")
            else:
                 logger.warning("Resultados de acertos reais presentes, mas sem sorteios avaliados (hits_per_draw vazio).")
        else:
            logger.warning("Resultados de acertos reais ('real_hits') não encontrados ou em formato inválido.")

        # 4. Historical Data Used Sheet
        logger.debug("Preparando aba 'Historico_Utilizado'...")
        # Export the validated and potentially filtered df_full_valid

        # 5. Configuration Used Sheet
        logger.debug("Preparando aba 'Configuracao_Usada'...")
        # Convert config dict to DataFrame, handle nested dicts like hyperparameter_search
        config_items = []
        for k, v in config.items():
             if isinstance(v, dict): # Flatten nested dicts slightly for readability
                 config_items.append([k, json.dumps(v, indent=2)]) # Store as JSON string
             elif isinstance(v, list):
                  config_items.append([k, ', '.join(map(str,v))]) # Join lists
             else:
                 config_items.append([k, v])
        config_df = pd.DataFrame(config_items, columns=['Parametro', 'Valor'])


        # --- Write to Excel ---
        logger.info("Escrevendo abas no arquivo Excel...")
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(export_file), exist_ok=True)

        with pd.ExcelWriter(export_file, engine='openpyxl') as writer:
            # Write each DataFrame to a separate sheet
            predictions_df.to_excel(writer, sheet_name='Previsao_Probabilidades', index=False)
            logger.debug(" -> Aba Previsao_Probabilidades OK.")
            metrics_df.to_excel(writer, sheet_name='Metricas_Avaliacao', index=False)
            logger.debug(" -> Aba Metricas_Avaliacao OK.")
            hits_summary_df.to_excel(writer, sheet_name='Sumario_Acertos_Reais', index=False)
            logger.debug(" -> Aba Sumario_Acertos_Reais OK.")
            hits_dist_df.to_excel(writer, sheet_name='Distribuicao_Acertos_Reais', index=False)
            logger.debug(" -> Aba Distribuicao_Acertos_Reais OK.")
            if detailed_hits_df is not None and not detailed_hits_df.empty:
                 detailed_hits_df.to_excel(writer, sheet_name='Detalhes_Acertos_Teste', index=False)
                 logger.debug(" -> Aba Detalhes_Acertos_Teste OK.")
            else:
                 logger.debug(" -> Aba Detalhes_Acertos_Teste pulada (vazia ou erro).")
            df_full_valid.to_excel(writer, sheet_name='Historico_Utilizado', index=False)
            logger.debug(" -> Aba Historico_Utilizado OK.")
            config_df.to_excel(writer, sheet_name='Configuracao_Usada', index=False)
            logger.debug(" -> Aba Configuracao_Usada OK.")

        logger.info(f"Resultados exportados com sucesso para '{export_file}'")

    except PermissionError:
        logger.error(f"Erro de Permissão ao tentar escrever o arquivo Excel '{export_file}'. "
                     "Verifique se o arquivo está aberto ou se há permissões de escrita na pasta.")
    except Exception as e:
        logger.error(f"Erro inesperado ao exportar resultados para Excel: {e}", exc_info=True)


# validate_config updated for V3.1 (dynamic time features)
def validate_config(config, check_total_features=False):
    """
    Valida as configurações V3.1.
    Se check_total_features=True, também valida num_features_time e num_features_total (após cálculo).
    """
    stage = "base" if not check_total_features else "final"
    logger.info(f"Validando configuração V3.1 ({stage})...")
    is_valid = True
    try:
        # --- Required Fields (Base Validation) ---
        required_base_fields = [
            'data_url', 'data_file', 'export_file', 'sequence_length', 'num_features_base',
            'num_features_statistical', 'rolling_freq_windows', 'gru_units', 'use_batch_norm',
            'dropout_rate', 'epochs', 'batch_size', 'test_size_ratio', 'validation_split_ratio',
            'cache_duration_hours', 'cache_dir', 'tensorboard_log_dir',
            'test_hyperparameters', # Even if False, key should exist
            # hyperparameter_search can be None, but key should exist if test_hyperparameters is True
        ]
        for field in required_base_fields:
            if field not in config:
                logger.error(f"Campo obrigatório ausente na configuração ({stage}): {field}")
                is_valid = False

        if not is_valid: return False # Early exit if basic fields missing

        # --- Type/Value Checks (Always Performed) ---
        if config.get('data_url') is None and config.get('data_file') is None:
             logger.error("É necessário definir 'data_url' ou 'data_file'.")
             is_valid = False
        if not isinstance(config['sequence_length'], int) or config['sequence_length'] < 1:
             logger.error(f"sequence_length ({config['sequence_length']}) inválido (deve ser int >= 1).")
             is_valid = False
        if not isinstance(config['num_features_base'], int) or config['num_features_base'] <= 0:
             logger.error(f"num_features_base ({config['num_features_base']}) inválido (deve ser int > 0).")
             is_valid = False
        if not isinstance(config['num_features_statistical'], int) or config['num_features_statistical'] < 0:
             logger.error(f"num_features_statistical ({config['num_features_statistical']}) inválido (deve ser int >= 0).")
             is_valid = False
        if not isinstance(config['rolling_freq_windows'], list) or not all(isinstance(x, int) and x > 0 for x in config['rolling_freq_windows']):
             logger.error(f"rolling_freq_windows ({config['rolling_freq_windows']}) deve ser uma lista de inteiros positivos.")
             is_valid = False
        if not isinstance(config['gru_units'], int) or config['gru_units'] < 1:
             logger.error(f"gru_units ({config['gru_units']}) inválido (deve ser int >= 1).")
             is_valid = False
        if not isinstance(config['use_batch_norm'], bool):
             logger.error(f"use_batch_norm ({config['use_batch_norm']}) inválido (deve ser true ou false).")
             is_valid = False
        if not isinstance(config['dropout_rate'], (int, float)) or not 0 <= config['dropout_rate'] < 1:
             logger.error(f"dropout_rate ({config['dropout_rate']}) inválido (deve ser float/int em [0, 1)).")
             is_valid = False
        if not isinstance(config['epochs'], int) or config['epochs'] < 1:
             logger.error(f"epochs ({config['epochs']}) inválido (deve ser int >= 1).")
             is_valid = False
        if not isinstance(config['batch_size'], int) or config['batch_size'] < 1:
             logger.error(f"batch_size ({config['batch_size']}) inválido (deve ser int >= 1).")
             is_valid = False
        if not isinstance(config['test_size_ratio'], (int, float)) or not 0 < config['test_size_ratio'] < 1:
             logger.error(f"test_size_ratio ({config['test_size_ratio']}) inválido (deve ser float em (0, 1)).")
             is_valid = False
        if not isinstance(config['validation_split_ratio'], (int, float)) or not 0 <= config['validation_split_ratio'] < 1:
             logger.error(f"validation_split_ratio ({config['validation_split_ratio']}) inválido (deve ser float/int em [0, 1)).")
             is_valid = False
        if (config['test_size_ratio'] + config['validation_split_ratio']) >= 1.0:
             logger.error(f"Soma de test_size_ratio ({config['test_size_ratio']}) + validation_split_ratio ({config['validation_split_ratio']}) deve ser < 1.0.")
             is_valid = False
        if not isinstance(config['cache_duration_hours'], (int, float)) or config['cache_duration_hours'] < 0:
             logger.error(f"cache_duration_hours ({config['cache_duration_hours']}) inválido (deve ser numérico >= 0).")
             is_valid = False
        if not isinstance(config['test_hyperparameters'], bool):
             logger.error(f"test_hyperparameters ({config['test_hyperparameters']}) inválido (deve ser true ou false).")
             is_valid = False
        if config['test_hyperparameters'] and config.get('hyperparameter_search') is None:
             logger.error("Se 'test_hyperparameters' é true, 'hyperparameter_search' deve ser definido no JSON.")
             is_valid = False


        # --- Consistency Checks (Always Performed, but interpreted differently based on stage) ---
        # Check consistency of statistical features count with rolling windows definition
        expected_stat_count = 1 + 1 + 1 + 4 + len(config['rolling_freq_windows']) * config['num_features_base']
        if config['num_features_statistical'] != expected_stat_count:
            # This might be intentional if explicitly set in config, so just warn.
            logger.warning(f"num_features_statistical ({config['num_features_statistical']}) no config "
                           f"não corresponde ao esperado pelo cálculo baseado em rolling_freq_windows ({expected_stat_count}). "
                           f"Verifique se a intenção era sobrescrever o valor.")
            # Decide if this should be a fatal error or just warning. Warning seems reasonable.
            # is_valid = False


        # --- Final Validation (only if check_total_features is True) ---
        if check_total_features:
            required_final_fields = ['num_features_time', 'num_features_total']
            for field in required_final_fields:
                if field not in config:
                     logger.error(f"Campo obrigatório ausente na configuração ({stage}): {field}")
                     is_valid = False
                elif not isinstance(config[field], int) or config[field] < 0:
                     logger.error(f"{field} ({config.get(field)}) inválido (deve ser int >= 0).")
                     is_valid = False

            if not is_valid: return False # Exit if final fields missing/invalid

            # Check if num_features_total matches the sum of its components
            calculated_total = (config.get('num_features_base', 0) +
                                config.get('num_features_time', 0) +
                                config.get('num_features_statistical', 0))
            if config['num_features_total'] != calculated_total:
                logger.error(f"num_features_total ({config['num_features_total']}) não bate com a soma das partes "
                             f"(Base={config.get('num_features_base', 0)} + Time={config.get('num_features_time', 0)} + Stat={config.get('num_features_statistical', 0)} = {calculated_total}).")
                is_valid = False

            # Check if num_features_time matches the expected calculation (base * 3 for complex features)
            expected_time_count = config.get('num_features_base', 0) * 3 # For complex time features
            if config['num_features_time'] != expected_time_count:
                 logger.error(f"num_features_time ({config['num_features_time']}) não bate com o esperado para features complexas "
                              f"(num_features_base * 3 = {expected_time_count}).")
                 is_valid = False


        # --- Conclusion ---
        if is_valid:
            logger.info(f"Configuração V3.1 ({stage}) validada com sucesso.")
        else:
            logger.error(f"Validação da configuração V3.1 ({stage}) falhou.")
        return is_valid

    except Exception as e:
        logger.error(f"Erro inesperado durante a validação da config ({stage}): {e}", exc_info=True)
        return False


# --- Fluxo Principal (Main) ---
def main():
    """ Função principal do programa V3.1 (com complex time features e correção de fluxo). """
    run_start_time = datetime.now()
    logger.info(f"--- Iniciando Script Mega-Sena V3.1 em {run_start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    try:
        # --- 1. Carregar e Validar Configuração Base ---
        logger.info("Etapa 1: Carregando Configuração Base...")
        global config
        config = load_config()
        if not validate_config(config, check_total_features=False): # Base validation only
            logger.critical("Configuração base inválida. Verifique configv3.json e logs. Abortando.")
            return
        logger.info("Configuração base carregada e validada.")

        # --- 2. Download/Preparação dos Dados ---
        logger.info("Etapa 2: Download/Preparação dos Dados Históricos...")
        df_full = download_and_prepare_data(
            url=config['data_url'],
            file_path=config['data_file'],
            cache_dir=config['cache_dir'],
            cache_duration_hours=config['cache_duration_hours']
        )
        if df_full is None or df_full.empty:
             logger.critical("Falha na Etapa 2 (Download/Preparação Dados). Abortando.")
             return
        logger.info(f"Dados históricos carregados: {len(df_full)} sorteios.")

        # --- 3. Pré-processamento dos Labels (Resultados) ---
        logger.info("Etapa 3: Pré-processamento dos Labels (Resultados Sorteados)...")
        bola_cols = [f'Bola{i+1}' for i in range(6)] # Assuming 6 balls
        encoded_labels, mlb, valid_indices = preprocess_data_labels(df_full, config['num_features_base']) # Pass full df
        if encoded_labels is None or mlb is None or valid_indices is None:
             logger.critical("Falha na Etapa 3 (Pré-processamento Labels). Abortando.")
             return
        # Filter the original DataFrame to keep only rows corresponding to valid labels
        df_full_valid = df_full.loc[valid_indices].reset_index(drop=True)
        logger.info(f"Labels processados e DataFrame filtrado: {len(df_full_valid)} sorteios válidos restantes.")
        # Check if enough data remains after filtering for labels
        min_data_needed = config.get('sequence_length', 15) * 3
        if len(df_full_valid) < min_data_needed:
            logger.error(f"Dados insuficientes ({len(df_full_valid)} sorteios) após limpeza de labels. Mínimo recomendado ~{min_data_needed}. Abortando.")
            return


        # --- 4. Cálculo das Features (Input para o Modelo) ---
        logger.info("Etapa 4: Cálculo das Features (Tempo e Estatísticas)...")
        # 4a. Features de Tempo Complexas
        logger.info("  4a: Calculando Features de Tempo Complexas...")
        time_features_raw = add_complex_time_features(df_full_valid[bola_cols], config['num_features_base'])
        if time_features_raw is None or len(time_features_raw) != len(encoded_labels):
             logger.critical("Falha no cálculo das Features de Tempo Complexas (Etapa 4a). Abortando.")
             return
        # Update config dynamically with the actual number of time features
        config['num_features_time'] = time_features_raw.shape[1]
        logger.info(f"   -> Features de Tempo Complexas calculadas. Shape: {time_features_raw.shape}. "
                    f"Config 'num_features_time' atualizada para: {config['num_features_time']}")

        # 4b. Features Estatísticas
        logger.info("  4b: Calculando Features Estatísticas...")
        statistical_features_raw = add_statistical_features(df_full_valid[bola_cols], config['num_features_base'], config['rolling_freq_windows'])
        if statistical_features_raw is None or len(statistical_features_raw) != len(encoded_labels):
             logger.critical("Falha no cálculo das Features Estatísticas (Etapa 4b). Abortando.")
             return
        # Verify or update config with the actual number of statistical features
        actual_stat_count = statistical_features_raw.shape[1]
        if config['num_features_statistical'] != actual_stat_count:
             logger.warning(f"Número real de features estatísticas ({actual_stat_count}) difere do config ({config['num_features_statistical']}). "
                           f"Ajustando config para {actual_stat_count}.")
             config['num_features_statistical'] = actual_stat_count
        logger.info(f"   -> Features Estatísticas calculadas. Shape: {statistical_features_raw.shape}. "
                     f"Config 'num_features_statistical' verificada/ajustada: {config['num_features_statistical']}")

        # --- 5. Cálculo e Validação Final das Features Totais ---
        logger.info("Etapa 5: Cálculo e Validação Final das Features Totais...")
        config['num_features_total'] = (config['num_features_base'] +
                                        config['num_features_time'] +
                                        config['num_features_statistical'])
        logger.info(f" -> Config 'num_features_total' calculado: {config['num_features_total']}")
        # Perform final validation including total features
        if not validate_config(config, check_total_features=True):
            logger.critical("Configuração final inválida após cálculo de features. Verifique logs. Abortando.")
            return
        logger.info("Contagem total de features validada.")


        # --- 6. Teste de Hiperparâmetros (Opcional) ---
        logger.info("Etapa 6: Verificação de Teste de Hiperparâmetros...")
        run_hyperparameter_tuning = config.get('test_hyperparameters', False) and hyperparameter_tuning_available
        if run_hyperparameter_tuning:
            logger.info("-" * 60)
            logger.info("MODO DE TESTE DE HIPERPARÂMETROS ATIVADO")
            logger.info("Os resultados deste modo podem sobrescrever parâmetros no 'config' para o treinamento final.")
            logger.info("-" * 60)

            # Check if tuner configuration exists
            if config.get('hyperparameter_search') is None:
                logger.error("Teste de hiperparâmetros ativado, mas 'hyperparameter_search' não definido no config. Abortando.")
                return

            # Instantiate the tuner - Pass necessary components
            # The tuner's internal logic must handle iterating through params,
            # potentially adjusting sequence_length, recalculating num_features_total internally for build_model,
            # calling split_data and build_model for each trial.
            try:
                logger.info("Instanciando HyperparameterTuner...")
                tuner = HyperparameterTuner(
                    base_config=config.copy(), # Pass a copy to avoid modifying original during search
                    encoded_labels=encoded_labels,
                    time_features_raw=time_features_raw,
                    statistical_features_raw=statistical_features_raw,
                    build_model_fn=build_model, # Pass the function itself
                    split_data_fn=split_data,   # Pass the function itself
                    validate_config_fn=validate_config, # Pass the function itself
                    output_dir=output_dir
                )

                # Execute the hyperparameter search
                logger.info("Iniciando busca de hiperparâmetros...")
                best_params = tuner.run_search()

                if best_params and isinstance(best_params, dict):
                    logger.info("*"*10 + " MELHORES HIPERPARÂMETROS ENCONTRADOS " + "*"*10)
                    logger.info(json.dumps(best_params, indent=2))
                    logger.info("Aplicando a melhor configuração encontrada ao 'config' global para o treinamento final.")
                    # Update the main config dict with the best parameters found
                    for key, value in best_params.items():
                        config[key] = value
                    logger.info(f"Configuração principal atualizada.")

                    # IMPORTANT: Recalculate num_features_total AFTER potentially changing parameters like sequence_length
                    logger.info("Recalculando num_features_total com base nos melhores hiperparâmetros...")
                    config['num_features_total'] = (config['num_features_base'] +
                                                    config['num_features_time'] +
                                                    config['num_features_statistical'])
                    logger.info(f" -> Config 'num_features_total' recalculado: {config['num_features_total']}")
                    # Re-validate the final configuration with best params
                    if not validate_config(config, check_total_features=True):
                         logger.critical("Configuração final com melhores hiperparâmetros é inválida. Abortando.")
                         return
                    logger.info("Configuração final com melhores hiperparâmetros validada.")

                else:
                    logger.warning("Não foi possível determinar a melhor configuração de hiperparâmetros ou a busca falhou. "
                                   "Continuando com a configuração original (ou a última carregada).")

            except Exception as e_tuner:
                logger.error(f"Erro durante a execução do teste de hiperparâmetros: {e_tuner}", exc_info=True)
                logger.warning("Continuando com a configuração original devido ao erro no tuner.")

            logger.info("-" * 60)
            logger.info("CONCLUÍDO TESTE DE HIPERPARÂMETROS. CONTINUANDO COM TREINAMENTO FINAL.")
            logger.info("-" * 60)
        else:
             logger.info("Teste de hiperparâmetros desativado ou módulo indisponível. Usando configuração carregada.")


        # --- 7. Divisão Final / Escalonamento / Sequenciamento ---
        # Uses the potentially updated config (e.g., sequence_length from tuning)
        logger.info("Etapa 7: Divisão Final / Escalonamento / Sequenciamento...")
        X_train, X_val, X_test, y_train, y_val, y_test, time_scaler, stat_scaler = split_data(
            encoded_labels,
            time_features_raw,         # Pass the calculated raw features
            statistical_features_raw,
            config['test_size_ratio'],
            config['validation_split_ratio'],
            config['sequence_length']  # Use potentially updated sequence length
        )
        # Check if split_data returned valid data (returns 8 items + 2 scalers)
        if any(data is None for data in [X_train, X_val, X_test, y_train, y_val, y_test, time_scaler, stat_scaler]):
             logger.critical("Falha na Etapa 7 (Divisão/Escalonamento/Sequenciamento). Abortando.")
             return
        if X_train.size == 0 or y_train.size == 0: # Double check train set isn't empty
             logger.critical("Conjunto de Treino Vazio após divisão/sequenciamento! Verifique dados e parâmetros. Abortando.")
             return
        logger.info("Dados divididos, escalados e sequenciados para Treino/Validação/Teste.")


        # --- 8. Construção do Modelo GRU Final ---
        # Uses the potentially updated config (gru_units, dropout, batch_norm, seq_len) and final num_features_total
        logger.info("Etapa 8: Construção do Modelo GRU Final...")
        model = build_model(
            config['sequence_length'],
            config['num_features_total'], # Use final total features count
            config['num_features_base'],
            config['gru_units'],
            config['dropout_rate'],
            config['use_batch_norm']
        )
        if model is None:
            logger.critical("Falha na Etapa 8 (Construção Modelo). Abortando.")
            return
        logger.info("Modelo GRU construído com sucesso.")


        # --- 9. Treinamento do Modelo Final ---
        logger.info("Etapa 9: Treinamento do Modelo Final...")
        # Create unique log dir for TensorBoard for this specific run
        tb_log_dir = os.path.join(config['tensorboard_log_dir'], datetime.now().strftime("%Y%m%d-%H%M%S"))
        history = train_model(
            model, X_train, y_train, X_val, y_val,
            config['epochs'], config['batch_size'], tb_log_dir # Pass TB log dir
        )
        if history is None:
             logger.critical("Falha na Etapa 9 (Treinamento Modelo). Abortando.")
             return
        logger.info("Modelo treinado com sucesso.")


        # --- 10. Avaliação do Modelo Final ---
        logger.info("Etapa 10: Avaliação do Modelo Final no Conjunto de Teste...")
        evaluation_results = None
        if X_test is not None and X_test.size > 0 and y_test is not None and y_test.size > 0:
             evaluation_results = evaluate_model(model, X_test, y_test, config['batch_size'])
             if evaluation_results is None: # Check if evaluation function itself failed
                 logger.warning("Falha na função de avaliação (Etapa 10). Continuando sem resultados de avaliação.")
                 evaluation_results = {'basic_metrics': {}, 'real_hits': None} # Ensure structure exists but is empty
        else:
             logger.warning("Conjunto de teste vazio ou ausente. Pulando avaliação final.")
             evaluation_results = {'basic_metrics': {}, 'real_hits': None} # Set default empty results
        logger.info("Avaliação no conjunto de teste concluída (ou pulada).")


        # --- 11. Previsão para o Próximo Sorteio ---
        logger.info("Etapa 11: Previsão para o Próximo Sorteio...")
        # Prepare the absolute last sequence from the RAW features
        try:
            final_sequence_length = config['sequence_length'] # Use the final sequence length
            last_sequence_labels = encoded_labels[-final_sequence_length:]
            last_sequence_time_raw = time_features_raw[-final_sequence_length:]
            last_sequence_stat_raw = statistical_features_raw[-final_sequence_length:]

            predicted_numbers, predicted_probabilities = predict_next_draw(
                 model, last_sequence_labels, last_sequence_time_raw, last_sequence_stat_raw,
                 time_scaler, stat_scaler, mlb # Pass fitted scalers
            )
        except IndexError:
             logger.error(f"Erro de índice ao tentar extrair a última sequência de tamanho {final_sequence_length}. "
                          f"Verifique se há dados suficientes (encoded_labels: {len(encoded_labels)}, time_raw: {len(time_features_raw)}, stat_raw: {len(statistical_features_raw)}).")
             predicted_numbers, predicted_probabilities = None, None # Set to None on error
        except Exception as e_last_seq:
             logger.error(f"Erro ao preparar ou prever com a última sequência: {e_last_seq}", exc_info=True)
             predicted_numbers, predicted_probabilities = None, None # Set to None on error

        if predicted_numbers is None or predicted_probabilities is None:
             logger.critical("Falha na Etapa 11 (Previsão Próximo Sorteio). Abortando.")
             return
        logger.info("Previsão para o próximo sorteio realizada.")


        # --- 12. Visualizações ---
        logger.info("Etapa 12: Geração de Visualizações...")
        try:
            plot_training_history(history)
            plot_prediction_analysis(predicted_numbers, predicted_probabilities, df_full_valid, config['sequence_length'])
            if X_test is not None and X_test.size > 0 and y_test is not None and y_test.size > 0:
                plot_hits_over_time(model, X_test, y_test, mlb)
            else:
                logger.info("Pulando gráfico de acertos ao longo do tempo (teste vazio/ausente).")
            logger.info("Visualizações geradas.")
        except Exception as e_viz:
             logger.error(f"Erro durante a geração de visualizações: {e_viz}", exc_info=True)
             # Continue process even if plotting fails


        # --- 13. Exportação dos Resultados ---
        logger.info("Etapa 13: Exportação dos Resultados para Excel...")
        export_results(df_full_valid, predicted_numbers, predicted_probabilities, evaluation_results, config)
        logger.info("Exportação concluída.")


        # --- Conclusão ---
        run_end_time = datetime.now()
        total_duration = run_end_time - run_start_time
        logger.info("-" * 60)
        logger.info(f"--- Processo V3.1 CONCLUÍDO com sucesso ---")
        logger.info(f"Tempo Total de Execução: {total_duration}")
        logger.info("-" * 60)
        logger.info(f"Log principal: {log_file}")
        logger.info(f"Resultados exportados: {config.get('export_file', 'N/A')}")
        logger.info(f"Gráficos salvos em: {output_dir}")
        logger.info(f"Logs TensorBoard: {config.get('tensorboard_log_dir', 'N/A')} (use 'tensorboard --logdir \"{config.get('tensorboard_log_dir')}\"' para visualizar)")
        if run_hyperparameter_tuning:
             logger.info(f"Resultados do teste de hiperparâmetros: {config.get('hyperparameter_search', {}).get('export_results_file', os.path.join(output_dir, 'hyperparameter_results.xlsx'))}")
        logger.info("-" * 60)
        logger.info("AVISO FINAL: Lembre-se que este é um modelo experimental baseado em dados passados.")
        logger.info("Resultados da loteria são inerentemente aleatórios. NÃO HÁ GARANTIA DE ACERTO.")
        logger.info("Jogue com responsabilidade e moderação.")
        logger.info("-" * 60)

    except FileNotFoundError as e:
         logger.critical(f"Erro Crítico: Arquivo não encontrado - {e}. Verifique caminhos e nomes de arquivos.", exc_info=True)
    except ValueError as e:
         logger.critical(f"Erro Crítico: Valor inválido encontrado - {e}. Verifique dados e parâmetros.", exc_info=True)
    except ImportError as e:
         logger.critical(f"Erro Crítico: Falha ao importar módulo - {e}. Verifique instalações (e.g., `pip install -r requirements.txt`).", exc_info=True)
    except tf.errors.OpError as e: # Catch TensorFlow runtime errors
         logger.critical(f"Erro Crítico do TensorFlow: {e}. Verifique compatibilidade de versões, dados e modelo.", exc_info=True)
    except Exception as e: # Catch-all for any other unexpected errors
        logger.critical(f"Erro GERAL inesperado e não tratado no fluxo principal: {e}", exc_info=True)
        # Indicate failure in the log's end
        logger.info("-" * 60)
        logger.info(f"--- Processo V3.1 INTERROMPIDO devido a erro crítico ---")
        logger.info("-" * 60)
    finally:
        logging.shutdown() # Ensure all handlers are closed properly


if __name__ == "__main__":
    main()