# -*- coding: utf-8 -*-
"""
Script Adaptado para "Previsão" da Lotofácil - Versão V3.1-LF
Baseado no script Mega-Sena V3.1, modificado para Lotofácil (15 dezenas de 1 a 25).
MODIFIED: Adapted parameters, data handling, features, model output, evaluation for Lotofácil.
MODIFIED: Simplified logging and comments. Removed redundant functions (none were fully redundant).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
# import joblib # Opcional: se quiser salvar/carregar scalers
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
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

# Importar módulo de otimização de hiperparâmetros (opcional)
try:
    from hyperparameter_tuning import HyperparameterTuner
    hyperparameter_tuning_available = True
except ImportError:
    hyperparameter_tuning_available = False
    warnings.warn("Módulo de otimização de hiperparâmetros (hyperparameter_tuning.py) não encontrado.")

# Criação da pasta output se não existir
output_dir = "output_lotofacil" # Nome da pasta alterado
os.makedirs(output_dir, exist_ok=True)

# Configuração de logging
log_file = os.path.join(output_dir, 'lotofacil_v3.log') # Nome do arquivo de log alterado
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')

# Handler para stdout com fallback de encoding
class EncodingStreamHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__(stream=sys.stdout)
    def emit(self, record):
        try:
            msg = self.format(record)
            try:
                self.stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                self.stream.write(msg.encode('ascii', 'replace').decode('ascii') + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        file_handler,
        EncodingStreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ignorar warnings (opcional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Configuração via Arquivo ---
def load_config(config_file='config_lotofacil.json'): # Nome do arquivo de config alterado
    """Carrega configurações de um arquivo JSON para Lotofácil."""
    NUM_DEZENAS = 15 # Lotofácil
    NUM_TOTAL_DEZENAS = 25 # Lotofácil
    NUM_ZONAS = 5 # 5 zonas de 5 números (1-5, 6-10, ..., 21-25)

    default_config = {
        "data_url": "https://loteriascaixa-api.herokuapp.com/api/lotofacil", # URL Lotofácil
        "data_file": None,
        "export_file": os.path.join(output_dir, "historico_e_previsoes_lotofacil_v3.xlsx"), # Nome do arquivo de exportação
        "sequence_length": 20, # Ajustado (pode precisar de tuning)
        "num_features_base": NUM_TOTAL_DEZENAS, # 25 para Lotofácil
        "num_features_statistical": 0, # Será calculado abaixo
        "rolling_freq_windows": [10, 50, 100], # Janelas para frequência
        "gru_units": 128, # Ajustado (pode precisar de tuning)
        "use_batch_norm": True,
        "dropout_rate": 0.3, # Ajustado (pode precisar de tuning)
        "epochs": 150, # Ajustado (pode precisar de tuning)
        "batch_size": 64,
        "test_size_ratio": 0.15,
        "validation_split_ratio": 0.15,
        "cache_duration_hours": 24,
        "cache_dir": os.path.join(output_dir, "cache"),
        "tensorboard_log_dir": os.path.join(output_dir, "logs/fit/"),
        "test_hyperparameters": False,
        "hyperparameter_search": None
    }

    # Calcular features estatísticas com base nos parâmetros da Lotofácil
    num_stat_features = 1 + 1 + 1 + NUM_ZONAS + len(default_config["rolling_freq_windows"]) * default_config["num_features_base"]
    default_config["num_features_statistical"] = num_stat_features

    try:
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config_loaded = json.load(f)
                logger.info(f"Configurações carregadas de {config_file}")

                # Atualizar config padrão com valores do arquivo
                for key, value in config_loaded.items():
                    if key in ["num_features_time", "num_features_total"]:
                         logger.warning(f"Ignorando '{key}' de {config_file}. Será calculado dinamicamente.")
                    elif key == "num_features_statistical" and key in config_loaded:
                         default_config[key] = value # Permitir override explícito
                         logger.info(f"Usando 'num_features_statistical' explícito do config: {value}")
                    elif key in default_config:
                         if key in ['export_file', 'cache_dir', 'tensorboard_log_dir']:
                             if not os.path.isabs(value): value = os.path.join(output_dir, value)
                         default_config[key] = value
                    elif key == "hyperparameter_search" and isinstance(value, dict):
                        default_config[key] = value
                    else:
                        logger.warning(f"Ignorando chave desconhecida/depreciada '{key}' de {config_file}")

                # Recalcular stat features se janelas ou base mudaram E NÃO foi explicitamente definido no arquivo
                if ("rolling_freq_windows" in config_loaded or "num_features_base" in config_loaded) and \
                   "num_features_statistical" not in config_loaded and \
                   default_config["num_features_base"] == NUM_TOTAL_DEZENAS: # Recalcula apenas se base for 25
                     num_stat_features = 1 + 1 + 1 + NUM_ZONAS + len(default_config["rolling_freq_windows"]) * default_config["num_features_base"]
                     if default_config["num_features_statistical"] != num_stat_features:
                         logger.info(f"Recalculando num_features_statistical para {num_stat_features} baseado no config.")
                         default_config["num_features_statistical"] = num_stat_features

                logger.info("Configurações padrão mescladas com valores do arquivo.")
        else:
            logger.warning(f"Arquivo {config_file} não encontrado. Usando configurações padrão para Lotofácil.")

    except Exception as e:
        logger.error(f"Erro ao carregar/mesclar configurações: {e}")

    # Garantir que diretórios existam
    Path(default_config['tensorboard_log_dir']).mkdir(parents=True, exist_ok=True)
    Path(default_config['cache_dir']).mkdir(parents=True, exist_ok=True)
    if default_config.get('export_file'):
        Path(os.path.dirname(default_config['export_file'])).mkdir(parents=True, exist_ok=True)

    logger.info("Configuração carregada. 'num_features_time' e 'num_features_total' serão calculados dinamicamente.")
    return default_config

# --- Sistema de Cache (sem alterações significativas, apenas logging) ---
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
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        # logger.info(f"Dados salvos no cache: {cache_file}") # Removido log menos essencial
    except Exception as e: logger.error(f"Erro ao salvar cache: {e}")

def load_from_cache(cache_file):
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # logger.info(f"Arquivo de cache não encontrado: {cache_file}") # Removido log menos essencial
        return None
    except Exception as e:
        logger.error(f"Erro ao carregar cache {cache_file}: {e}")
        return None

# --- Funções de Dados (Adaptadas para Lotofácil) ---

def download_and_prepare_data(url=None, file_path=None, cache_dir=None, cache_duration_hours=24):
    """Baixa/carrega dados da Lotofácil, garante colunas 'BolaX' (1-15) numéricas."""
    logger.info("Iniciando carregamento e preparação de dados da Lotofácil...")
    df = None
    data = None
    NUM_DEZENAS = 15
    NUM_TOTAL_DEZENAS = 25

    # --- Lógica de Cache e Download ---
    if url and cache_dir:
        cache_key = get_cache_key(url)
        safe_cache_dir = os.path.abspath(cache_dir)
        Path(safe_cache_dir).mkdir(parents=True, exist_ok=True)
        cache_file = os.path.join(safe_cache_dir, f"{cache_key}.json")

        if is_cache_valid(cache_file, cache_duration_hours):
            data = load_from_cache(cache_file)
            if data: logger.info("Dados carregados com sucesso do cache.")
            else: logger.warning("Cache inválido. Tentando baixar dados."); data = None

        if data is None:
             logger.info("Baixando dados da API da Lotofácil...")
             try:
                 headers = {'User-Agent': 'Mozilla/5.0'}
                 response = requests.get(url, headers=headers, verify=False, timeout=60) # Cuidado com verify=False
                 response.raise_for_status()
                 try: data = response.json()
                 except UnicodeDecodeError: response.encoding = response.apparent_encoding; data = response.json()
                 save_to_cache(data, cache_file)
                 logger.info("Dados baixados e salvos no cache.")
             except requests.exceptions.RequestException as e: logger.error(f"Erro de rede: {e}"); data = None
             except json.JSONDecodeError as json_err: logger.error(f"Erro JSON: {json_err}"); data = None

    # --- Fallback para arquivo local ---
    if data is None and file_path and os.path.exists(file_path):
        logger.info(f"Tentando carregar do arquivo local: {file_path}...")
        try:
            # (Mesma lógica de leitura de CSV anterior)
            df_loaded = None
            common_seps = [';', ',', '\t', '|']
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            for enc in encodings_to_try:
                for sep in common_seps:
                    try:
                        df_try = pd.read_csv(file_path, sep=sep, encoding=enc)
                        if df_try.shape[1] >= NUM_DEZENAS: df_loaded = df_try; break
                    except Exception: continue
                if df_loaded is not None: break
                if df_loaded is None:
                    try:
                        df_auto = pd.read_csv(file_path, sep=None, engine='python', encoding=enc)
                        if df_auto.shape[1] >= NUM_DEZENAS: df_loaded = df_auto; break
                    except Exception: continue
            if df_loaded is not None: df = df_loaded; logger.info(f"Dados carregados de {file_path}")
            else: logger.error(f"Não foi possível ler {file_path}."); return None
        except Exception as e_file: logger.error(f"Erro crítico ao carregar {file_path}: {e_file}"); return None

    # --- Processa dados da API/Cache ---
    elif data is not None:
        logger.info("Processando dados da API/Cache para DataFrame...")
        # Verifica se 'data' é uma lista (esperado pela API antiga) ou um dict (formato do exemplo)
        if isinstance(data, dict) and 'listaConcursos' in data:
            # Se for um dict com 'listaConcursos', usa essa lista
            logger.info("Formato de dados da API detectado: Dicionário com 'listaConcursos'.")
            draw_list = data.get('listaConcursos', [])
        elif isinstance(data, list):
            # Se for diretamente uma lista, usa a lista
            logger.info("Formato de dados da API detectado: Lista de sorteios.")
            draw_list = data
        else:
            logger.error(f"Formato de dados da API/Cache não reconhecido (tipo: {type(data)}). Esperado lista ou dict com 'listaConcursos'.")
            draw_list = [] # Define como lista vazia para evitar erro abaixo

        if draw_list:
            results, concursos, datas = [], [], []
            # **** CORREÇÃO DAS CHAVES AQUI ****
            api_dezenas_key = 'dezenas'       # <--- CORRIGIDO
            api_concurso_key = 'numeroConcurso'  # <--- CORRIGIDO (Verificar se é 'concurso' ou 'numeroConcurso' na lista)
            api_data_key = 'dataApuracao'         # <--- CORRIGIDO (Verificar se é 'data' ou 'dataApuracao' na lista)
            # Atualiza as chaves requeridas para validação
            required_keys = {api_dezenas_key, api_concurso_key, api_data_key}

            # Tenta deduzir as chaves corretas se as iniciais falharem (mais robusto)
            if draw_list and isinstance(draw_list[0], dict):
                first_draw_keys = draw_list[0].keys()
                if 'concurso' in first_draw_keys: api_concurso_key = 'concurso'
                if 'data' in first_draw_keys: api_data_key = 'data'
                # Atualiza as chaves requeridas novamente
                required_keys = {api_dezenas_key, api_concurso_key, api_data_key}
                logger.info(f"Chaves ajustadas/confirmadas: Dezenas='{api_dezenas_key}', Concurso='{api_concurso_key}', Data='{api_data_key}'")


            for i, sorteio in enumerate(draw_list):
                if not isinstance(sorteio, dict): continue

                # Validação das chaves requeridas
                if not required_keys.issubset(sorteio.keys()):
                    missing_keys = required_keys - set(sorteio.keys())
                    #logger.warning(f"Sorteio {sorteio.get(api_concurso_key, f'(índice {i})')} pulado. Chaves ausentes: {missing_keys}")
                    continue # Pula este sorteio

                try:
                    dezenas_raw = sorteio.get(api_dezenas_key, [])
                    if not isinstance(dezenas_raw, list): continue

                    dezenas_int = []
                    valid_dezenas = True
                    for d_str in dezenas_raw:
                        try: dezenas_int.append(int(d_str))
                        except (ValueError, TypeError): valid_dezenas = False; break
                    if not valid_dezenas: continue

                    dezenas = sorted(dezenas_int)

                    # Validação Lotofácil
                    if len(dezenas) == NUM_DEZENAS and all(1 <= d <= NUM_TOTAL_DEZENAS for d in dezenas):
                        results.append(dezenas)
                        concursos.append(sorteio.get(api_concurso_key))
                        try:
                            # Usa a chave de data correta e tenta o formato dd/mm/yyyy
                            datas.append(pd.to_datetime(sorteio.get(api_data_key), format='%d/%m/%Y', errors='coerce'))
                        except ValueError:
                            datas.append(pd.NaT)
                    else:
                         #logger.warning(f"Sorteio {sorteio.get(api_concurso_key)} inválido (número/valor dezenas): {dezenas}")
                         pass # Reduz log

                except Exception as e_proc:
                    logger.warning(f"Erro ao processar sorteio {sorteio.get(api_concurso_key, f'index {i}')}: {e_proc}")
                    continue

            if not results:
                # Este erro está ocorrendo, precisamos investigar o loop acima
                logger.error("Nenhum sorteio válido encontrado após processar dados da API/Cache.")
                df = None
            else:
                df = pd.DataFrame(results, columns=[f'Bola{i+1}' for i in range(NUM_DEZENAS)])
                if concursos and len(concursos) == len(df):
                    df['Concurso'] = pd.to_numeric(concursos, errors='coerce')
                if datas and len(datas) == len(df):
                    df['Data'] = datas
                    if df['Data'].isnull().any():
                         logger.warning(f"Coluna 'Data' com {df['Data'].isnull().sum()} valores não convertidos.")

                logger.info(f"Dados processados da API/Cache ({len(df)} sorteios válidos).")
        else:
            logger.error("Lista de sorteios ('draw_list') vazia após carregar/detectar dados.")
            df = None

    # --- Checagem Final e Processamento de Colunas (sem alterações aqui) ---
    if df is None:
        logger.critical("Nenhuma fonte de dados resultou em DataFrame válido.")
        return None

    # --- Identificação e Renomeação de Colunas (sem alterações aqui) ---
    bola_cols_found = []
    potential_patterns = [ [f'Bola{i+1}' for i in range(NUM_DEZENAS)], [f'bola{i+1}' for i in range(NUM_DEZENAS)],
                           [f'Dezena{i+1}' for i in range(NUM_DEZENAS)], [f'dezena{i+1}' for i in range(NUM_DEZENAS)],
                           [f'N{i+1}' for i in range(NUM_DEZENAS)], [f'n{i+1}' for i in range(NUM_DEZENAS)] ]
    df_cols_lower = {c.lower().strip(): c for c in df.columns}
    for pattern_list in potential_patterns:
        pattern_lower = [p.lower() for p in pattern_list]
        if all(pat_low in df_cols_lower for pat_low in pattern_lower):
             bola_cols_found = [df_cols_lower[pat_low] for pat_low in pattern_lower]; break
    if not bola_cols_found:
        logger.warning("Nenhum padrão de coluna. Tentando heurística numérica...")
        numeric_cols = df.select_dtypes(include=np.number).columns
        potential_bola_cols = []
        for c in numeric_cols:
             try:
                 numeric_col = pd.to_numeric(df[c], errors='coerce')
                 is_likely_bola = (numeric_col.dropna().between(1, NUM_TOTAL_DEZENAS).all() and
                                   numeric_col.dropna().apply(lambda x: x == int(x)).all())
                 if is_likely_bola: potential_bola_cols.append(c)
             except Exception: continue
        if len(potential_bola_cols) >= NUM_DEZENAS:
            preferred_cols = [c for c in potential_bola_cols if any(kw in c.lower() for kw in ['bola', 'dezena', 'n'])]
            if len(preferred_cols) >= NUM_DEZENAS: bola_cols_found = preferred_cols[:NUM_DEZENAS]
            else: bola_cols_found = potential_bola_cols[:NUM_DEZENAS]
            logger.warning(f"Colunas identificadas heuristicamente ({NUM_DEZENAS}): {bola_cols_found}. VERIFIQUE!")
        else: logger.error(f"Erro: Não identificadas {NUM_DEZENAS} colunas numéricas válidas (1-{NUM_TOTAL_DEZENAS})."); return None
    rename_map = {found_col: f'Bola{i+1}' for i, found_col in enumerate(bola_cols_found)}
    try:
        df.rename(columns=rename_map, inplace=True)
        bola_cols = [f'Bola{i+1}' for i in range(NUM_DEZENAS)]
        logger.info(f"Colunas renomeadas para: {bola_cols}")
    except Exception as e_rename: logger.error(f"Erro ao renomear: {e_rename}"); return None

    # --- Conversão Numérica e Limpeza (sem alterações) ---
    try:
        initial_rows = len(df)
        for col in bola_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=[col], inplace=True) # Remove NaNs criados ou existentes
        for col in bola_cols: df[col] = df[col].astype(int)
        rows_after_cleaning = len(df)
        if rows_after_cleaning < initial_rows: logger.info(f"{initial_rows - rows_after_cleaning} linhas removidas (bolas inválidas).")
        if rows_after_cleaning == 0: logger.error("Nenhuma linha válida restante."); return None
    except Exception as e_num: logger.error(f"Erro ao converter/limpar bolas: {e_num}"); return None

    # --- Seleção Final e Ordenação (sem alterações) ---
    cols_to_keep = bola_cols + [col for col in ['Concurso', 'Data'] if col in df.columns]
    final_df = df[cols_to_keep].copy()
    sort_col = None
    if 'Concurso' in final_df.columns and pd.api.types.is_numeric_dtype(final_df['Concurso']): sort_col = 'Concurso'
    elif 'Data' in final_df.columns and pd.api.types.is_datetime64_any_dtype(final_df['Data']): sort_col = 'Data'
    if sort_col:
        try: final_df = final_df.sort_values(by=sort_col).reset_index(drop=True); logger.info(f"Dados ordenados por '{sort_col}'.")
        except Exception as e_sort: logger.error(f"Erro ao ordenar por '{sort_col}': {e_sort}.")
    else: logger.warning("Não foi possível ordenar cronologicamente.")
    logger.info(f"Processamento final: {len(final_df)} sorteios válidos carregados.")
    min_data_needed = config.get('sequence_length', 20) * 3
    if len(final_df) < min_data_needed: logger.error(f"Dados insuficientes ({len(final_df)}). Mínimo ~{min_data_needed}"); return None

    return final_df

def preprocess_data_labels(df_balls_only, num_features_base):
    """Transforma números sorteados em formato MultiLabelBinarizer (labels y) para Lotofácil."""
    logger.info("Pré-processando labels (MultiLabelBinarizer)...")
    NUM_DEZENAS = 15 # Lotofácil
    try:
        if df_balls_only is None or df_balls_only.empty:
             logger.error("DataFrame vazio para pré-processar labels.")
             return None, None, None
        required_cols = [f'Bola{i+1}' for i in range(NUM_DEZENAS)]
        if not all(col in df_balls_only.columns for col in required_cols):
            logger.error(f"Colunas das bolas ausentes para labels: {required_cols}")
            return None, None, None

        balls_df = df_balls_only[required_cols].copy()

        # Validar dados antes de converter
        initial_rows = len(balls_df)
        rows_to_drop = []
        for index, row in balls_df.iterrows():
            valid_row = True
            for col in required_cols:
                val = row[col]
                # Checa se é inteiro e no range 1-25 (num_features_base)
                if not (isinstance(val, (int, np.integer)) and 1 <= val <= num_features_base):
                    valid_row = False; break
            if not valid_row: rows_to_drop.append(index)

        if rows_to_drop:
            original_indices = balls_df.index
            balls_df.drop(rows_to_drop, inplace=True)
            logger.warning(f"Removidas {len(rows_to_drop)} linhas com valores inválidos/fora do range [1, {num_features_base}] nas bolas.")
            if balls_df.empty:
                logger.error("Nenhuma linha válida restante após validação dos labels."); return None, None, None
            valid_original_indices = original_indices.difference(rows_to_drop)
        else:
            valid_original_indices = balls_df.index

        draws_list = balls_df.values.tolist()

        # Ajustar MLB para Lotofácil (classes 1 a 25)
        mlb = MultiLabelBinarizer(classes=list(range(1, num_features_base + 1)))
        encoded_data = mlb.fit_transform(draws_list)

        logger.info(f"Labels transformados: {encoded_data.shape[0]} amostras, {encoded_data.shape[1]} features base (números 1-{num_features_base}).")
        return encoded_data, mlb, valid_original_indices

    except Exception as e:
        logger.error(f"Erro no pré-processamento dos labels: {e}", exc_info=True)
        return None, None, None

def add_complex_time_features(df_balls_only, num_features_base):
    """Calcula features de tempo complexas (intervalo, média, desv padrão) para Lotofácil."""
    logger.info("Calculando features de tempo COMPLEXAS...")
    NUM_DEZENAS = 15 # Lotofácil
    try:
        if df_balls_only is None or df_balls_only.empty:
             logger.error("DataFrame vazio para calcular features de tempo."); return None

        bola_cols = [f'Bola{i+1}' for i in range(NUM_DEZENAS)]
        if not all(col in df_balls_only.columns for col in bola_cols):
            logger.error(f"Colunas das bolas ausentes para features de tempo: {bola_cols}"); return None

        draws = df_balls_only[bola_cols].values
        num_draws = len(draws)
        num_time_feat_per_num = 3 # interval, mean, std
        total_time_features = num_features_base * num_time_feat_per_num

        time_features_complex = np.zeros((num_draws, total_time_features), dtype=np.float32)
        seen_history = {num: [] for num in range(1, num_features_base + 1)}

        for i in range(num_draws):
            current_numbers_in_draw = set(draws[i])
            for num in range(1, num_features_base + 1):
                history = seen_history[num]
                num_sightings = len(history)

                # Intervalo Atual
                current_interval = float(i + 1) if num_sightings == 0 else float(i - history[-1])

                # Média e Desvio Padrão dos Intervalos Passados
                mean_interval, std_dev_interval = 0.0, 0.0
                if num_sightings >= 2:
                    past_intervals = np.diff(np.array(history))
                    if len(past_intervals) > 0: mean_interval = np.mean(past_intervals)
                    if len(past_intervals) >= 2: std_dev_interval = np.std(past_intervals)

                # Armazenar
                base_col_index = (num - 1) * num_time_feat_per_num
                time_features_complex[i, base_col_index]     = current_interval
                time_features_complex[i, base_col_index + 1] = mean_interval
                time_features_complex[i, base_col_index + 2] = std_dev_interval

            # Atualizar histórico
            for drawn_num in current_numbers_in_draw:
                if 1 <= drawn_num <= num_features_base:
                    seen_history[drawn_num].append(i)

        logger.info(f"Features de tempo complexas calculadas. Shape: {time_features_complex.shape}")
        if np.isnan(time_features_complex).any() or np.isinf(time_features_complex).any():
            logger.error("Features de tempo complexas contêm NaN ou Inf!"); return None

        return time_features_complex

    except Exception as e:
        logger.error(f"Erro ao calcular features de tempo complexas: {e}", exc_info=True)
        return None

def add_statistical_features(df_balls_only, num_features_base, rolling_windows):
    """Calcula features estatísticas para Lotofácil (Par/Ímpar, Soma, Range, Zonas, Freq. Rolante)."""
    logger.info(f"Calculando features estatísticas (Janelas: {rolling_windows})...")
    NUM_DEZENAS = 15 # Lotofácil
    NUM_ZONAS = 5 # Lotofácil (1-5, 6-10, 11-15, 16-20, 21-25)
    zone_defs = [(i * 5 + 1, (i + 1) * 5) for i in range(NUM_ZONAS)] # Define as 5 zonas

    try:
        if df_balls_only is None or df_balls_only.empty:
             logger.error("DataFrame vazio para calcular features estatísticas."); return None

        bola_cols = [f'Bola{i+1}' for i in range(NUM_DEZENAS)]
        if not all(col in df_balls_only.columns for col in bola_cols):
            logger.error(f"Colunas das bolas ausentes para features estatísticas: {bola_cols}"); return None

        draws = df_balls_only[bola_cols].values
        num_draws = len(draws)

        # --- Frequências Rolantes ---
        num_freq_features = len(rolling_windows) * num_features_base
        rolling_freq_features = np.zeros((num_draws, num_freq_features), dtype=np.float32)
        try:
            mlb_freq = MultiLabelBinarizer(classes=list(range(1, num_features_base + 1)))
            draws_list_for_mlb = [[int(n) for n in row] for row in draws]
            encoded_draws_freq = mlb_freq.fit_transform(draws_list_for_mlb)
            encoded_draws_df = pd.DataFrame(encoded_draws_freq, columns=mlb_freq.classes_, index=df_balls_only.index)
        except Exception as e_mlb:
            logger.error(f"Erro ao criar MLB para cálculo de frequência: {e_mlb}", exc_info=True); return None

        # logger.info("Calculando frequências rolantes...") # Log Reduzido
        freq_col_offset = 0
        for window in rolling_windows:
            if window <= 0: continue
            # logger.debug(f"  Calculando frequência para janela: {window}") # Log Reduzido
            try:
                rolling_sum = encoded_draws_df.rolling(window=window, min_periods=1).sum()
                rolling_sum_shifted = rolling_sum.shift(1).fillna(0)
                start_idx, end_idx = freq_col_offset, freq_col_offset + num_features_base
                rolling_freq_features[:, start_idx : end_idx] = rolling_sum_shifted.values
                freq_col_offset += num_features_base
            except Exception as e_roll:
                 logger.error(f"Erro na frequência rolante (janela {window}): {e_roll}", exc_info=True); return None
        # logger.info("Frequências rolantes calculadas.") # Log Reduzido

        # --- Estatísticas por Sorteio ---
        # logger.info("Calculando estatísticas por sorteio...") # Log Reduzido
        odd_counts, sums, ranges = [], [], []
        zone_counts_arr = np.zeros((num_draws, NUM_ZONAS), dtype=np.int32)

        try:
            for i in range(num_draws):
                current_numbers = draws[i]
                odd_counts.append(np.sum(current_numbers % 2 != 0))
                sums.append(np.sum(current_numbers))
                ranges.append(np.max(current_numbers) - np.min(current_numbers))
                for zone_idx, (z_min, z_max) in enumerate(zone_defs):
                    zone_counts_arr[i, zone_idx] = np.sum((current_numbers >= z_min) & (current_numbers <= z_max))
        except Exception as e_stat:
             logger.error(f"Erro ao calcular estatísticas básicas no sorteio {i}: {e_stat}", exc_info=True); return None

        # Combinar features
        odd_counts_arr = np.array(odd_counts).reshape(-1, 1)
        sums_arr = np.array(sums).reshape(-1, 1)
        ranges_arr = np.array(ranges).reshape(-1, 1)

        statistical_features_raw = np.concatenate([
            odd_counts_arr, sums_arr, ranges_arr, zone_counts_arr, rolling_freq_features
        ], axis=1).astype(np.float32)

        # Validação Final
        expected_cols = 1 + 1 + 1 + NUM_ZONAS + num_freq_features
        actual_cols = statistical_features_raw.shape[1]
        logger.info(f"Features estatísticas combinadas. Shape: {statistical_features_raw.shape}")

        if actual_cols != expected_cols:
             logger.error(f"Erro de shape nas features estatísticas! Esperado {expected_cols}, obtido {actual_cols}"); return None
        if np.isnan(statistical_features_raw).any() or np.isinf(statistical_features_raw).any():
            logger.error("Features estatísticas contêm NaN ou Inf!"); return None

        return statistical_features_raw

    except Exception as e:
        logger.error(f"Erro geral ao calcular features estatísticas: {e}", exc_info=True)
        return None

def split_data(encoded_labels, time_features_raw, statistical_features_raw,
               test_size_ratio, validation_split_ratio, sequence_length):
    """Divide dados, escala features (tempo e stat separadamente) e cria sequências."""
    logger.info("Dividindo dados, escalando features e criando sequências...")
    try:
        n_samples = len(encoded_labels)
        if not (n_samples == len(time_features_raw) == len(statistical_features_raw)):
             logger.error(f"Disparidade no número de amostras: Labels({n_samples}), Time({len(time_features_raw)}), Stat({len(statistical_features_raw)})"); return [None] * 8
        if n_samples <= sequence_length:
             logger.error(f"Amostras ({n_samples}) <= sequence_length ({sequence_length}). Impossível criar sequências."); return [None] * 8

        # --- 1. Dividir Índices Cronologicamente ---
        test_split_index = int(n_samples * (1 - test_size_ratio))
        val_split_index = int(test_split_index * (1 - validation_split_ratio))
        train_indices = np.arange(val_split_index)
        val_indices = np.arange(val_split_index, test_split_index)
        test_indices = np.arange(test_split_index, n_samples)
        logger.info(f"Índices - Treino: {len(train_indices)}, Val: {len(val_indices)}, Teste: {len(test_indices)}")

        # Validar tamanho mínimo para sequências em cada split
        min_len_for_seq = sequence_length + 1
        if len(train_indices) < min_len_for_seq: logger.error(f"Treino ({len(train_indices)}) pequeno para sequências ({min_len_for_seq})."); return [None] * 8
        if validation_split_ratio > 0 and len(val_indices) < min_len_for_seq: logger.warning(f"Validação ({len(val_indices)}) pequena para sequências ({min_len_for_seq}).")
        if len(test_indices) < min_len_for_seq: logger.warning(f"Teste ({len(test_indices)}) pequeno para sequências ({min_len_for_seq}).")

        # --- 2. Fatiar Dados ---
        train_labels = encoded_labels[train_indices]
        val_labels = encoded_labels[val_indices] if len(val_indices) > 0 else np.array([])
        test_labels = encoded_labels[test_indices] if len(test_indices) > 0 else np.array([])
        train_time_raw = time_features_raw[train_indices]
        val_time_raw = time_features_raw[val_indices] if len(val_indices) > 0 else np.array([])
        test_time_raw = time_features_raw[test_indices] if len(test_indices) > 0 else np.array([])
        train_stat_raw = statistical_features_raw[train_indices]
        val_stat_raw = statistical_features_raw[val_indices] if len(val_indices) > 0 else np.array([])
        test_stat_raw = statistical_features_raw[test_indices] if len(test_indices) > 0 else np.array([])
        # logger.debug(f"Dados brutos divididos: Treino({train_labels.shape[0]}), Val({val_labels.shape[0]}), Teste({test_labels.shape[0]})") # Log reduzido

        # --- 3. Escalar Features (Fit SÓ no Treino) ---
        time_scaler = StandardScaler()
        stat_scaler = StandardScaler()

        if train_time_raw.size > 0: train_time_scaled = time_scaler.fit_transform(train_time_raw)
        else: logger.error("train_time_raw vazio antes do scaling."); return [None] * 8
        if train_stat_raw.size > 0: train_stat_scaled = stat_scaler.fit_transform(train_stat_raw)
        else: logger.error("train_stat_raw vazio antes do scaling."); return [None] * 8

        val_time_scaled = time_scaler.transform(val_time_raw) if val_time_raw.size > 0 else np.array([])
        test_time_scaled = time_scaler.transform(test_time_raw) if test_time_raw.size > 0 else np.array([])
        val_stat_scaled = stat_scaler.transform(val_stat_raw) if val_stat_raw.size > 0 else np.array([])
        test_stat_scaled = stat_scaler.transform(test_stat_raw) if test_stat_raw.size > 0 else np.array([])
        logger.info("Scalers ajustados no treino e aplicados a Val/Teste.")

        # --- 4. Criar Sequências ---
        X_train, y_train = create_sequences(train_labels, train_time_scaled, train_stat_scaled, sequence_length)
        X_val, y_val = create_sequences(val_labels, val_time_scaled, val_stat_scaled, sequence_length)
        X_test, y_test = create_sequences(test_labels, test_time_scaled, test_stat_scaled, sequence_length)

        logger.info(f"Sequências criadas:")
        logger.info(f" Treino:    X={X_train.shape if X_train.size>0 else 'Vazio'}, y={y_train.shape if y_train.size>0 else 'Vazio'}")
        logger.info(f" Validação: X={X_val.shape if X_val.size>0 else 'Vazio'}, y={y_val.shape if y_val.size>0 else 'Vazio'}")
        logger.info(f" Teste:     X={X_test.shape if X_test.size>0 else 'Vazio'}, y={y_test.shape if y_test.size>0 else 'Vazio'}")

        if X_train.size == 0 or y_train.size == 0:
            logger.error("Conjunto de Treino VAZIO após sequenciamento."); return [None] * 8

        return X_train, X_val, X_test, y_train, y_val, y_test, time_scaler, stat_scaler

    except Exception as e:
        logger.error(f"Erro ao dividir/escalar/sequenciar dados: {e}", exc_info=True)
        return [None] * 8

def create_sequences(encoded_labels, time_features_scaled, statistical_features_scaled, sequence_length):
    """Cria sequências combinando labels e features escaladas (tempo, stat)."""
    if (encoded_labels is None or encoded_labels.size == 0 or
        time_features_scaled is None or time_features_scaled.size == 0 or
        statistical_features_scaled is None or statistical_features_scaled.size == 0):
        # logger.debug("Input vazio para create_sequences, retornando vazio.") # Log reduzido
        return np.array([]), np.array([])

    n_samples_total = len(encoded_labels)
    if not (n_samples_total == len(time_features_scaled) == len(statistical_features_scaled)):
        logger.error(f"Inconsistência de amostras em create_sequences: Labels({n_samples_total}), Time({len(time_features_scaled)}), Stat({len(statistical_features_scaled)})")
        return np.array([]), np.array([])
    if n_samples_total <= sequence_length:
        # logger.debug(f"Dados insuficientes ({n_samples_total}) para sequências ({sequence_length}).") # Log reduzido
        return np.array([]), np.array([])

    # logger.debug(f"Criando sequências de tamanho {sequence_length}...") # Log reduzido
    try:
        num_sequences = n_samples_total - sequence_length
        num_features_base = encoded_labels.shape[1]
        num_features_time = time_features_scaled.shape[1]
        num_features_stat = statistical_features_scaled.shape[1]
        num_features_total = num_features_base + num_features_time + num_features_stat

        X = np.zeros((num_sequences, sequence_length, num_features_total), dtype=np.float32)
        y = np.zeros((num_sequences, num_features_base), dtype=encoded_labels.dtype)

        for i in range(num_sequences):
            seq_labels = encoded_labels[i : i + sequence_length]
            seq_time = time_features_scaled[i : i + sequence_length]
            seq_stat = statistical_features_scaled[i : i + sequence_length]
            X[i] = np.concatenate((seq_labels, seq_time, seq_stat), axis=-1)
            y[i] = encoded_labels[i + sequence_length]

        # logger.debug(f"{len(X)} sequências combinadas criadas. Shape X: {X.shape}, Shape y: {y.shape}") # Log reduzido
        return X, y

    except Exception as e:
        logger.error(f"Erro ao criar sequências combinadas: {e}", exc_info=True)
        return np.array([]), np.array([])

# --- Modelo (Adaptado para Lotofácil) ---

def build_model(sequence_length, num_features_total, num_features_base, gru_units, dropout_rate, use_batch_norm):
    """ Constrói o modelo GRU para Lotofácil. """
    logger.info(f"Construindo modelo GRU: SeqLen={sequence_length}, TotalFeat={num_features_total}, BaseFeat(Saída)={num_features_base}, GRU={gru_units}, DO={dropout_rate}, BN={use_batch_norm}")
    try:
        if not all(isinstance(arg, int) and arg > 0 for arg in [sequence_length, num_features_total, num_features_base, gru_units]):
            logger.error("Argumentos numéricos inválidos para build_model."); return None
        if not 0 <= dropout_rate < 1: logger.error("dropout_rate inválido."); return None
        if not isinstance(use_batch_norm, bool): logger.error("use_batch_norm inválido."); return None

        model = Sequential(name=f"Modelo_GRU_Lotofacil_V3_F{num_features_base}")
        model.add(Input(shape=(sequence_length, num_features_total)))
        if use_batch_norm: model.add(BatchNormalization())
        model.add(GRU(gru_units, return_sequences=True, kernel_initializer='he_normal'))
        if use_batch_norm: model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        # Segunda camada GRU opcional (pode remover se overfit ou lento)
        model.add(GRU(gru_units // 2, return_sequences=False, kernel_initializer='he_normal'))
        if use_batch_norm: model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        # Camada densa opcional
        model.add(Dense(gru_units // 2, activation='relu'))
        if use_batch_norm: model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        # Camada de Saída: num_features_base (25 para Lotofácil), ativação sigmoid
        model.add(Dense(num_features_base, activation='sigmoid', name="Output_Layer"))

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # LR inicial
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy', tf.keras.metrics.AUC(name='auc')])
        model.build((None, sequence_length, num_features_total))
        logger.info("Resumo do Modelo (Keras):")
        model.summary(print_fn=logger.info)
        return model

    except Exception as e:
        logger.error(f"Erro ao construir o modelo GRU: {e}", exc_info=True)
        return None

# train_model permanece funcionalmente o mesmo, mas com logs simplificados
def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, log_dir):
    """ Treina o modelo GRU com callbacks e TensorBoard. """
    logger.info("Iniciando o treinamento do modelo GRU...")
    try:
        if model is None: logger.error("Modelo inválido para treinamento."); return None
        if X_train.size == 0 or y_train.size == 0: logger.error("Dados de treinamento vazios."); return None

        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=1e-6, verbose=1)
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        logger.info(f"Logs do TensorBoard: {log_dir}")

        class TrainingLogger(tf.keras.callbacks.Callback): # Log mais conciso por época
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                log_str = f"Época {epoch + 1}/{self.params['epochs']}"
                log_str += f" - Loss: {logs.get('loss'):.4f}"
                if 'val_loss' in logs: log_str += f" - Val Loss: {logs.get('val_loss'):.4f}"
                if 'auc' in logs: log_str += f" - AUC: {logs.get('auc'):.4f}"
                if 'val_auc' in logs: log_str += f" - Val AUC: {logs.get('val_auc'):.4f}"
                try: lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate); log_str += f" - LR: {lr:.6f}"
                except: pass
                logger.info(log_str)

        validation_data = None
        if X_val is not None and X_val.size > 0 and y_val is not None and y_val.size > 0:
             if X_val.shape[1:] == model.input_shape[1:] and y_val.shape[1:] == model.output_shape[1:]:
                 validation_data = (X_val, y_val)
                 logger.info(f"Usando conjunto de validação ({len(X_val)} amostras).")
             else: logger.error("Shape de validação incompatível. Validação desativada.")
        else:
             logger.warning("Conjunto de validação vazio/ausente. Callbacks usarão 'loss' do treino.")
             early_stopping.monitor = 'loss'
             reduce_lr.monitor = 'loss'

        logger.info(f"Iniciando treinamento por até {epochs} épocas (batch_size={batch_size})...")
        history = model.fit( X_train, y_train, epochs=epochs, batch_size=batch_size,
                             validation_data=validation_data,
                             callbacks=[early_stopping, reduce_lr, tensorboard_callback, TrainingLogger()],
                             verbose=0 ) # Usa logger customizado
        logger.info("Treinamento concluído.")
        return history

    except tf.errors.ResourceExhaustedError as e:
         logger.error(f"Erro OOM durante treinamento: {e}. Tente reduzir batch_size/sequence_length/gru_units.")
         return None
    except Exception as e:
        logger.error(f"Erro inesperado durante treinamento: {e}", exc_info=True)
        return None

# --- Avaliação e Previsão (Adaptadas para Lotofácil Top 15) ---

def evaluate_real_hits(model, X_test, y_test, batch_size=32):
    """Avalia acertos reais (Top 15 previstos vs sorteados) no teste da Lotofácil."""
    logger.info("Avaliando acertos reais (Top 15) no conjunto de teste...")
    NUM_PREDICTIONS = 15 # Lotofácil
    try:
        if model is None: logger.error("Modelo inválido para avaliação."); return None
        if X_test is None or X_test.size == 0 or y_test is None or y_test.size == 0:
             logger.warning("Dados de teste vazios. Pulando avaliação de acertos.")
             return { 'hits_per_draw': [], 'avg_hits': 0, 'max_hits': 0, 'hits_distribution': {}, 'detailed_hits': [] }
        if X_test.shape[0] != y_test.shape[0]: logger.error("Inconsistência X_test/y_test."); return None
        if model.output_shape[-1] != y_test.shape[-1]: logger.error("Inconsistência output modelo/y_test."); return None

        logger.info(f"Realizando previsões no teste ({len(X_test)} amostras)...")
        y_pred_probs = model.predict(X_test, batch_size=batch_size)

        hits_per_draw, detailed_hits = [], []
        for i in range(len(y_pred_probs)):
            topN_pred_indices = np.argsort(y_pred_probs[i])[-NUM_PREDICTIONS:] # Top 15
            predicted_numbers = sorted((topN_pred_indices + 1).tolist())
            actual_winning_indices = np.where(y_test[i] == 1)[0]
            actual_numbers = sorted((actual_winning_indices + 1).tolist())
            hits = set(predicted_numbers) & set(actual_numbers)
            num_hits = len(hits)
            hits_per_draw.append(num_hits)
            detailed_hits.append({'previstos': predicted_numbers, 'sorteados': actual_numbers, 'acertos': sorted(list(hits)), 'num_acertos': num_hits})

        num_test_draws = len(hits_per_draw)
        avg_hits, max_hits, hits_distribution = 0, 0, {}
        if num_test_draws > 0:
             avg_hits = np.mean(hits_per_draw)
             max_hits = np.max(hits_per_draw) if hits_per_draw else 0
             hits_distribution = {i: hits_per_draw.count(i) for i in range(max_hits + 1)}

        logger.info("-" * 60)
        logger.info(f"ANÁLISE DE ACERTOS REAIS (TOP {NUM_PREDICTIONS} PREVISTOS vs SORTEADOS)")
        logger.info(f"Sorteios no teste avaliados: {num_test_draws}")
        if num_test_draws > 0:
             logger.info(f"Média de acertos: {avg_hits:.3f}")
             logger.info(f"Máximo de acertos: {max_hits}")
             logger.info("Distribuição:")
             for hits_count in sorted(hits_distribution.keys()):
                 count = hits_distribution[hits_count]
                 if count > 0: logger.info(f"  - {hits_count} acerto(s): {count} sorteios ({(count / num_test_draws) * 100:.1f}%)")
        logger.info("-" * 60 + "\nAVISO: Acertos passados NÃO garantem acertos futuros.\n" + "-" * 60)

        return {'hits_per_draw': hits_per_draw, 'avg_hits': avg_hits, 'max_hits': max_hits, 'hits_distribution': hits_distribution, 'detailed_hits': detailed_hits}
    except Exception as e:
        logger.error(f"Erro ao avaliar acertos reais: {e}", exc_info=True)
        return None

def evaluate_model(model, X_test, y_test, batch_size=32):
    """Avalia o modelo no teste usando métricas Keras e acertos reais (Top 15)."""
    logger.info("Avaliando o modelo final no conjunto de teste...")
    evaluation_summary = {'basic_metrics': {}, 'real_hits': None}
    try:
        if model is None: logger.error("Modelo inválido."); return None
        if X_test is None or X_test.size == 0 or y_test is None or y_test.size == 0:
            logger.warning("Dados de teste vazios. Pulando avaliação completa.")
            return evaluation_summary

        # 1. Métricas Keras
        logger.info("Calculando métricas padrão Keras...")
        try:
            results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
            basic_metrics_dict = dict(zip(model.metrics_names, results))
            evaluation_summary['basic_metrics'] = basic_metrics_dict
            # logger.info("Métricas Keras calculadas:") # Log Reduzido
            # for name, value in basic_metrics_dict.items(): logger.info(f"  - {name}: {value:.4f}") # Log Reduzido
        except Exception as e_eval:
             logger.error(f"Erro durante model.evaluate: {e_eval}", exc_info=True)

        # 2. Acertos Reais (Top 15)
        real_hits_results = evaluate_real_hits(model, X_test, y_test, batch_size)
        if real_hits_results is None: logger.error("Falha na avaliação de acertos reais.")
        evaluation_summary['real_hits'] = real_hits_results

        # Log Resumo
        logger.info("-" * 60 + "\nResumo da Avaliação no Teste (Lotofácil)\n" + "-" * 60)
        logger.info("1. Métricas Keras:")
        if evaluation_summary['basic_metrics']:
            for name, value in evaluation_summary['basic_metrics'].items(): logger.info(f"  - {name}: {value:.4f}")
        else: logger.info("  N/A")
        logger.info("\n2. Estatísticas Acertos Reais (Top 15):")
        if real_hits_results:
             logger.info(f"  - Média acertos: {real_hits_results.get('avg_hits', 'N/A'):.3f}")
             logger.info(f"  - Máx acertos: {real_hits_results.get('max_hits', 'N/A')}")
             logger.info("  - Distribuição:")
             total_test = len(real_hits_results.get('hits_per_draw', []))
             if total_test > 0:
                 hits_dist = real_hits_results.get('hits_distribution', {})
                 if hits_dist:
                     for hc in sorted(hits_dist.keys()):
                          cnt = hits_dist[hc]; logger.info(f"    * {hc} acerto(s): {cnt} ({(cnt / total_test) * 100:.1f}%)")
                 else: logger.info("    Distribuição N/A.")
             else: logger.info("    N/A (sem sorteios no teste)")
        else: logger.info("  N/A")
        logger.info("-" * 60)
        return evaluation_summary

    except Exception as e:
        logger.error(f"Erro inesperado na avaliação final: {e}", exc_info=True)
        return evaluation_summary

def predict_next_draw(model, last_sequence_labels, last_sequence_time_raw, last_sequence_stat_raw,
                      time_scaler, stat_scaler, mlb, num_predictions=15): # Default 15 para Lotofácil
    """Prepara a última sequência, escala features e prevê o próximo sorteio (Top 15)."""
    logger.info(f"Preparando última sequência e prevendo PRÓXIMO sorteio (Top {num_predictions} Lotofácil)...")
    try:
        if model is None: logger.error("Modelo inválido."); return None, None
        if time_scaler is None or stat_scaler is None: logger.error("Scalers inválidos."); return None, None
        if mlb is None: logger.error("MLB inválido."); return None, None

        try: seq_len = model.input_shape[1]
        except: logger.error("Não foi possível obter seq_len do modelo."); return None, None

        # Validar shapes da última sequência
        valid_shape = True
        if last_sequence_labels is None or last_sequence_labels.shape[0] != seq_len: valid_shape = False
        if last_sequence_time_raw is None or last_sequence_time_raw.shape[0] != seq_len: valid_shape = False
        if last_sequence_stat_raw is None or last_sequence_stat_raw.shape[0] != seq_len: valid_shape = False
        if not valid_shape: logger.error(f"Última sequência com tamanho incorreto (esperado {seq_len})."); return None, None

        # 1. Escalar Features Raw
        last_sequence_time_scaled = time_scaler.transform(last_sequence_time_raw)
        last_sequence_stat_scaled = stat_scaler.transform(last_sequence_stat_raw)

        # 2. Combinar Features
        last_sequence_combined = np.concatenate(
            (last_sequence_labels, last_sequence_time_scaled, last_sequence_stat_scaled), axis=-1
        ).astype(np.float32)

        # Verificar dimensão total
        if last_sequence_combined.shape[-1] != model.input_shape[-1]:
            logger.error(f"Features combinadas ({last_sequence_combined.shape[-1]}) != Input esperado ({model.input_shape[-1]})."); return None, None

        # 3. Adicionar Batch e Prever
        last_sequence_batch = np.expand_dims(last_sequence_combined, axis=0)
        predicted_probabilities = model.predict(last_sequence_batch)[0]

        # 4. Processar Previsões
        if predicted_probabilities.shape[0] != model.output_shape[-1]:
            logger.error(f"Shape inesperado da previsão: {predicted_probabilities.shape}."); return None, None

        predicted_indices = np.argsort(predicted_probabilities)[-num_predictions:] # Top 15
        predicted_numbers = sorted((predicted_indices + 1).tolist())
        confidence_scores = predicted_probabilities[predicted_indices]
        avg_conf = np.mean(confidence_scores) if confidence_scores.size > 0 else 0.0
        max_conf = np.max(confidence_scores) if confidence_scores.size > 0 else 0.0
        min_conf = np.min(confidence_scores) if confidence_scores.size > 0 else 0.0

        # 5. Log Resultados
        logger.info("-" * 50)
        logger.info(f"PREVISÃO LOTOFÁCIL - PRÓXIMO SORTEIO")
        logger.info(f"Números Mais Prováveis ({num_predictions}): {predicted_numbers}")
        logger.info(f"Confiança (Média/Máx/Mín): {avg_conf:.4f} / {max_conf:.4f} / {min_conf:.4f}")
        logger.info("Probabilidades individuais (Top 15 ordenado):")
        sorted_indices = predicted_indices[np.argsort(confidence_scores)[::-1]]
        for idx in sorted_indices: logger.info(f"  - Número {idx + 1:02d}: {predicted_probabilities[idx]:.4f}")
        logger.info("-" * 50 + "\nAVISO CRÍTICO: Previsão estatística experimental. NÃO HÁ GARANTIA DE ACERTO.\n" + "-" * 50)

        return predicted_numbers, predicted_probabilities

    except Exception as e:
        logger.error(f"Erro na previsão do próximo sorteio: {e}", exc_info=True)
        return None, None

# --- Visualização e Exportação (Adaptadas) ---

def plot_training_history(history, filename=None):
    """Plota histórico de treinamento (Loss, Accuracy, AUC, LR)."""
    if filename is None: filename = os.path.join(output_dir, 'training_history_lotofacil.png')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logger.info(f"Gerando gráficos do histórico de treinamento...")
    try:
        if history is None or not history.history: logger.error("Histórico vazio."); return
        history_dict = history.history
        metrics = [k for k in history_dict if not k.startswith('val_') and k != 'lr']
        has_val = any(k.startswith('val_') for k in history_dict)
        has_lr = 'lr' in history_dict
        num_plots = len(metrics) + (1 if has_lr else 0)
        if num_plots == 0: logger.error("Nenhuma métrica no histórico."); return

        num_cols = 2; num_rows = (num_plots + num_cols - 1) // num_cols
        plt.figure(figsize=(max(12, num_cols * 6), num_rows * 5))
        plot_index = 1
        for metric in metrics:
            plt.subplot(num_rows, num_cols, plot_index)
            plt.plot(history_dict[metric], label=f'Treino {metric.capitalize()}')
            if has_val and f'val_{metric}' in history_dict: plt.plot(history_dict[f'val_{metric}'], label=f'Validação {metric.capitalize()}')
            plt.title(f'{metric.capitalize()} por Época'); plt.xlabel('Época'); plt.ylabel(metric.capitalize()); plt.legend(); plt.grid(True, alpha=0.6)
            plot_index += 1
        if has_lr:
            plt.subplot(num_rows, num_cols, plot_index)
            plt.plot(history_dict['lr'], label='Taxa Aprendizado (LR)'); plt.title('Taxa de Aprendizado'); plt.xlabel('Época'); plt.ylabel('LR'); plt.legend(); plt.grid(True, alpha=0.6); plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.tight_layout(); plt.savefig(filename); plt.close()
        logger.info(f"Gráficos de treinamento salvos em '{filename}'")
    except Exception as e: logger.error(f"Erro ao gerar gráficos de treinamento: {e}", exc_info=True)

def plot_prediction_analysis(predicted_numbers, predicted_probabilities, df_full_valid, sequence_length, filename=None):
    """ Gera análise visual comparando previsões (Top 15), probabilidades e frequência recente (Lotofácil). """
    if filename is None: filename = os.path.join(output_dir, 'prediction_analysis_lotofacil.png')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logger.info(f"Gerando análise visual das previsões...")
    NUM_DEZENAS = 15 # Lotofácil
    try:
        if predicted_numbers is None or predicted_probabilities is None: logger.error("Dados de previsão inválidos."); return
        if not isinstance(predicted_probabilities, np.ndarray) or predicted_probabilities.ndim != 1: logger.error("Probabilidades devem ser array NumPy 1D."); return
        num_features_base = len(predicted_probabilities) # Deveria ser 25
        if num_features_base == 0: logger.error("Vetor de probabilidades vazio."); return
        if df_full_valid is None or df_full_valid.empty: logger.error("DataFrame histórico vazio."); return
        if sequence_length <= 0: logger.error("sequence_length inválido."); return
        if len(df_full_valid) < sequence_length: sequence_length = len(df_full_valid)

        all_numbers = np.arange(1, num_features_base + 1)
        predicted_numbers_arr = np.array(predicted_numbers)
        probs_for_predicted = predicted_probabilities[predicted_numbers_arr - 1]

        # Frequência Recente
        bola_cols = [f'Bola{i+1}' for i in range(NUM_DEZENAS)]
        if not all(col in df_full_valid.columns for col in bola_cols): logger.error("Colunas das bolas não encontradas."); return
        last_n_draws_df = df_full_valid.iloc[-sequence_length:]
        try: last_numbers_flat = pd.concat([last_n_draws_df[col] for col in bola_cols]).dropna().astype(int).values
        except: logger.error("Erro ao achatar números para frequência."); return
        number_freq = np.zeros(num_features_base)
        if last_numbers_flat.size > 0:
            unique_nums, counts = np.unique(last_numbers_flat, return_counts=True)
            valid_mask = (unique_nums >= 1) & (unique_nums <= num_features_base)
            if valid_mask.any(): number_freq[unique_nums[valid_mask] - 1] = counts[valid_mask]
        freq_for_predicted = number_freq[predicted_numbers_arr - 1]

        # Plots
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10)) # Ajuste tamanho se necessário
        fig.suptitle('Análise da Previsão Lotofácil vs Histórico Recente', fontsize=16, y=1.02)

        # Plot 1: Todas Probabilidades
        ax1 = axes[0, 0]
        ax1.bar(all_numbers, predicted_probabilities, width=0.8, color='skyblue'); ax1.set_title(f'Probabilidades Previstas (1-{num_features_base})'); ax1.set_xlabel('Número'); ax1.set_ylabel('Probabilidade'); ax1.grid(True, axis='y', alpha=0.7); ax1.set_xticks(np.arange(0, num_features_base + 1, 5)); ax1.set_xlim(0.5, num_features_base + 0.5)
        # Plot 2: Probabilidades Top 15
        ax2 = axes[0, 1]
        bars = ax2.bar(predicted_numbers_arr, probs_for_predicted, width=0.6, color='coral'); ax2.set_title(f'Probabilidades dos {len(predicted_numbers)} Números Previstos'); ax2.set_xlabel('Número Previsto'); ax2.set_ylabel('Probabilidade'); ax2.grid(True, axis='y', alpha=0.7); ax2.bar_label(bars, fmt='%.4f', padding=3, fontsize=8); ax2.set_xticks(predicted_numbers_arr); ax2.set_xticklabels(predicted_numbers_arr, rotation=45, ha="right");
        if probs_for_predicted.size > 0: ax2.set_ylim(0, max(probs_for_predicted) * 1.20)
        # Plot 3: Frequência Recente
        ax3 = axes[1, 0]
        ax3.bar(all_numbers, number_freq, width=0.8, color='lightgreen'); ax3.set_title(f'Frequência nos Últimos {sequence_length} Sorteios'); ax3.set_xlabel('Número'); ax3.set_ylabel('Frequência'); ax3.grid(True, axis='y', alpha=0.7); ax3.set_xticks(np.arange(0, num_features_base + 1, 5)); ax3.set_xlim(0.5, num_features_base + 0.5);
        if number_freq.max() > 0: ax3.set_yticks(np.arange(0, int(number_freq.max()) + 2, 1))
        # Plot 4: Frequência vs Probabilidade
        ax4 = axes[1, 1]
        ax4.scatter(number_freq, predicted_probabilities, alpha=0.5, label='Outros', s=30); ax4.scatter(freq_for_predicted, probs_for_predicted, color='red', s=80, label='Previstos (Top 15)', edgecolors='black', zorder=5); ax4.set_title('Frequência Recente vs Probabilidade Prevista'); ax4.set_xlabel(f'Frequência (Últimos {sequence_length})'); ax4.set_ylabel('Probabilidade Prevista'); ax4.grid(True, alpha=0.7); ax4.legend()
        for i, num in enumerate(predicted_numbers_arr): ax4.text(freq_for_predicted[i]*1.01, probs_for_predicted[i], str(num), fontsize=9, va='center')

        plt.tight_layout(rect=[0, 0, 1, 0.98]); plt.savefig(filename); plt.close(fig)
        logger.info(f"Análise visual salva em '{filename}'")
    except Exception as e: logger.error(f"Erro ao gerar análise visual: {e}", exc_info=True)

def plot_hits_over_time(model, X_test, y_test, mlb, filename=None):
    """ Plota acertos (Top 15) ao longo do tempo no teste da Lotofácil. """
    if filename is None: filename = os.path.join(output_dir, 'hits_over_time_lotofacil.png')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logger.info(f"Gerando gráfico de acertos (Top 15) ao longo do tempo...")
    NUM_PREDICTIONS = 15 # Lotofácil
    try:
        if model is None: logger.error("Modelo inválido."); return None
        if X_test is None or X_test.size == 0 or y_test is None or y_test.size == 0: logger.warning("Teste vazio, pulando gráfico."); return None
        if X_test.shape[0] != y_test.shape[0]: logger.error("Inconsistência X_test/y_test."); return None
        if model.output_shape[-1] != y_test.shape[-1]: logger.error("Inconsistência output/y_test."); return None

        # logger.info("Calculando acertos no teste para plotagem...") # Log Reduzido
        y_pred_probs_test = model.predict(X_test)
        hits_per_draw = []
        for i in range(len(y_pred_probs_test)):
            topN_pred_indices = np.argsort(y_pred_probs_test[i])[-NUM_PREDICTIONS:]
            actual_winning_indices = np.where(y_test[i] == 1)[0]
            num_hits = len(set(topN_pred_indices) & set(actual_winning_indices))
            hits_per_draw.append(num_hits)

        if not hits_per_draw: logger.warning("Nenhum acerto calculado."); return None

        num_test_draws = len(hits_per_draw)
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(15, 6))
        plt.plot(range(num_test_draws), hits_per_draw, marker='o', linestyle='-', markersize=4, label=f'Nº Acertos (Top {NUM_PREDICTIONS}) / Sorteio Teste', alpha=0.7)
        if num_test_draws >= 10:
             rolling_avg = pd.Series(hits_per_draw).rolling(window=10, min_periods=1).mean()
             plt.plot(range(num_test_draws), rolling_avg, linestyle='--', color='red', linewidth=2, label=f'Média Móvel (10 Sorteios)')
        plt.xlabel("Índice Sorteio Teste"); plt.ylabel("Número de Acertos"); plt.title(f"Acertos Lotofácil (Top {NUM_PREDICTIONS}) no Conjunto de Teste"); plt.yticks(np.arange(0, NUM_PREDICTIONS + 1, 1)); plt.ylim(bottom=-0.2, top=NUM_PREDICTIONS + 0.2); plt.grid(True, axis='y', alpha=0.7); plt.legend(); plt.tight_layout()
        plt.savefig(filename); plt.close()
        logger.info(f"Gráfico de acertos salvo em '{filename}'")
        return hits_per_draw
    except Exception as e: logger.error(f"Erro ao gerar gráfico de acertos: {e}", exc_info=True); return None

def export_results(df_full_valid, predicted_numbers, predicted_probabilities, evaluation_results, config):
    """ Exporta histórico, previsões (Top 15), avaliação e config para Excel (Lotofácil). """
    export_file = config.get('export_file')
    if not export_file: logger.error("'export_file' não definido."); return
    logger.info(f"Exportando resultados para Excel: {export_file}...")
    NUM_PREDICTIONS = 15 # Lotofácil
    try:
        if df_full_valid is None or df_full_valid.empty: logger.error("Histórico vazio."); return
        if predicted_numbers is None or predicted_probabilities is None: logger.error("Previsão inválida."); return
        if not isinstance(predicted_probabilities, np.ndarray) or predicted_probabilities.ndim != 1: logger.error("Probabilidades devem ser array NumPy 1D."); return
        num_features_base = len(predicted_probabilities)
        if num_features_base == 0: logger.error("Probabilidades vazias."); return
        if evaluation_results is None:
            logger.warning("Resultados da avaliação não fornecidos.")
            evaluation_results = {'basic_metrics': {}, 'real_hits': None}

        # 1. Previsão e Probabilidades
        predictions_df = pd.DataFrame({'Número': range(1, num_features_base + 1), 'Probabilidade_Prevista': predicted_probabilities})
        predictions_df[f'Previsto_Top_{NUM_PREDICTIONS}'] = predictions_df['Número'].isin(predicted_numbers)
        predictions_df = predictions_df.sort_values('Probabilidade_Prevista', ascending=False).reset_index(drop=True)

        # 2. Métricas Básicas
        basic_metrics_dict = evaluation_results.get('basic_metrics', {})
        if not basic_metrics_dict: metrics_df = pd.DataFrame([{'Métrica': 'N/A', 'Valor': 'N/A'}])
        else: metrics_df = pd.DataFrame({'Métrica': list(basic_metrics_dict.keys()), 'Valor': [f"{v:.5f}" if isinstance(v, (float, int)) else str(v) for v in basic_metrics_dict.values()]})

        # 3. Acertos Reais
        real_hits_results = evaluation_results.get('real_hits')
        hits_summary_df = pd.DataFrame([{'Estatística': 'N/A', 'Valor': 'N/A'}])
        hits_dist_df = pd.DataFrame([{'Número Acertos': 'N/A', 'Qtd Sorteios': 'N/A', 'Porcentagem (%)': 'N/A'}])
        detailed_hits_df = pd.DataFrame([{'Info': 'Detalhes de acertos não disponíveis'}])
        if real_hits_results and isinstance(real_hits_results, dict):
            total_draws = len(real_hits_results.get('hits_per_draw', []))
            if total_draws > 0:
                hits_summary_df = pd.DataFrame({'Estatística': [f'Média Acertos (Top {NUM_PREDICTIONS})', f'Máx Acertos (Top {NUM_PREDICTIONS})', 'Total Sorteios Teste'], 'Valor': [f"{real_hits_results.get('avg_hits', 'N/A'):.3f}", f"{real_hits_results.get('max_hits', 'N/A')}", total_draws]})
                hits_dist = real_hits_results.get('hits_distribution')
                if hits_dist and isinstance(hits_dist, dict):
                    hits_dist_df = pd.DataFrame({'Número Acertos': list(hits_dist.keys()), 'Qtd Sorteios': list(hits_dist.values())})
                    hits_dist_df['Porcentagem (%)'] = hits_dist_df['Qtd Sorteios'].apply(lambda c: f"{(c / total_draws) * 100:.1f}%" if total_draws > 0 else 'N/A')
                    hits_dist_df = hits_dist_df.sort_values('Número Acertos').reset_index(drop=True)
                detailed_hits = real_hits_results.get('detailed_hits')
                if detailed_hits and isinstance(detailed_hits, list) and len(detailed_hits) > 0:
                     try:
                        detailed_hits_df = pd.DataFrame(detailed_hits)
                        for col in ['previstos', 'sorteados', 'acertos']:
                            if col in detailed_hits_df.columns: detailed_hits_df[col] = detailed_hits_df[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
                     except Exception as e_det: logger.error(f"Erro ao formatar detalhes de acertos: {e_det}"); detailed_hits_df = pd.DataFrame([{'Info': 'Erro ao formatar detalhes'}])

        # 4. Histórico Utilizado
        # df_full_valid já está pronto

        # 5. Configuração Usada
        config_items = []
        for k, v in config.items():
             if isinstance(v, dict): config_items.append([k, json.dumps(v, indent=2)])
             elif isinstance(v, list): config_items.append([k, ', '.join(map(str,v))])
             else: config_items.append([k, v])
        config_df = pd.DataFrame(config_items, columns=['Parametro', 'Valor'])

        # Escrever no Excel
        logger.info("Escrevendo abas no Excel...")
        os.makedirs(os.path.dirname(export_file), exist_ok=True)
        with pd.ExcelWriter(export_file, engine='openpyxl') as writer:
            predictions_df.to_excel(writer, sheet_name='Previsao_Probabilidades', index=False)
            metrics_df.to_excel(writer, sheet_name='Metricas_Avaliacao', index=False)
            hits_summary_df.to_excel(writer, sheet_name='Sumario_Acertos_Reais', index=False)
            hits_dist_df.to_excel(writer, sheet_name='Distribuicao_Acertos_Reais', index=False)
            if detailed_hits_df is not None and not detailed_hits_df.empty: detailed_hits_df.to_excel(writer, sheet_name='Detalhes_Acertos_Teste', index=False)
            df_full_valid.to_excel(writer, sheet_name='Historico_Utilizado', index=False)
            config_df.to_excel(writer, sheet_name='Configuracao_Usada', index=False)
        logger.info(f"Resultados exportados com sucesso para '{export_file}'")
    except PermissionError: logger.error(f"Erro de Permissão ao escrever Excel '{export_file}'.")
    except Exception as e: logger.error(f"Erro ao exportar resultados para Excel: {e}", exc_info=True)

# validate_config adaptado para V3.1-LF
def validate_config(config, check_total_features=False):
    """Valida as configurações V3.1 para Lotofácil."""
    stage = "base" if not check_total_features else "final"
    logger.info(f"Validando configuração V3.1-LF ({stage})...")
    is_valid = True
    NUM_TOTAL_DEZENAS = 25
    NUM_ZONAS = 5
    try:
        required_fields = ['data_url', 'data_file', 'export_file', 'sequence_length', 'num_features_base',
                           'num_features_statistical', 'rolling_freq_windows', 'gru_units', 'use_batch_norm',
                           'dropout_rate', 'epochs', 'batch_size', 'test_size_ratio', 'validation_split_ratio',
                           'cache_duration_hours', 'cache_dir', 'tensorboard_log_dir', 'test_hyperparameters']
        for field in required_fields:
            if field not in config: logger.error(f"Campo obrigatório ausente ({stage}): {field}"); is_valid = False
        if not is_valid: return False

        # Checagens de tipo/valor
        if config.get('data_url') is None and config.get('data_file') is None: logger.error("'data_url' ou 'data_file' necessário."); is_valid = False
        if not isinstance(config['sequence_length'], int) or config['sequence_length'] < 1: logger.error("sequence_length inválido."); is_valid = False
        if config['num_features_base'] != NUM_TOTAL_DEZENAS: logger.error(f"num_features_base ({config['num_features_base']}) deve ser {NUM_TOTAL_DEZENAS} para Lotofácil."); is_valid = False
        if not isinstance(config['num_features_statistical'], int) or config['num_features_statistical'] < 0: logger.error("num_features_statistical inválido."); is_valid = False
        if not isinstance(config['rolling_freq_windows'], list) or not all(isinstance(x, int) and x > 0 for x in config['rolling_freq_windows']): logger.error("rolling_freq_windows inválido."); is_valid = False
        if not isinstance(config['gru_units'], int) or config['gru_units'] < 1: logger.error("gru_units inválido."); is_valid = False
        if not isinstance(config['use_batch_norm'], bool): logger.error("use_batch_norm inválido."); is_valid = False
        if not 0 <= config['dropout_rate'] < 1: logger.error("dropout_rate inválido."); is_valid = False
        if not isinstance(config['epochs'], int) or config['epochs'] < 1: logger.error("epochs inválido."); is_valid = False
        if not isinstance(config['batch_size'], int) or config['batch_size'] < 1: logger.error("batch_size inválido."); is_valid = False
        if not 0 < config['test_size_ratio'] < 1: logger.error("test_size_ratio inválido."); is_valid = False
        if not 0 <= config['validation_split_ratio'] < 1: logger.error("validation_split_ratio inválido."); is_valid = False
        if (config['test_size_ratio'] + config['validation_split_ratio']) >= 1.0: logger.error("Soma test+val ratio >= 1.0."); is_valid = False
        if not isinstance(config['test_hyperparameters'], bool): logger.error("test_hyperparameters inválido."); is_valid = False
        if config['test_hyperparameters'] and config.get('hyperparameter_search') is None: logger.error("Se 'test_hyperparameters'=true, 'hyperparameter_search' deve ser definido."); is_valid = False

        # Checagem de consistência features estatísticas
        expected_stat_count = 1 + 1 + 1 + NUM_ZONAS + len(config['rolling_freq_windows']) * config['num_features_base']
        if config['num_features_statistical'] != expected_stat_count:
            logger.warning(f"num_features_statistical ({config['num_features_statistical']}) difere do esperado ({expected_stat_count}). Verifique override.")

        # Validação Final (se check_total_features=True)
        if check_total_features:
            required_final = ['num_features_time', 'num_features_total']
            for field in required_final:
                if field not in config: logger.error(f"Campo final ausente ({stage}): {field}"); is_valid = False
                elif not isinstance(config[field], int) or config[field] < 0: logger.error(f"{field} inválido."); is_valid = False
            if not is_valid: return False

            calculated_total = (config.get('num_features_base', 0) + config.get('num_features_time', 0) + config.get('num_features_statistical', 0))
            if config['num_features_total'] != calculated_total:
                logger.error(f"num_features_total ({config['num_features_total']}) != soma das partes ({calculated_total})."); is_valid = False
            expected_time_count = config.get('num_features_base', 0) * 3 # Complex time features
            if config['num_features_time'] != expected_time_count:
                 logger.error(f"num_features_time ({config['num_features_time']}) != esperado (base * 3 = {expected_time_count})."); is_valid = False

        if is_valid: logger.info(f"Configuração V3.1-LF ({stage}) validada.")
        else: logger.error(f"Validação da configuração V3.1-LF ({stage}) falhou.")
        return is_valid
    except Exception as e: logger.error(f"Erro na validação da config ({stage}): {e}", exc_info=True); return False

# --- Fluxo Principal (Main Adaptado) ---
def main():
    """ Função principal do programa V3.1 para Lotofácil. """
    run_start_time = datetime.now()
    logger.info(f"--- Iniciando Script Lotofácil V3.1-LF em {run_start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    NUM_DEZENAS = 15
    try:
        # 1. Carregar e Validar Config Base
        logger.info("Etapa 1: Carregando Configuração Base Lotofácil...")
        global config
        config = load_config()
        if not validate_config(config, check_total_features=False):
            logger.critical("Configuração base inválida. Verifique config_lotofacil.json. Abortando."); return
        logger.info("Configuração base carregada e validada.")

        # 2. Download/Preparação Dados
        logger.info("Etapa 2: Download/Preparação Dados Lotofácil...")
        df_full = download_and_prepare_data( url=config['data_url'], file_path=config['data_file'],
                                             cache_dir=config['cache_dir'], cache_duration_hours=config['cache_duration_hours'] )
        if df_full is None or df_full.empty: logger.critical("Falha na Etapa 2 (Dados). Abortando."); return
        logger.info(f"Dados históricos carregados: {len(df_full)} sorteios.")

        # 3. Pré-processamento Labels
        logger.info("Etapa 3: Pré-processamento dos Labels (Resultados Sorteados)...")
        encoded_labels, mlb, valid_indices = preprocess_data_labels(df_full, config['num_features_base'])
        if encoded_labels is None or mlb is None or valid_indices is None: logger.critical("Falha na Etapa 3 (Labels). Abortando."); return
        df_full_valid = df_full.loc[valid_indices].reset_index(drop=True)
        logger.info(f"Labels processados: {len(df_full_valid)} sorteios válidos restantes.")
        min_data_needed = config.get('sequence_length', 20) * 3
        if len(df_full_valid) < min_data_needed: logger.error(f"Dados insuficientes ({len(df_full_valid)}) após limpeza. Mínimo ~{min_data_needed}. Abortando."); return

        # 4. Cálculo das Features
        logger.info("Etapa 4: Cálculo das Features (Tempo e Estatísticas)...")
        bola_cols = [f'Bola{i+1}' for i in range(NUM_DEZENAS)] # Colunas Lotofácil
        # 4a. Features de Tempo
        logger.info("  4a: Calculando Features de Tempo Complexas...")
        time_features_raw = add_complex_time_features(df_full_valid[bola_cols], config['num_features_base'])
        if time_features_raw is None or len(time_features_raw) != len(encoded_labels): logger.critical("Falha no cálculo Features de Tempo (4a). Abortando."); return
        config['num_features_time'] = time_features_raw.shape[1]
        logger.info(f"   -> Features Tempo. Shape: {time_features_raw.shape}. Config 'num_features_time': {config['num_features_time']}")
        # 4b. Features Estatísticas
        logger.info("  4b: Calculando Features Estatísticas...")
        statistical_features_raw = add_statistical_features(df_full_valid[bola_cols], config['num_features_base'], config['rolling_freq_windows'])
        if statistical_features_raw is None or len(statistical_features_raw) != len(encoded_labels): logger.critical("Falha no cálculo Features Estatísticas (4b). Abortando."); return
        actual_stat_count = statistical_features_raw.shape[1]
        if config['num_features_statistical'] != actual_stat_count:
             logger.warning(f"Num features stat ({actual_stat_count}) difere do config ({config['num_features_statistical']}). Ajustando config.")
             config['num_features_statistical'] = actual_stat_count
        logger.info(f"   -> Features Estatísticas. Shape: {statistical_features_raw.shape}. Config 'num_features_statistical': {config['num_features_statistical']}")

        # 5. Cálculo e Validação Final Features Totais
        logger.info("Etapa 5: Cálculo e Validação Final Features Totais...")
        config['num_features_total'] = config['num_features_base'] + config['num_features_time'] + config['num_features_statistical']
        logger.info(f" -> Config 'num_features_total' calculado: {config['num_features_total']}")
        if not validate_config(config, check_total_features=True): logger.critical("Configuração final inválida. Abortando."); return
        logger.info("Contagem total de features validada.")

        # 6. Teste de Hiperparâmetros (Opcional)
        logger.info("Etapa 6: Verificação de Teste de Hiperparâmetros...")
        run_hyperparameter_tuning = config.get('test_hyperparameters', False) and hyperparameter_tuning_available
        if run_hyperparameter_tuning:
            logger.info("--- MODO TESTE DE HIPERPARÂMETROS ATIVADO ---")
            if config.get('hyperparameter_search') is None: logger.error("'hyperparameter_search' não definido no config. Abortando."); return
            try:
                logger.info("Instanciando HyperparameterTuner...")
                tuner = HyperparameterTuner( base_config=config.copy(), encoded_labels=encoded_labels, time_features_raw=time_features_raw,
                                             statistical_features_raw=statistical_features_raw, build_model_fn=build_model,
                                             split_data_fn=split_data, validate_config_fn=validate_config, output_dir=output_dir )
                logger.info("Iniciando busca de hiperparâmetros...")
                best_params = tuner.run_search()
                if best_params and isinstance(best_params, dict):
                    logger.info("*"*10 + " MELHORES HIPERPARÂMETROS " + "*"*10 + f"\n{json.dumps(best_params, indent=2)}")
                    logger.info("Aplicando melhor config encontrada...")
                    config.update(best_params) # Atualiza config global
                    logger.info("Recalculando num_features_total com melhores params...")
                    config['num_features_total'] = config['num_features_base'] + config['num_features_time'] + config['num_features_statistical']
                    logger.info(f" -> Config 'num_features_total' recalculado: {config['num_features_total']}")
                    if not validate_config(config, check_total_features=True): logger.critical("Config final com melhores params é inválida. Abortando."); return
                    logger.info("Config final com melhores params validada.")
                else: logger.warning("Não foi possível determinar melhor config. Continuando com a original.")
            except Exception as e_tuner: logger.error(f"Erro durante teste de hiperparâmetros: {e_tuner}", exc_info=True); logger.warning("Continuando com config original.")
            logger.info("--- FIM TESTE DE HIPERPARÂMETROS ---")
        else: logger.info("Teste de hiperparâmetros desativado ou módulo indisponível.")

        # 7. Divisão Final / Escalonamento / Sequenciamento
        logger.info("Etapa 7: Divisão Final / Escalonamento / Sequenciamento...")
        X_train, X_val, X_test, y_train, y_val, y_test, time_scaler, stat_scaler = split_data(
            encoded_labels, time_features_raw, statistical_features_raw,
            config['test_size_ratio'], config['validation_split_ratio'], config['sequence_length'] )
        if any(data is None for data in [X_train, y_train, time_scaler, stat_scaler]): # Checa essenciais
             logger.critical("Falha na Etapa 7 (Divisão/Escalonamento). Abortando."); return
        if X_train.size == 0 or y_train.size == 0: logger.critical("Treino Vazio após Etapa 7! Abortando."); return
        logger.info("Dados divididos, escalados e sequenciados.")

        # 8. Construção do Modelo GRU Final
        logger.info("Etapa 8: Construção do Modelo GRU Final...")
        model = build_model( config['sequence_length'], config['num_features_total'], config['num_features_base'],
                             config['gru_units'], config['dropout_rate'], config['use_batch_norm'] )
        if model is None: logger.critical("Falha na Etapa 8 (Construção Modelo). Abortando."); return
        logger.info("Modelo GRU construído.")

        # 9. Treinamento do Modelo Final
        logger.info("Etapa 9: Treinamento do Modelo Final...")
        tb_log_dir = os.path.join(config['tensorboard_log_dir'], datetime.now().strftime("%Y%m%d-%H%M%S"))
        history = train_model( model, X_train, y_train, X_val, y_val, config['epochs'], config['batch_size'], tb_log_dir )
        if history is None: logger.critical("Falha na Etapa 9 (Treinamento). Abortando."); return
        logger.info("Modelo treinado.")

        # 10. Avaliação do Modelo Final
        logger.info("Etapa 10: Avaliação do Modelo Final no Teste...")
        evaluation_results = None
        if X_test is not None and X_test.size > 0 and y_test is not None and y_test.size > 0:
             evaluation_results = evaluate_model(model, X_test, y_test, config['batch_size'])
             if evaluation_results is None: logger.warning("Falha na função de avaliação (Etapa 10)."); evaluation_results = {'basic_metrics': {}, 'real_hits': None}
        else: logger.warning("Teste vazio. Pulando avaliação final."); evaluation_results = {'basic_metrics': {}, 'real_hits': None}
        logger.info("Avaliação no teste concluída (ou pulada).")

        # 11. Previsão Próximo Sorteio
        logger.info("Etapa 11: Previsão para o Próximo Sorteio Lotofácil...")
        predicted_numbers, predicted_probabilities = None, None
        try:
            final_seq_len = config['sequence_length']
            if len(encoded_labels) >= final_seq_len:
                last_labels = encoded_labels[-final_seq_len:]
                last_time_raw = time_features_raw[-final_seq_len:]
                last_stat_raw = statistical_features_raw[-final_seq_len:]
                predicted_numbers, predicted_probabilities = predict_next_draw(
                    model, last_labels, last_time_raw, last_stat_raw, time_scaler, stat_scaler, mlb ) # Usa default de 15 predições
            else: logger.error(f"Dados insuficientes ({len(encoded_labels)}) para extrair última sequência de {final_seq_len}.")
        except Exception as e_last_seq: logger.error(f"Erro ao preparar/prever última sequência: {e_last_seq}", exc_info=True)
        if predicted_numbers is None or predicted_probabilities is None: logger.critical("Falha na Etapa 11 (Previsão). Abortando."); return
        logger.info("Previsão para próximo sorteio realizada.")

        # 12. Visualizações
        logger.info("Etapa 12: Geração de Visualizações...")
        try:
            plot_training_history(history)
            plot_prediction_analysis(predicted_numbers, predicted_probabilities, df_full_valid, config['sequence_length'])
            if X_test is not None and X_test.size > 0 and y_test is not None and y_test.size > 0:
                plot_hits_over_time(model, X_test, y_test, mlb)
            else: logger.info("Pulando gráfico acertos (teste vazio).")
            logger.info("Visualizações geradas.")
        except Exception as e_viz: logger.error(f"Erro na geração de visualizações: {e_viz}", exc_info=True)

        # 13. Exportação Resultados
        logger.info("Etapa 13: Exportação dos Resultados para Excel...")
        export_results(df_full_valid, predicted_numbers, predicted_probabilities, evaluation_results, config)
        # logger.info("Exportação concluída.") # Log Reduzido

        # --- Conclusão ---
        run_end_time = datetime.now()
        logger.info("-" * 60)
        logger.info(f"--- Processo Lotofácil V3.1-LF CONCLUÍDO ---")
        logger.info(f"Tempo Total: {run_end_time - run_start_time}")
        logger.info(f"Log principal: {log_file}")
        logger.info(f"Resultados exportados: {config.get('export_file', 'N/A')}")
        logger.info(f"Gráficos salvos em: {output_dir}")
        logger.info(f"Logs TensorBoard: {config.get('tensorboard_log_dir', 'N/A')}")
        if run_hyperparameter_tuning: logger.info(f"Resultados Hiperparâmetros: {config.get('hyperparameter_search', {}).get('export_results_file', os.path.join(output_dir, 'hyperparameter_results.xlsx'))}")
        logger.info("-" * 60)
        logger.info("AVISO FINAL: Modelo experimental. Loteria é aleatória. NÃO HÁ GARANTIA DE ACERTO.")
        logger.info("Jogue com responsabilidade.")
        logger.info("-" * 60)

    except Exception as e:
        logger.critical(f"Erro GERAL inesperado no fluxo principal: {e}", exc_info=True)
        logger.info("-" * 60 + "\n--- Processo INTERROMPIDO devido a erro crítico ---\n" + "-" * 60)
    finally:
        logging.shutdown()

if __name__ == "__main__":
    main()