# -*- coding: utf-8 -*-
"""
Módulo para otimização de hiperparâmetros do modelo de previsão da Mega-Sena
Implementa Grid Search e Random Search para encontrar a melhor configuração
"""

import numpy as np
import pandas as pd
import os
import time
import copy
import logging
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf

# Configurar logger
logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """Classe para otimização de hiperparâmetros do modelo GRU"""
    
    def __init__(self, base_config, encoded_labels, time_features_raw, statistical_features_raw, 
                 build_model_fn, split_data_fn, validate_config_fn, output_dir="output"):
        """
        Inicializa o otimizador de hiperparâmetros.
        
        Args:
            base_config: Configuração base do modelo
            encoded_labels: Labels codificados (MLBinarizer)
            time_features_raw: Features de tempo não-escaladas
            statistical_features_raw: Features estatísticas não-escaladas
            build_model_fn: Função para construir o modelo (referência)
            split_data_fn: Função para dividir os dados (referência)
            validate_config_fn: Função para validar a configuração (referência)
            output_dir: Diretório de saída
        """
        self.base_config = copy.deepcopy(base_config)
        self.search_config = self.base_config.get('hyperparameter_search', {})
        self.encoded_labels = encoded_labels
        self.time_features_raw = time_features_raw
        self.statistical_features_raw = statistical_features_raw
        self.build_model_fn = build_model_fn
        self.split_data_fn = split_data_fn
        self.validate_config_fn = validate_config_fn
        self.output_dir = output_dir
        self.results = []
        
        # Arquivo de resultados
        self.results_file = self.search_config.get('export_results_file', 
                                               os.path.join(output_dir, 'hyperparameter_results.xlsx'))
        
    def generate_params_grid(self):
        """Gera todas as combinações de parâmetros para Grid Search"""
        param_grid = self.search_config.get('param_grid', {})
        if not param_grid:
            logger.error("Nenhum grid de parâmetros definido!")
            return []
        
        # Gerar todas as combinações
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        # Limitar o número de combinações se necessário
        n_iterations = self.search_config.get('n_iterations', 20)
        all_combinations = list(itertools.product(*values))
        total_combinations = len(all_combinations)
        
        if total_combinations > n_iterations and self.search_config.get('method') == 'random':
            logger.info(f"Gerando {n_iterations} combinações aleatórias de {total_combinations} possíveis...")
            selected_indices = np.random.choice(total_combinations, n_iterations, replace=False)
            all_combinations = [all_combinations[i] for i in selected_indices]
        
        # Converter combinações em dicionários
        param_sets = []
        for combo in all_combinations:
            param_dict = {keys[i]: combo[i] for i in range(len(keys))}
            param_sets.append(param_dict)
            
        logger.info(f"Geradas {len(param_sets)} combinações de hiperparâmetros para teste.")
        return param_sets
    
    def evaluate_model(self, model, X_val, y_val, X_test, y_test, batch_size=32):
        """Avalia o modelo em dados de validação e teste"""
        val_results = None
        test_results = None
        
        if X_val is not None and y_val is not None and X_val.size > 0:
            val_results = model.evaluate(X_val, y_val, batch_size=batch_size, verbose=0)
            
        if X_test is not None and y_test is not None and X_test.size > 0:
            test_results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
            
        return val_results, test_results
    
    def train_and_evaluate_config(self, config, verbose=False):
        """Treina e avalia uma configuração específica de hiperparâmetros"""
        start_time = time.time()
        
        if not self.validate_config_fn(config):
            logger.error("Configuração inválida para teste!")
            return None
        
        # Dividir os dados com a nova configuração
        X_train, X_val, X_test, y_train, y_val, y_test, time_scaler, stat_scaler = self.split_data_fn(
            self.encoded_labels,
            self.time_features_raw,
            self.statistical_features_raw,
            config['test_size_ratio'],
            config['validation_split_ratio'],
            config['sequence_length']
        )
        
        # Construir o modelo
        model = self.build_model_fn(
            config['sequence_length'],
            config['num_features_total'],
            config['num_features_base'],
            config['gru_units'],
            config['dropout_rate'],
            config['use_batch_norm']
        )
        
        if model is None:
            logger.error("Erro ao construir modelo para configuração!")
            return None
        
        if verbose:
            logger.info(f"Treinando modelo com: seq_len={config['sequence_length']}, "
                      f"gru_units={config['gru_units']}, batch_size={config['batch_size']}, "
                      f"dropout={config['dropout_rate']}, batch_norm={config['use_batch_norm']}")
        
        # Callback de early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss' if X_val.size > 0 else 'loss',
            patience=self.search_config.get('early_stopping_patience', 25),
            restore_best_weights=True,
            verbose=0
        )
        
        # Treinar
        history = model.fit(
            X_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_data=(X_val, y_val) if X_val.size > 0 else None,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Avaliar
        val_metrics, test_metrics = self.evaluate_model(model, X_val, y_val, X_test, y_test, config['batch_size'])
        
        # Coletar métricas
        metrics = {}
        
        if val_metrics is not None:
            metrics_dict = dict(zip(model.metrics_names, val_metrics))
            metrics['val_loss'] = metrics_dict.get('loss')
            metrics['val_binary_accuracy'] = metrics_dict.get('binary_accuracy')
            metrics['val_auc'] = metrics_dict.get('auc')
        
        if test_metrics is not None:
            metrics_dict = dict(zip(model.metrics_names, test_metrics))
            metrics['test_loss'] = metrics_dict.get('loss')
            metrics['test_binary_accuracy'] = metrics_dict.get('binary_accuracy')
            metrics['test_auc'] = metrics_dict.get('auc')
        
        metrics['epochs_used'] = len(history.history['loss'])
        metrics['training_time'] = time.time() - start_time
        
        # Limpar memória
        tf.keras.backend.clear_session()
        
        result = {
            'params': {
                'sequence_length': config['sequence_length'],
                'gru_units': config['gru_units'],
                'dropout_rate': config['dropout_rate'],
                'use_batch_norm': config['use_batch_norm'],
                'batch_size': config['batch_size']
            },
            'metrics': metrics
        }
        
        if verbose:
            logger.info(f"Resultado: val_auc={metrics.get('val_auc', 'N/A'):.4f}, "
                      f"test_auc={metrics.get('test_auc', 'N/A'):.4f}, "
                      f"epochs={metrics['epochs_used']}, "
                      f"time={metrics['training_time']:.1f}s")
        
        return result
    
    def run_search(self):
        """Executa a busca de hiperparâmetros e retorna os melhores resultados"""
        param_sets = self.generate_params_grid()
        if not param_sets:
            logger.error("Não foi possível gerar conjuntos de parâmetros para teste!")
            return None
            
        logger.info(f"Iniciando busca de hiperparâmetros - Método: {self.search_config.get('method', 'grid')}")
        logger.info(f"Testando {len(param_sets)} combinações de parâmetros...")
        
        # Garantir que as pastas existam
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        
        # Progress bar
        pbar = tqdm(total=len(param_sets), desc="Testando hiperparâmetros")
        
        for i, params in enumerate(param_sets):
            # Aplicar parâmetros à configuração de teste
            test_config = copy.deepcopy(self.base_config)
            for key, value in params.items():
                test_config[key] = value
            
            # Recalcular num_features_total caso sequence_length tenha mudado
            test_config['num_features_total'] = (
                test_config['num_features_base'] +
                test_config['num_features_time'] +
                test_config['num_features_statistical']
            )
            
            # Treinar e avaliar
            result = self.train_and_evaluate_config(test_config, verbose=(i % 5 == 0))
            if result:
                self.results.append(result)
                
                # Export parcial a cada 5 iterações
                if i % 5 == 0 and i > 0:
                    self._export_results_partial()
            
            pbar.update(1)
        
        pbar.close()
        
        # Ordenar resultados pelo AUC de validação
        self.results.sort(key=lambda x: x['metrics'].get('val_auc', 0), reverse=True)
        
        # Exportar resultados
        self._export_results()
        
        # Exportar gráficos
        self._plot_results()
        
        # Retornar a melhor configuração
        if self.results:
            best_params = self.results[0]['params']
            logger.info(f"Melhor configuração encontrada:")
            for key, value in best_params.items():
                logger.info(f"  {key}: {value}")
            logger.info(f"Métricas:")
            for key, value in self.results[0]['metrics'].items():
                logger.info(f"  {key}: {value}")
            
            return best_params
        return None
        
    def _export_results_partial(self):
        """Exporta resultados parciais para Excel"""
        try:
            # Criar DataFrame com os resultados
            results_df = pd.DataFrame([
                {**r['params'], **r['metrics']} for r in self.results
            ])
            
            # Salvar resultados parciais
            temp_file = self.results_file.replace('.xlsx', '_partial.xlsx')
            with pd.ExcelWriter(temp_file, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Resultados', index=False)
                logger.debug(f"Resultados parciais exportados para {temp_file}")
        except Exception as e:
            logger.error(f"Erro ao exportar resultados parciais: {e}")
    
    def _export_results(self):
        """Exporta resultados completos para Excel"""
        try:
            # Criar DataFrame com os resultados
            results_df = pd.DataFrame([
                {**r['params'], **r['metrics']} for r in self.results
            ])
            
            # Ordenar por val_auc (decrescente)
            if 'val_auc' in results_df.columns:
                results_df = results_df.sort_values('val_auc', ascending=False)
            
            # Salvar resultados
            with pd.ExcelWriter(self.results_file, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Resultados', index=False)
                
                # Criar tabela pivot para each parameter
                for param in self.search_config.get('param_grid', {}).keys():
                    if param in results_df.columns:
                        pivot_df = pd.pivot_table(
                            results_df, 
                            values=['val_auc', 'test_auc'], 
                            index=[param],
                            aggfunc=['mean', 'max', 'min', 'count']
                        )
                        pivot_df.to_excel(writer, sheet_name=f'Análise_{param}')
                
            logger.info(f"Resultados completos exportados para {self.results_file}")
            
        except Exception as e:
            logger.error(f"Erro ao exportar resultados: {e}")
    
    def _plot_results(self):
        """Gera gráficos de análise dos resultados de hiperparâmetros"""
        try:
            if not self.results:
                logger.warning("Sem resultados para gerar gráficos!")
                return
                
            # Criar DataFrame com os resultados
            results_df = pd.DataFrame([
                {**r['params'], **r['metrics']} for r in self.results
            ])
            
            # Para cada hiperparâmetro, criar um boxplot do AUC vs. valor do parâmetro
            param_keys = self.search_config.get('param_grid', {}).keys()
            fig, axes = plt.subplots(len(param_keys), 2, figsize=(16, 4 * len(param_keys)))
            
            for i, param in enumerate(param_keys):
                if param in results_df.columns:
                    # AUC de validação
                    ax1 = axes[i][0] if len(param_keys) > 1 else axes[0]
                    results_df.boxplot(column=['val_auc'], by=param, ax=ax1)
                    ax1.set_title(f'Validação AUC vs {param}')
                    ax1.set_ylabel('Validação AUC')
                    
                    # AUC de teste
                    ax2 = axes[i][1] if len(param_keys) > 1 else axes[1]
                    results_df.boxplot(column=['test_auc'], by=param, ax=ax2)
                    ax2.set_title(f'Teste AUC vs {param}')
                    ax2.set_ylabel('Teste AUC')
            
            plt.tight_layout()
            plot_file = os.path.join(self.output_dir, 'hyperparameter_analysis.png')
            plt.savefig(plot_file)
            logger.info(f"Gráficos de análise salvos em {plot_file}")
            plt.close()
            
            # Gráfico de dispersão para as melhores configurações
            top_n = min(20, len(results_df))
            top_df = results_df.sort_values('val_auc', ascending=False).head(top_n)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(top_df['val_auc'], top_df['test_auc'], alpha=0.7)
            
            for i, row in top_df.iterrows():
                plt.annotate(f"{i+1}", 
                           (row['val_auc'], row['test_auc']),
                           fontsize=9)
            
            plt.xlabel('Validação AUC')
            plt.ylabel('Teste AUC')
            plt.title(f'Top {top_n} Configurações: Validação vs Teste AUC')
            plt.grid(True, alpha=0.3)
            
            # Adicionar legenda das melhores configurações
            textstr = "\n".join([
                f"{i+1}: " + ", ".join([f"{p}={row[p]}" for p in param_keys])
                for i, (_, row) in enumerate(top_df.head(5).iterrows())
            ])
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.figtext(0.15, 0.05, textstr, fontsize=9, bbox=props)
            
            scatter_file = os.path.join(self.output_dir, 'top_hyperparameters.png')
            plt.savefig(scatter_file)
            logger.info(f"Gráfico de top configurações salvo em {scatter_file}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráficos de análise: {e}") 