#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para executar o modelo de previsão da Mega-Sena
Verifica se a pasta output existe, cria se necessário e executa o programa principal
"""

import os
import sys
import json
import argparse
import subprocess

def main():
    """Verifica a existência da pasta output e executa o programa principal"""
    
    # Configurar argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Executa o modelo de previsão da Mega-Sena')
    parser.add_argument('--test-hyperparameters', action='store_true', 
                        help='Ativa o modo de teste de hiperparâmetros')
    parser.add_argument('--method', choices=['grid', 'random'], default='grid',
                        help='Método de busca de hiperparâmetros (grid ou random)')
    parser.add_argument('--iterations', type=int, default=20,
                        help='Número máximo de iterações para teste de hiperparâmetros')
    parser.add_argument('--config', type=str, default='configv3.json',
                        help='Arquivo de configuração a ser usado')
    args = parser.parse_args()
    
    print("===== Iniciando execução da previsão da Mega-Sena =====")
    
    # Verifica e cria a pasta output se não existir
    output_dir = "output"
    if not os.path.exists(output_dir):
        print(f"Criando diretório de saída: {output_dir}")
        os.makedirs(output_dir)
    else:
        print(f"Diretório de saída já existe: {output_dir}")
    
    # Verifica se o arquivo principal existe
    main_script = "mega_sena_v3.py"
    if not os.path.exists(main_script):
        print(f"ERRO: Arquivo {main_script} não encontrado!")
        sys.exit(1)
    
    # Se o modo de teste de hiperparâmetros estiver ativado, modificar o arquivo de configuração
    if args.test_hyperparameters:
        print("Modo de teste de hiperparâmetros ativado!")
        
        if not os.path.exists(args.config):
            print(f"ERRO: Arquivo de configuração {args.config} não encontrado!")
            sys.exit(1)
        
        try:
            # Carregar configuração
            with open(args.config, 'r') as f:
                config = json.load(f)
            
            # Ativar teste de hiperparâmetros
            config['test_hyperparameters'] = True
            
            # Configurar método de busca
            if 'hyperparameter_search' not in config:
                config['hyperparameter_search'] = {}
            
            config['hyperparameter_search']['method'] = args.method
            config['hyperparameter_search']['n_iterations'] = args.iterations
            
            # Salvar configuração modificada
            config_backup = f"{args.config}.bak"
            print(f"Criando backup da configuração em: {config_backup}")
            with open(config_backup, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"Atualizando configuração para teste de hiperparâmetros...")
            with open(args.config, 'w') as f:
                json.dump(config, f, indent=4)
                
            print(f"Configuração atualizada: test_hyperparameters=True, method={args.method}, iterations={args.iterations}")
            
        except Exception as e:
            print(f"ERRO ao modificar arquivo de configuração: {e}")
            sys.exit(1)
    
    # Executa o script principal
    print(f"Executando {main_script}...")
    try:
        subprocess.run([sys.executable, main_script], check=True)
        print("===== Execução concluída com sucesso! =====")
        print(f"Todos os resultados foram salvos na pasta '{output_dir}'")
        
        if args.test_hyperparameters:
            print(f"Os resultados do teste de hiperparâmetros estão em:")
            print(f"  - {os.path.join(output_dir, 'hyperparameter_results.xlsx')}")
            print(f"  - {os.path.join(output_dir, 'hyperparameter_analysis.png')}")
            print(f"  - {os.path.join(output_dir, 'top_hyperparameters.png')}")
            
    except subprocess.CalledProcessError as e:
        print(f"ERRO na execução: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 