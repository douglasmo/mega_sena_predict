#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para executar o modelo de previsão da Mega-Sena
Verifica se a pasta output existe, cria se necessário e executa o programa principal
"""

import os
import sys
import subprocess

def main():
    """Verifica a existência da pasta output e executa o programa principal"""
    
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
    
    # Executa o script principal
    print(f"Executando {main_script}...")
    try:
        subprocess.run([sys.executable, main_script], check=True)
        print("===== Execução concluída com sucesso! =====")
        print(f"Todos os resultados foram salvos na pasta '{output_dir}'")
    except subprocess.CalledProcessError as e:
        print(f"ERRO na execução: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 