#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Execute o modelo de previsão da Lotofácil de forma simples."""

import os
import sys
import subprocess
import argparse


def show_last_prediction(log_file):
    """Imprime a última previsão registrada no log da Lotofácil."""
    if not os.path.exists(log_file):
        print(f"Log não encontrado: {log_file}")
        return
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    start = None
    for idx in range(len(lines) - 1, -1, -1):
        if "PREVISÃO LOTOFÁCIL - PRÓXIMO SORTEIO" in lines[idx]:
            start = idx
            break
    if start is None:
        print("Não foi possível localizar a previsão no log.")
        return
    print("\n===== Previsão do próximo sorteio (Lotofácil) =====")
    for line in lines[start:]:
        line = line.strip()
        print(line)
        if "AVISO CRÍTICO" in line:
            break
    print("===================================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Executa o modelo da Lotofácil e exibe a previsão final."
    )
    parser.add_argument(
        "--config", default="config_lotofacil.json", help="Arquivo de configuração."
    )
    args = parser.parse_args()

    output_dir = "output_lotofacil"
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "lotofacil_v3.log")

    cmd = [sys.executable, "lotofacil.py"]
    if args.config:
        cmd += (
            ["--config", args.config]
            if "--config" in open("lotofacil.py").read()
            else cmd
        )

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERRO ao executar Lotofácil: {e}")
        sys.exit(1)

    # Exibe a previsão que foi registrada no log
    show_last_prediction(log_file)


if __name__ == "__main__":
    main()
