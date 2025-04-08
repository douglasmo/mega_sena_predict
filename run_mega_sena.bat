@echo off
echo ===== Execucao da previsao da Mega-Sena V3 =====

REM Ativar o ambiente virtual Python
echo Ativando ambiente virtual Python...
if exist ".env\Scripts\activate.bat" (
    call .env\Scripts\activate.bat
) else if exist ".env\bin\activate.bat" (
    call .env\bin\activate.bat
) else (
    echo AVISO: Ambiente virtual .env nao encontrado.
    echo Usando Python global do sistema.
    goto menu
)

:menu
echo.
echo Escolha o modo de execucao:
echo [1] Execucao normal
echo [2] Teste de hiperparametros (Grid Search)
echo [3] Teste de hiperparametros (Random Search)
echo [4] Sair
echo.
set /p opcao="Opcao: "

if "%opcao%"=="1" (
    echo.
    echo Iniciando execucao normal...
    python run_mega_sena.py
    goto fim
)

if "%opcao%"=="2" (
    echo.
    echo Iniciando teste de hiperparametros com Grid Search...
    python run_mega_sena.py --test-hyperparameters --method grid
    goto fim
)

if "%opcao%"=="3" (
    echo.
    set /p iterations="Numero de iteracoes [20]: "
    if "%iterations%"=="" set iterations=20
    echo Iniciando teste de hiperparametros com Random Search (%iterations% iteracoes)...
    python run_mega_sena.py --test-hyperparameters --method random --iterations %iterations%
    goto fim
)

if "%opcao%"=="4" (
    echo Saindo...
    goto fim
)

echo.
echo Opcao invalida! Tente novamente.
goto menu

:fim
echo.
REM Desativar o ambiente virtual Python
if defined VIRTUAL_ENV (
    echo Desativando ambiente virtual Python...
    call deactivate
)
echo Pressione qualquer tecla para sair...
pause > nul 