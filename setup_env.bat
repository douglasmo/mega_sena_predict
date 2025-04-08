@echo off
echo ===== Configuracao do Ambiente Python para Mega-Sena V3 =====

REM Verificar se Python esta instalado
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: Python nao encontrado!
    echo Por favor, instale o Python 3.8 ou superior.
    goto fim
)

echo Python encontrado. Verificando ambiente virtual...

REM Verificar se o ambiente virtual ja existe
if exist ".env" (
    echo Ambiente virtual .env ja existe.
    
    echo.
    set /p atualizar="Deseja atualizar as dependencias? (S/N): "
    if /i "%atualizar%"=="S" (
        echo Ativando ambiente virtual...
        call .env\Scripts\activate.bat
        
        echo Atualizando dependencias...
        pip install -r requirements.txt
        
        echo Ambiente atualizado com sucesso!
        call deactivate
    ) else (
        echo Operacao cancelada.
    )
) else (
    echo Criando novo ambiente virtual .env...
    python -m venv .env
    
    if %ERRORLEVEL% NEQ 0 (
        echo ERRO: Falha ao criar ambiente virtual!
        goto fim
    )
    
    echo Ativando ambiente virtual...
    call .env\Scripts\activate.bat
    
    echo Atualizando pip...
    python -m pip install --upgrade pip
    
    echo Instalando dependencias...
    pip install -r requirements.txt
    
    echo Ambiente configurado com sucesso!
    call deactivate
)

echo.
echo Ambiente Python preparado. Execute run_mega_sena.bat para iniciar o programa.

:fim
echo.
echo Pressione qualquer tecla para sair...
pause > nul 