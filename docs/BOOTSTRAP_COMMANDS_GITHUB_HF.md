# Comandos Exatos - GitHub Repo + Hugging Face Space

Este roteiro cria:
1) Repositorio do app no GitHub
2) Space Gradio no Hugging Face
3) Primeiro deploy do app

## Pre-requisitos
- Git instalado
- Conta GitHub autenticada no terminal (gh auth login)
- Conta Hugging Face com token (write)
- Python 3.11+

## A) Criar repositorio no GitHub e publicar codigo
PowerShell (Windows):

1. Ir para a pasta do novo app
   cd "c:\Users\jonat\Documents\Python\BirdNET-validator-App"

2. Inicializar git local
   git init
   git add .
   git commit -m "chore: bootstrap sprint 0 for HF validator app"

3. Criar repo remoto e enviar
   gh repo create BirdNET-validator-App --public --source . --remote origin --push

Observacao:
- Se quiser privado, troque --public por --private.

## B) Criar Space Gradio no Hugging Face
1. Login no HF CLI
   huggingface-cli login

2. Criar Space
   huggingface-cli repo create BirdNET-validator-App --type space --space_sdk gradio --public

Observacao:
- Para privado, troque --public por --private.

## C) Conectar app ao Space e fazer deploy inicial
1. Adicionar remoto do Space
   git remote add hf https://huggingface.co/spaces/SEU_USUARIO/BirdNET-validator-App

2. Enviar codigo para o Space
   git push hf main

Se sua branch principal local for master, use:
   git push hf master:main

## D) Configurar secrets no Space
No painel do Space (Settings > Variables and secrets), adicionar:
- HF_TOKEN (token com permissao write para datasets)
- APP_AUTH_SECRET (segredo do app)
- APP_ENV=prod

## E) Validar deploy
1. Abrir URL do Space:
   https://huggingface.co/spaces/SEU_USUARIO/BirdNET-validator-App
2. Verificar se a tela "BirdNET Validator HF" carregou.

## F) Primeiros comandos da CLI do projeto (quando implementar Sprint 1)
A partir da raiz do app:

1. Criar projeto dataset
   python -m cli.hf_dataset_cli create-project --project-slug ppbio-rabeca --dataset-repo SEU_USUARIO/birdnet-ppbio-rabeca-dataset

2. Verificar projeto
   python -m cli.hf_dataset_cli verify-project

## G) Criar issue backlog no repo novo
Use o arquivo da raiz do repo atual:
- GITHUB_ISSUES_HF_VALIDATOR.md

Sugestao:
- Criar milestones Sprint 0..6
- Abrir issues na ordem de dependencias do documento
