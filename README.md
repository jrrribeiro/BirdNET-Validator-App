# BirdNET Validator HF App

Aplicacao de validacao de deteccoes para Hugging Face Spaces (Gradio), com foco em:
- Multi-projeto
- Audio sob demanda
- Persistencia de validacoes
- Escalabilidade para datasets grandes

## Desenvolvimento local
1. Criar ambiente virtual Python 3.11+
2. Instalar dependencias:
   pip install -r requirements.txt
3. Rodar app:
   python app.py

## Estrutura
- src/domain: modelos de dominio
- src/repositories: contratos de persistencia
- src/services: servicos de aplicacao
- src/auth: autenticacao e autorizacao
- src/ui: montagem da interface
- src/cache: cache efemero local
- cli: comandos de ingestao/publicacao
- tests: testes unitarios e integracao
