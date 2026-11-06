# 🦺 Inspetor de EPIs com VLM

Aplicação de **Segurança e Monitoramento** que usa um *Vision Language Model* para analisar fotos de canteiros de obras / áreas industriais e verificar o uso de EPIs (NR-6).

## O que faz
1. Recebe uma foto (upload ou câmera).
2. Envia a imagem + um prompt estruturado ao VLM, pedindo resposta em **JSON**.
3. Para cada pessoa identificada, lista EPIs **presentes**, **ausentes** e **não verificáveis**.
4. Aplica a regra de conformidade com base nos EPIs obrigatórios escolhidos:
   - ❌ **Não conforme** – falta algum EPI obrigatório
   - ⚠️ **Verificar em campo** – item não visível (pessoa de costas, fora do quadro)
   - ✅ **Conforme**
5. Aponta riscos adicionais visíveis (trabalho em altura sem proteção, material solto etc.).
6. Permite **perguntas livres sobre a imagem** (VQA), mantendo o contexto da análise.
7. Exporta o relatório em JSON e CSV.

## Arquitetura
```
Imagem ──► Pré-processamento (RGB, redimensiona 1280px, JPEG base64)
       ──► VLM (Ollama local ou Claude API) + prompt de sistema (técnico de segurança)
       ──► JSON estruturado ──► regras de conformidade (Python)
       ──► Interface Streamlit (métricas, tabela, riscos, chat, exportação)
```
O VLM faz a **percepção** (o que está na imagem); a **decisão** de conformidade é feita em código, de forma determinística, a partir da lista de EPIs obrigatórios.

## Como executar

### Opção A – Local e gratuito (Ollama)
```bash
# 1. Instale o Ollama: https://ollama.com
ollama pull qwen2.5vl:7b        # ou llava:7b / gemma3:4b (máquinas mais fracas)

# 2. Dependências e app
pip install -r requirements.txt
streamlit run app.py
```

### Opção B – Claude API
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sua-chave"     # Windows: set ANTHROPIC_API_KEY=sua-chave
streamlit run app.py
```
Na barra lateral, selecione **Claude API**.

## Limitações
- O VLM pode errar em imagens com baixa resolução, oclusão ou muitas pessoas; por isso existe o status "Verificar em campo" e o campo de confiança.
- Modelos locais pequenos (≤ 7B) são menos precisos que modelos de API.
- É uma ferramenta de **apoio**: não substitui a inspeção do técnico de segurança.
