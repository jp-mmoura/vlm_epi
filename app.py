"""
Inspetor de EPIs com VLM (Vision Language Model)
------------------------------------------------
Aplicação Streamlit que recebe uma foto de canteiro de obras / área industrial,
envia para um VLM e devolve:
  - descrição da cena
  - lista de pessoas com EPIs presentes / ausentes
  - conformidade por pessoa e geral
  - riscos adicionais visíveis
Também permite fazer perguntas livres sobre a imagem (Visual Question Answering).

Backends suportados:
  - Ollama (local, gratuito)  -> ex.: qwen2.5vl:7b, llava:7b, gemma3:4b
  - Claude API (Anthropic)    -> requer ANTHROPIC_API_KEY

Executar:  streamlit run app.py
"""

import base64
import io
import json
import os
import re
import unicodedata
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
EPIS_DISPONIVEIS = [
    "Capacete",
    "Colete refletivo",
    "Óculos de proteção",
    "Luvas",
    "Calçado de segurança (botina)",
    "Protetor auricular",
    "Máscara / respirador",
    "Cinto paraquedista (trabalho em altura)",
]
EPIS_PADRAO = ["Capacete", "Colete refletivo", "Óculos de proteção", "Calçado de segurança (botina)"]
# palavras-chave (sem acento, minúsculas) usadas para casar o texto livre do modelo com cada EPI
PALAVRAS_CHAVE = {
    "Capacete": ["capacete"],
    "Colete refletivo": ["colete"],
    "Óculos de proteção": ["oculos"],
    "Luvas": ["luva"],
    "Calçado de segurança (botina)": ["calcado", "botina", "bota", "sapato"],
    "Protetor auricular": ["auricular", "abafador", "plug"],
    "Máscara / respirador": ["mascara", "respirador"],
    "Cinto paraquedista (trabalho em altura)": ["cinto", "paraquedista", "talabarte"],
}
MAX_LADO_PX = 1280  # reduz a imagem para economizar tokens / memória

SYSTEM_PROMPT = """Você é um técnico de segurança do trabalho experiente, especialista em
inspeção visual de EPIs (NR-6) em canteiros de obras e ambientes industriais.
Analise SOMENTE o que é visível na imagem. Não invente pessoas nem equipamentos.
Quando um item não puder ser verificado (pessoa de costas, item fora do enquadramento,
imagem borrada), classifique-o como "nao_verificavel" em vez de presente ou ausente."""

PROMPT_INSPECAO = """Inspecione a imagem quanto ao uso dos seguintes EPIs obrigatórios:
{epis}

Responda APENAS com um JSON válido, sem texto antes ou depois, no formato:
{{
  "descricao_cena": "1-2 frases descrevendo o ambiente e a atividade",
  "pessoas": [
    {{
      "id": 1,
      "localizacao": "posição na imagem (ex.: esquerda, primeiro plano)",
      "atividade": "o que a pessoa está fazendo",
      "epis_presentes": ["..."],
      "epis_ausentes": ["..."],
      "epis_nao_verificaveis": ["..."],
      "observacoes": "uso incorreto, ex.: capacete sem jugular, colete aberto"
    }}
  ],
  "riscos_adicionais": ["outros riscos visíveis: queda, material solto, falta de sinalização..."],
  "confianca": "alta | media | baixa"
}}
Use exatamente os nomes de EPI da lista acima. Se não houver pessoas, retorne "pessoas": []."""


# ---------------------------------------------------------------------------
# Utilidades de imagem
# ---------------------------------------------------------------------------
def preparar_imagem(arquivo) -> tuple[Image.Image, str]:
    """Abre, converte para RGB, redimensiona e retorna (imagem, base64 JPEG)."""
    img = Image.open(arquivo).convert("RGB")
    img.thumbnail((MAX_LADO_PX, MAX_LADO_PX))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return img, base64.b64encode(buf.getvalue()).decode("utf-8")


def normalizar(texto) -> str:
    """Minúsculas e sem acentos: 'Óculos de Proteção' -> 'oculos de protecao'."""
    sem_acento = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return sem_acento.lower().strip()


def normalizar_chaves(obj):
    """Remove acentos das chaves do JSON (o modelo às vezes devolve 'confiança' em vez de 'confianca')."""
    if isinstance(obj, dict):
        return {normalizar(k).replace(" ", "_"): normalizar_chaves(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalizar_chaves(v) for v in obj]
    return obj


def extrair_json(texto: str) -> dict:
    """Extrai o primeiro objeto JSON da resposta do modelo (tolerante a ```json``` e texto extra)."""
    texto = re.sub(r"```(?:json)?", "", texto).strip()
    inicio, fim = texto.find("{"), texto.rfind("}")
    if inicio == -1 or fim == -1:
        raise ValueError("O modelo não retornou JSON.")
    return normalizar_chaves(json.loads(texto[inicio : fim + 1]))


# ---------------------------------------------------------------------------
# Backends VLM
# ---------------------------------------------------------------------------
def chamar_ollama(prompt, img_b64, modelo, url, historico=None, json_mode=False):
    mensagens = [{"role": "system", "content": SYSTEM_PROMPT}]
    # a imagem vai na primeira mensagem do usuário
    primeira = {"role": "user", "content": prompt if not historico else historico[0]["content"], "images": [img_b64]}
    mensagens.append(primeira)
    if historico:
        mensagens += historico[1:]
        mensagens.append({"role": "user", "content": prompt})
    payload = {"model": modelo, "messages": mensagens, "stream": False, "options": {"temperature": 0.1}}
    if json_mode:
        payload["format"] = "json"
    r = requests.post(f"{url.rstrip('/')}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]


def chamar_claude(prompt, img_b64, modelo, api_key, historico=None, json_mode=False):
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    bloco_img = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}}
    if historico:
        mensagens = [{"role": "user", "content": [bloco_img, {"type": "text", "text": historico[0]["content"]}]}]
        mensagens += historico[1:]
        mensagens.append({"role": "user", "content": prompt})
    else:
        mensagens = [{"role": "user", "content": [bloco_img, {"type": "text", "text": prompt}]}]
    resp = client.messages.create(
        model=modelo, max_tokens=2000, system=SYSTEM_PROMPT, messages=mensagens
    )
    return resp.content[0].text


def chamar_vlm(prompt, img_b64, cfg, historico=None, json_mode=False):
    if cfg["backend"] == "Ollama (local)":
        return chamar_ollama(prompt, img_b64, cfg["modelo"], cfg["ollama_url"], historico, json_mode)
    return chamar_claude(prompt, img_b64, cfg["modelo"], cfg["api_key"], historico, json_mode)


# ---------------------------------------------------------------------------
# Regras de negócio
# ---------------------------------------------------------------------------
def mencionado(epi: str, lista: list) -> bool:
    """True se algum item escrito pelo modelo corresponde ao EPI (por palavra-chave, sem acento/maiúscula)."""
    chaves = PALAVRAS_CHAVE.get(epi, [normalizar(epi)])
    return any(k in normalizar(item) for item in lista or [] for k in chaves)


def classificar(p: dict, epis_obrigatorios: list[str]) -> tuple[list, list, list]:
    """Classifica cada EPI obrigatório em (presentes, ausentes, não verificáveis).
    Ausência tem prioridade; EPI obrigatório que o modelo não citou conta como não verificável,
    para nunca virar 'Conforme' por omissão."""
    presentes, ausentes, nao_verif = [], [], []
    for epi in epis_obrigatorios:
        if mencionado(epi, p.get("epis_ausentes")):
            ausentes.append(epi)
        elif mencionado(epi, p.get("epis_nao_verificaveis")):
            nao_verif.append(epi)
        elif mencionado(epi, p.get("epis_presentes")):
            presentes.append(epi)
        else:
            nao_verif.append(epi)
    return presentes, ausentes, nao_verif


def avaliar(resultado: dict, epis_obrigatorios: list[str]) -> pd.DataFrame:
    """Monta a tabela por pessoa e calcula o status de conformidade."""
    linhas = []
    for p in resultado.get("pessoas", []):
        presentes, ausentes, nao_verif = classificar(p, epis_obrigatorios)
        if ausentes:
            status = "❌ Não conforme"
        elif nao_verif:
            status = "⚠️ Verificar em campo"
        else:
            status = "✅ Conforme"
        linhas.append(
            {
                "Pessoa": p.get("id"),
                "Localização": p.get("localizacao", ""),
                "Atividade": p.get("atividade", ""),
                "Presentes": ", ".join(presentes),
                "Ausentes": ", ".join(ausentes),
                "Não verificáveis": ", ".join(nao_verif),
                "Observações": p.get("observacoes", ""),
                "Status": status,
            }
        )
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Inspetor de EPIs com VLM", page_icon="🦺", layout="wide")
st.title("🦺 Inspetor de EPIs com VLM")
st.caption("Análise visual de conformidade de EPIs em canteiros de obras usando Vision Language Models. "
           "Ferramenta de apoio — não substitui a inspeção do técnico de segurança.")

with st.sidebar:
    st.header("⚙️ Modelo")
    backend = st.radio("Backend", ["Ollama (local)", "Claude API"])
    if backend == "Ollama (local)":
        modelo = st.text_input("Modelo", "qwen2.5vl:7b", help="Ex.: qwen2.5vl:7b, llava:7b, gemma3:4b")
        ollama_url = st.text_input("URL do Ollama", "http://localhost:11434")
        api_key = None
    else:
        modelo = st.text_input("Modelo", "claude-sonnet-5-5")
        api_key = st.text_input("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""), type="password")
        ollama_url = None

    st.header("🦺 EPIs obrigatórios")
    epis = st.multiselect("Verificar", EPIS_DISPONIVEIS, default=EPIS_PADRAO)

cfg = {"backend": backend, "modelo": modelo, "ollama_url": ollama_url, "api_key": api_key}

col_img, col_res = st.columns([1, 1.4])

with col_img:
    fonte = st.radio("Imagem", ["Upload", "Câmera"], horizontal=True)
    arquivo = st.file_uploader("Foto do local", type=["jpg", "jpeg", "png", "webp"]) if fonte == "Upload" \
        else st.camera_input("Tire uma foto")

    if arquivo:
        img, img_b64 = preparar_imagem(arquivo)
        # reseta o estado quando a imagem muda
        if st.session_state.get("img_id") != img_b64[:200]:
            st.session_state.update(img_id=img_b64[:200], resultado=None, chat=[])
        st.image(img, use_container_width=True)
        analisar = st.button("🔍 Analisar conformidade", type="primary", use_container_width=True,
                             disabled=not epis or (backend == "Claude API" and not api_key))

with col_res:
    if arquivo and analisar:
        with st.spinner(f"Analisando com {modelo}..."):
            try:
                prompt = PROMPT_INSPECAO.format(epis="\n".join(f"- {e}" for e in epis))
                bruto = chamar_vlm(prompt, img_b64, cfg, json_mode=True)
                st.session_state.resultado = extrair_json(bruto)
                st.session_state.prompt_inspecao = prompt
                st.session_state.bruto = bruto
                st.session_state.chat = []
            except Exception as e:
                st.error(f"Erro na análise: {e}")

    res = st.session_state.get("resultado") if arquivo else None
    if res:
        df = avaliar(res, epis)
        n = len(df)
        nc = int((df["Status"] == "❌ Não conforme").sum()) if n else 0
        verif = int((df["Status"] == "⚠️ Verificar em campo").sum()) if n else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pessoas", n)
        m2.metric("Conformes", n - nc - verif)
        m3.metric("Não conformes", nc)
        m4.metric("Confiança do modelo", str(res.get("confianca", "-")).capitalize())

        if n == 0:
            st.info("Nenhuma pessoa identificada na imagem.")
        elif nc:
            st.error(f"🚨 {nc} pessoa(s) sem EPI obrigatório.")
        elif verif:
            st.warning("Itens não verificáveis pela imagem — confirmar em campo.")
        else:
            st.success("Todos os EPIs obrigatórios visíveis.")

        st.subheader("Cena")
        st.write(res.get("descricao_cena", ""))

        if n:
            st.subheader("Pessoas")
            st.dataframe(df, use_container_width=True, hide_index=True)

        riscos = res.get("riscos_adicionais", [])
        if riscos:
            st.subheader("Riscos adicionais")
            for r in riscos:
                st.markdown(f"- {r}")

        # exportação
        relatorio = {"data_hora": datetime.now().isoformat(timespec="seconds"), "modelo": modelo,
                     "epis_obrigatorios": epis, **res}
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Relatório JSON", json.dumps(relatorio, ensure_ascii=False, indent=2),
                           "inspecao_epi.json", "application/json", use_container_width=True)
        c2.download_button("⬇️ Tabela CSV", df.to_csv(index=False, sep=";").encode("utf-8-sig"),
                           "inspecao_epi.csv", "text/csv", use_container_width=True)
        with st.expander("Resposta bruta do modelo"):
            st.code(st.session_state.get("bruto", ""), language="json")

        # ------------------- Perguntas livres (VQA) -------------------
        st.subheader("💬 Pergunte sobre a imagem")
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        pergunta = st.chat_input("Ex.: A pessoa 2 está usando a jugular do capacete?")
        if pergunta:
            historico = [{"role": "user", "content": st.session_state.prompt_inspecao},
                         {"role": "assistant", "content": st.session_state.bruto}] + st.session_state.chat
            with st.spinner("Pensando..."):
                try:
                    resposta = chamar_vlm(pergunta + "\nResponda em português, em texto corrido.",
                                          img_b64, cfg, historico=historico)
                except Exception as e:
                    resposta = f"Erro: {e}"
            st.session_state.chat += [{"role": "user", "content": pergunta},
                                      {"role": "assistant", "content": resposta}]
            st.rerun()
    elif not arquivo:
        st.info("Envie uma foto ou use a câmera para começar.")
