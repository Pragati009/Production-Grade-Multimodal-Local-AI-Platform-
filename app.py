import streamlit as st
import requests
import time

API = "http://127.0.0.1:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Local AI Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Overall background */
.stApp { background-color: #0f1117; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1a1d27;
    border-right: 1px solid #2e3147;
}

/* Chat message bubbles */
[data-testid="stChatMessage"] {
    background-color: #1e2130;
    border-radius: 12px;
    padding: 4px 8px;
    margin-bottom: 8px;
    border: 1px solid #2e3147;
}

/* Source citation boxes */
.source-box {
    background-color: #1a2035;
    border-left: 3px solid #4f8ef7;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.85rem;
    color: #b0b8d1;
    line-height: 1.5;
}

/* Stat cards */
.stat-card {
    background-color: #1e2130;
    border: 1px solid #2e3147;
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
}
.stat-number { font-size: 1.6rem; font-weight: 700; color: #4f8ef7; }
.stat-label  { font-size: 0.75rem; color: #7a8099; margin-top: 2px; }

/* Model badge */
.badge-base     { background:#1e3a5f; color:#4f8ef7; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-finetuned{ background:#1e3d2f; color:#3ddc84; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }

/* Tab styling */
.stTabs [data-baseweb="tab"] { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "total_q"     not in st.session_state: st.session_state.total_q     = 0
if "total_ms"    not in st.session_state: st.session_state.total_ms    = 0.0

# ── API helpers ───────────────────────────────────────────────────────────────
def api_online() -> bool:
    try:    requests.get(f"{API}/model", timeout=2); return True
    except: return False

def get_active_model() -> str:
    try:    return requests.get(f"{API}/model", timeout=2).json()["active_model"]
    except: return "unknown"

def switch_model(name: str):
    try: requests.post(f"{API}/model?name={name}", timeout=2)
    except: pass

def ask(prompt: str) -> dict:
    r = requests.get(f"{API}/chat", params={"prompt": prompt}, timeout=120)
    return r.json()

def ask_voice(audio_bytes: bytes) -> dict:
    r = requests.post(
        f"{API}/voice",
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
        timeout=120,
    )
    return r.json()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Local AI Chat")
    st.caption("Production-grade RAG + LoRA fine-tuning")
    st.divider()

    # API status
    online = api_online()
    if online:
        st.success("API Online", icon="✅")
    else:
        st.error("API Offline — start FastAPI first", icon="❌")
        st.code("uvicorn main:app --port 8000")

    st.divider()

    # Model toggle
    st.markdown("### 🔀 AI Model")
    active = get_active_model()

    col_b, col_f = st.columns(2)
    with col_b:
        if st.button(
            "🧠 Base",
            use_container_width=True,
            type="primary" if active == "base" else "secondary",
        ):
            switch_model("base")
            st.rerun()
    with col_f:
        if st.button(
            "⚡ Fine-Tuned",
            use_container_width=True,
            type="primary" if active == "finetuned" else "secondary",
        ):
            switch_model("finetuned")
            st.rerun()

    active = get_active_model()
    if active == "base":
        st.markdown('<span class="badge-base">🧠 Gemma 2b · RAG · PDF-grounded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-finetuned">⚡ GPT-2 + LoRA · 8× faster · ML expert</span>', unsafe_allow_html=True)

    st.divider()

    # Session stats
    st.markdown("### 📊 Session Stats")
    avg_lat = round(st.session_state.total_ms / max(st.session_state.total_q, 1), 1)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{st.session_state.total_q}</div><div class="stat-label">Questions</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{avg_lat}s</div><div class="stat-label">Avg Latency</div></div>', unsafe_allow_html=True)

    st.divider()

    # Monitoring links
    st.markdown("### 🔗 Monitoring")
    st.markdown("- [📈 Grafana Dashboard](http://localhost:3000/d/local-ai-chat)")
    st.markdown("- [🔬 Prometheus](http://localhost:9090)")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_q  = 0
        st.session_state.total_ms = 0.0
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## 💬 Chat with Your Document")

model_label = "🧠 Gemma 2b (Base — RAG)" if active == "base" else "⚡ GPT-2 + LoRA (Fine-Tuned)"
st.caption(f"Talking to: **{model_label}** · Ask about Naive Bayes, Logistic Regression, or any ML topic")

st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

        # Show metadata under AI messages
        if msg["role"] == "assistant":
            meta_cols = st.columns([1, 1, 1, 3])
            with meta_cols[0]:
                st.caption(f"⏱ {msg.get('latency','?')}s")
            with meta_cols[1]:
                badge = "🧠 Base" if msg.get("model") == "base" else "⚡ Fine-Tuned"
                st.caption(badge)

            # Citations / retrieval results
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"📄 View {len(sources)} retrieved source(s)", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.markdown(
                            f'<div class="source-box">'
                            f'<strong>Source {i}</strong> · Chunk #{src["chunk_id"]}<br>'
                            f'{src["text"][:300]}{"…" if len(src["text"]) > 300 else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

# ── Input tabs ────────────────────────────────────────────────────────────────
tab_text, tab_voice = st.tabs(["💬 Type a Question", "🎤 Ask by Voice"])

# ── Text input ────────────────────────────────────────────────────────────────
with tab_text:
    prompt = st.chat_input("Ask about your document or any ML topic…")
    if prompt and online:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                try:
                    data    = ask(prompt)
                    answer  = data.get("response", "No response.")
                    latency = data.get("latency", "?")
                    model   = data.get("model", "?")
                    sources = data.get("sources", [])
                except Exception as e:
                    answer, latency, model, sources = f"❌ Error: {e}", "?", "?", []

            st.markdown(answer)

            meta_cols = st.columns([1, 1, 1, 3])
            with meta_cols[0]: st.caption(f"⏱ {latency}s")
            with meta_cols[1]: st.caption("🧠 Base" if model == "base" else "⚡ Fine-Tuned")

            if sources:
                with st.expander(f"📄 View {len(sources)} retrieved source(s)", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.markdown(
                            f'<div class="source-box">'
                            f'<strong>Source {i}</strong> · Chunk #{src["chunk_id"]}<br>'
                            f'{src["text"][:300]}{"…" if len(src["text"]) > 300 else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        st.session_state.messages.append({
            "role": "assistant", "content": answer,
            "latency": latency, "model": model, "sources": sources,
        })
        st.session_state.total_q  += 1
        st.session_state.total_ms += float(latency) if latency != "?" else 0

# ── Voice input ───────────────────────────────────────────────────────────────
with tab_voice:
    st.markdown("#### 🎤 Speak your question")
    st.caption("Click the microphone, ask your question, then click Stop. Whisper tiny will transcribe it.")

    audio = st.audio_input("Record question")

    if audio and online:
        st.audio(audio)
        with st.spinner("Transcribing with Whisper tiny…"):
            try:
                data          = ask_voice(audio.getvalue())
                transcription = data.get("transcription", "")
                answer        = data.get("response", "No response.")
                latency       = data.get("latency", "?")
                model         = data.get("model", "?")
                sources       = data.get("sources", [])
            except Exception as e:
                transcription, answer, latency, model, sources = "", f"❌ Error: {e}", "?", "?", []

        if transcription:
            st.success(f"**You said:** {transcription}")
            st.divider()

            st.markdown("**🤖 AI Answer:**")
            st.markdown(answer)
            st.caption(f"⏱ {latency}s · {'🧠 Base' if model == 'base' else '⚡ Fine-Tuned'}")

            if sources:
                with st.expander(f"📄 View {len(sources)} retrieved source(s)"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(
                            f'<div class="source-box">'
                            f'<strong>Source {i}</strong> · Chunk #{src["chunk_id"]}<br>'
                            f'{src["text"][:300]}{"…" if len(src["text"]) > 300 else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            st.session_state.messages.append({"role": "user",      "content": f"🎤 {transcription}"})
            st.session_state.messages.append({"role": "assistant", "content": answer,
                                               "latency": latency, "model": model, "sources": sources})
            st.session_state.total_q  += 1
            st.session_state.total_ms += float(latency) if latency != "?" else 0
        else:
            st.warning("Nothing transcribed — try speaking louder or longer.")
