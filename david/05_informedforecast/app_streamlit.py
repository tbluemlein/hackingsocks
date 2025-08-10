
import os
import sys
import json
import re
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
import requests

HERE = Path(__file__).parent.resolve()
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

from pdf_rag_pipeline import PDFIngestor, RAGQuery, md5sum
try:
    from pdf_fin_table_extractor import FinancialTableExtractor
    HAS_TABLES = True
except Exception:
    HAS_TABLES = False

try:
    from market_forecast import (
        fetch_alpha_vantage_free,
        parse_horizon,
        exp_smoothing_forecast,
        garch_price_forecast
    )
    HAS_MARKET = True
except Exception:
    HAS_MARKET = False

st.set_page_config(page_title="Annual Report Q&A + Forecast + AI Strategy", layout="wide")
st.title("Annual Report Q&A + Forecast + AI Strategy")

# ---------------- Helpers (moved above sidebar so we can use them there) ----------------
def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "company"

def catalog_path(db_dir: str) -> Path:
    return Path(db_dir) / "kb_catalog.json"

def load_catalog(db_dir: str) -> dict:
    p = catalog_path(db_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}

def save_catalog(db_dir: str, cat: dict):
    p = catalog_path(db_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cat, indent=2))

def add_doc_to_catalog(db_dir: str, collection: str, entry: dict):
    cat = load_catalog(db_dir)
    cat.setdefault(collection, [])
    if not any(x.get("md5")==entry.get("md5") for x in cat[collection]):
        cat[collection].append(entry)
        save_catalog(db_dir, cat)

def list_reports(db_dir: str, collection: str):
    cat = load_catalog(db_dir)
    return cat.get(collection, [])

def get_collection_name(company: str, coll_prefix: str, pdf_path: Path, use_kb: bool) -> str:
    if use_kb and company.strip():
        return f"{coll_prefix}_{slugify(company)}"
    else:
        return f"{coll_prefix}_{md5sum(str(pdf_path))[:10]}"

# ---------------- Session State ----------------
for key, val in {
    "collection": None, "company": "", "report_date": "", "last_file_md5": None,
    "table_rows_indexed": 0, "query_dates": [], "active_catalog": {}, "active_collection_list": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    db_dir = st.text_input("Vector store directory", value=str(HERE / "vectorstore_app"))
    coll_prefix = st.text_input("Collection prefix", value="finance_docs")
    use_company_kb = st.checkbox("Use company knowledge base (append across years)", value=True)
    ollama_url = st.text_input("Ollama endpoint", value="http://localhost:11434/v1/chat/completions")
    model_name = st.text_input("Model name", value="llama3.3")
    k = st.slider("Top-K chunks", 3, 12, 6)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    do_ocr = st.checkbox("Enable OCR fallback (ocrmypdf + tesseract)", value=False)
    index_tables = st.checkbox("Also index table rows", value=True, disabled=not HAS_TABLES)

    st.markdown("---")
    st.subheader("Alpha Vantage (FREE endpoints)")
    disable_market = st.checkbox("Docs-only mode (disable Alpha Vantage)", value=False)
    default_key = os.getenv("ALPHAVANTAGE_API_KEY", "3B10HMD7VDM29PS9")
    av_api_key = st.text_input("API Key", value=default_key, type="password", disabled=disable_market)
    granularity = st.selectbox("Granularity", options=["daily", "weekly", "monthly"], index=0, disabled=disable_market)
    outputsize = st.selectbox("Output size (daily only)", options=["compact", "full"], index=0, disabled=(disable_market or granularity!='daily'))

    st.markdown("---")
    st.subheader("Existing knowledge bases")
    if st.button("Refresh catalog"):
        st.session_state.active_catalog = load_catalog(db_dir)
        st.session_state.active_collection_list = sorted(list(st.session_state.active_catalog.keys()))
    if not st.session_state.active_catalog:
        st.session_state.active_catalog = load_catalog(db_dir)
        st.session_state.active_collection_list = sorted(list(st.session_state.active_catalog.keys()))
    kb_options = st.session_state.active_collection_list or ["(none found)"]
    chosen_kb = st.selectbox("Activate an existing KB (collection)", options=kb_options)
    colA, colB = st.columns(2)
    with colA:
        if st.button("Activate KB", use_container_width=True, type="primary") and chosen_kb and chosen_kb != "(none found)":
            st.session_state.collection = chosen_kb
            # Try to infer a company name from the first entry
            entries = st.session_state.active_catalog.get(chosen_kb, [])
            st.session_state.company = (entries[0].get("company") if entries else st.session_state.company) or st.session_state.company
            st.success(f"Activated collection: {chosen_kb}")
    with colB:
        if st.button("Clear active KB", use_container_width=True):
            st.session_state.collection = None
            st.session_state.company = ""
            st.info("Cleared active KB selection.")

# Tabs
tab_ingest, tab_query, tab_market, tab_strategy = st.tabs(
    ["Upload & Ingest", "Document Q&A", "Market Data & Forecast", "Assisted Forecast (Docs + Market + LLM)"]
)

# ---------------- Upload & Ingest ----------------
with tab_ingest:
    st.subheader("1) Upload annual report PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        company = st.text_input("Company (ticker or name)", value=st.session_state.company or "")
    with col_meta2:
        report_date = st.text_input("Report date (YYYY-MM-DD)", value=st.session_state.report_date or "")

    def save_upload(file):
        target_dir = Path(db_dir) / "uploads"
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / file.name
        with open(out_path, "wb") as f:
            f.write(file.getbuffer())
        return out_path

    if st.button("Ingest PDF"):
        if not uploaded:
            st.warning("Please upload a PDF first."); st.stop()
        if not company.strip():
            st.warning("Please provide a company/ticker."); st.stop()
        saved = save_upload(uploaded)
        st.session_state.company = company.strip()
        st.session_state.report_date = report_date.strip() or "NA"
        st.session_state.last_file_md5 = md5sum(str(saved))
        st.session_state.collection = get_collection_name(st.session_state.company, coll_prefix, saved, use_company_kb)

        st.info(f"Ingesting into collection: **{st.session_state.collection}**")
        with st.status("Extracting, chunking, and embedding…", expanded=True) as status:
            try:
                ingestor = PDFIngestor(db_dir=db_dir, collection=st.session_state.collection, embedding_model="BAAI/bge-small-en-v1.5", do_ocr=do_ocr)
                chunks = ingestor.ingest_pdf(str(saved), company=st.session_state.company, report_date=st.session_state.report_date, doc_type="annual_report")
                st.write(f"• Text chunks indexed: **{len(chunks)}**")
                add_doc_to_catalog(db_dir, st.session_state.collection, {
                    "company": st.session_state.company,
                    "report_date": st.session_state.report_date,
                    "doc_type": "annual_report",
                    "source_path": str(saved),
                    "md5": st.session_state.last_file_md5,
                    "chunks": len(chunks),
                })
                # refresh catalog in sidebar session after ingest
                st.session_state.active_catalog = load_catalog(db_dir)
                st.session_state.active_collection_list = sorted(list(st.session_state.active_catalog.keys()))
            except Exception as e:
                st.error(f"Ingestion error: {e}"); st.stop()

            st.session_state.table_rows_indexed = 0
            if index_tables and HAS_TABLES:
                try:
                    from pdf_rag_pipeline import VectorStore
                    fte = FinancialTableExtractor()
                    store = VectorStore(db_dir=db_dir, collection=st.session_state.collection)
                    table_chunks = fte.index_into_vectorstore(store, pdf_path=str(saved), company=st.session_state.company, report_date=st.session_state.report_date, doc_type="table")
                    st.session_state.table_rows_indexed = len(table_chunks)
                    cat = load_catalog(db_dir)
                    for e in cat.get(st.session_state.collection, []):
                        if e.get("md5")==st.session_state.last_file_md5:
                            e["table_rows"] = st.session_state.table_rows_indexed
                    save_catalog(db_dir, cat)
                    st.write(f"• Table rows indexed: **{len(table_chunks)}**")
                except Exception as e:
                    st.warning(f"Table indexing skipped: {e}")
            status.update(state="complete", label="Ingestion complete")

    if st.session_state.collection:
        rep = list_reports(db_dir, st.session_state.collection)
        if rep:
            st.markdown("### Company knowledge base")
            df_rep = pd.DataFrame(rep)
            st.dataframe(df_rep, use_container_width=True, height=240)

# ---------------- Document Q&A ----------------
with tab_query:
    st.subheader("2) Ask a question about the uploaded report(s)")
    if not st.session_state.collection:
        st.info("No active knowledge base selected. Choose one in the **sidebar** under 'Existing knowledge bases', or ingest a PDF in the first tab.")
    date_opts = []
    if st.session_state.collection:
        rep = list_reports(db_dir, st.session_state.collection)
        date_opts = sorted({r.get("report_date") for r in rep if r.get("report_date")})
    selected_dates = st.multiselect("Restrict to report dates (optional)", options=date_opts, default=date_opts)

    q = st.text_input("Your question", placeholder="e.g., Compare revenue growth 2023 vs 2024; what were key risks?")
    if st.button("Ask"):
        if not st.session_state.collection:
            st.warning("Please activate a knowledge base or ingest at least one PDF first."); st.stop()
        try:
            rag = RAGQuery(db_dir=db_dir, collection=st.session_state.collection, embedding_model="BAAI/bge-small-en-v1.5", ollama_url=ollama_url, model_name=model_name)
            where = None
            if selected_dates:
                where = {"report_date": {"$in": selected_dates}}
            with st.spinner("Retrieving and querying model…"):
                res = rag.answer(q, k=int(k), temperature=float(temperature), where=where)
        except Exception as e:
            st.error(f"Query error: {e}"); st.stop()

        st.markdown("### Answer")
        st.write(res.get("answer", ""))
        metas = res.get("citations", []) or []
        if metas:
            st.markdown("### Citations")
            for i, m in enumerate(metas, start=1):
                tag = f"{m.get('company','?')} | {m.get('report_date','?')} | p.{m.get('start_page','?')}-{m.get('end_page','?')}"
                with st.expander(f"Source {i}: {tag}"):
                    st.json(m, expanded=False)

# ---------------- Market Data & Forecast ----------------
with tab_market:
    st.subheader("3) Market Data & Forecast")
    if disable_market:
        st.info("Docs-only mode is ON. Market data fetching is disabled.")
    elif not HAS_MARKET:
        st.warning("market_forecast.py not found or dependencies missing. Place it next to this app and install: requests pandas numpy statsmodels plotly.")
    else:
        colA, colB = st.columns([2,1])
        with colA:
            ticker = st.text_input("Ticker (e.g., MMM, AAPL)", value=(st.session_state.company or "AAPL"), key="mdf_ticker")
        with colB:
            nl_prompt = st.text_input("Forecast request (optional, e.g., 'next 6 weeks')", value="next 4 weeks", key="mdf_prompt")

        periods_default = parse_horizon(nl_prompt or "", default_days=20)
        periods = st.slider("Forecast horizon (business days)", 5, 120, periods_default, step=5, key="mdf_periods")

        run_md = st.button("Fetch & Plot", key="btn_fetch_plot")
        run_fc = st.button("Forecast", key="btn_forecast")

        @st.cache_data(ttl=3600, show_spinner=False)
        def cached_av(symbol: str, key: str, gran: str, size: str):
            return fetch_alpha_vantage_free(symbol, key, granularity=gran, outputsize=size)

        if run_md:
            try:
                with st.spinner("Fetching Alpha Vantage… (free endpoint)"):
                    dfp = cached_av(st.session_state.get("mdf_ticker"), av_api_key, granularity, outputsize)
                st.markdown(f"**{st.session_state.get('mdf_ticker')}** price history ({len(dfp)} rows) • granularity: **{granularity}**")
                fig = px.line(dfp, x="date", y="close", title=f"{st.session_state.get('mdf_ticker')} Close ({granularity})")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(dfp.tail(10), use_container_width=True, height=220)
                st.session_state["last_market_df"] = dfp
            except Exception as e:
                st.error(f"Alpha Vantage error: {e}")

        if run_fc:
            try:
                dfp = st.session_state.get("last_market_df")
                if dfp is None:
                    dfp = cached_av(st.session_state.get("mdf_ticker"), av_api_key, granularity, outputsize)
                with st.spinner("Fitting Exponential Smoothing…"):
                    #fc = exp_smoothing_forecast(dfp, value_col="close", periods=int(st.session_state.get("mdf_periods")))
                    fc = garch_price_forecast(dfp, value_col="close", periods=int(st.session_state.get("mdf_periods")))
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=dfp["date"], y=dfp["close"], mode="lines", name="History"))
                fig2.add_trace(go.Scatter(x=fc["date"], y=fc["yhat"], mode="lines", name="Forecast"))
                fig2.add_trace(go.Scatter(
                    x=pd.concat([fc["date"], fc["date"][::-1]]),
                    y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
                    fill="toself", mode="lines", line=dict(width=0),
                    name="95% interval", opacity=0.2
                ))
                fig2.update_layout(title=f"{st.session_state.get('mdf_ticker')} Forecast ({st.session_state.get('mdf_periods')} business days)")
                st.plotly_chart(fig2, use_container_width=True)
                st.dataframe(fc, use_container_width=True, height=260)
                st.session_state["last_forecast_df"] = fc
            except Exception as e:
                st.error(f"Forecast error: {e}")

# ---------------- Assisted Forecast (Docs + Market + LLM) ----------------
with tab_strategy:
    st.subheader("4) Assisted Forecast (combines document insights + market data + LLM)")
    if not disable_market and not HAS_MARKET:
        st.warning("market_forecast.py not found or dependencies missing. Enable docs-only mode in the sidebar to proceed without market data.")
    else:
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            strat_ticker = st.text_input("Ticker", value=(st.session_state.company or "AAPL"), key="strat_ticker")
            conditioning = st.text_area("Strategy conditioning prompt", value="Focus on management guidance, segment trends, and risk disclosures. Incorporate recent revenue/earnings direction from the report.", height=120)
        with col2:
            nl_h = st.text_input("Forecast horizon (e.g., 'next 6 weeks')", value="next 6 weeks", key="strat_nl")
            periods = parse_horizon(nl_h or "", default_days=30) if not disable_market else 0
            if disable_market:
                st.metric("Docs-only mode", "ON")
            else:
                st.metric("Parsed horizon (business days)", periods)
        with col3:
            topk_docs = st.slider("Top-K doc chunks", 3, 12, int(k), key="strat_k")
            show_pi = st.checkbox("Show 95% forecast band", value=True, disabled=disable_market)

        rep = list_reports(db_dir, st.session_state.collection) if st.session_state.collection else []
        date_opts = sorted({r.get("report_date") for r in rep if r.get("report_date")})
        selected_dates = st.multiselect("Restrict doc context to report dates", options=date_opts, default=date_opts, key="strat_dates")

        if "strategy_chat" not in st.session_state:
            st.session_state.strategy_chat = []
        if "strategy_sources" not in st.session_state:
            st.session_state.strategy_sources = []
        if "strategy_last_fc" not in st.session_state:
            st.session_state.strategy_last_fc = None
        if "strategy_last_md" not in st.session_state:
            st.session_state.strategy_last_md = None

        def fetch_prices(symbol):
            return fetch_alpha_vantage_free(symbol, av_api_key, granularity="daily", outputsize="compact")

        def base_forecast(df, periods):
            return garch_price_forecast(df, value_col="close", periods=int(periods))

        def retrieve_docs(query_text, kdocs, where):
            if not st.session_state.collection:
                return {"documents":[[""]], "metadatas":[[]]}
            try:
                rag_tmp = RAGQuery(db_dir=db_dir, collection=st.session_state.collection, embedding_model="BAAI/bge-small-en-v1.5", ollama_url=ollama_url, model_name=model_name)
                res = rag_tmp.store.query(query_text, kdocs, where=where)
                return res
            except Exception as e:
                st.warning(f"Doc retrieval failed: {e}")
                return {"documents":[[""]], "metadatas":[[]]}

        def call_ollama(messages, temperature=0.1):
            payload = {"model": model_name, "messages": messages, "temperature": temperature, "stream": False}
            r = requests.post(ollama_url, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"].get("content", "")
            return str(data)[:2000]

        run_btn = st.button("Run Assisted Forecast")
        followup = st.text_input("Follow-up question (optional)")
        ask_followup = st.button("Ask Follow-up")

        if run_btn:
            try:
                where = {"report_date": {"$in": selected_dates}} if selected_dates else None
                doc_query = conditioning + " outlook guidance risk revenue earnings cost headwinds tailwinds capex margins segments"
                res = retrieve_docs(doc_query, int(topk_docs), where)
                docs = res["documents"][0] if res.get("documents") else []
                metas = res["metadatas"][0] if res.get("metadatas") else []
                st.session_state.strategy_sources = metas

                doc_blurbs = []
                for d,m in zip(docs, metas):
                    tag = f"{m.get('company','?')} | {m.get('report_date','?')} | p.{m.get('start_page','?')}-{m.get('end_page','?')}"
                    doc_blurbs.append(f"[{tag}] {d[:800].replace(chr(10),' ')}")

                system_msg = {"role": "system", "content": "You are a disciplined equity analyst. Ground your analysis ONLY in the cited document snippets. Do not invent figures. If the context is weak, say so. Provide a clear Buy/Hold/Sell with rationale and note uncertainties. Keep under 200 words. Return sections: Summary, Recommendation, Why, Risks."}

                if disable_market:
                    market_ctx = "Market data disabled by user. Provide a qualitative outlook based solely on the documents and highlight what market signals you would next seek to validate this view."
                    user_msg = {"role": "user", "content": (
                        f"Ticker: {st.session_state.get('strat_ticker')}\n"
                        f"Mode: DOCS-ONLY\n"
                        f"Conditioning prompt: {conditioning}\n\n"
                        f"{market_ctx}\n\n"
                        "Document snippets:\n" + "\n\n".join(doc_blurbs)
                    )}
                    st.session_state.strategy_chat = [system_msg, user_msg]
                else:
                    with st.spinner("Fetching prices & computing base forecast…"):
                        dfp = fetch_prices(st.session_state.get("strat_ticker"))
                        fc = base_forecast(dfp, periods)
                        st.session_state.strategy_last_md = dfp
                        st.session_state.strategy_last_fc = fc
                    last_close = float(dfp["close"].iloc[-1])
                    end_fc = float(fc["yhat"].iloc[-1])
                    pct = (end_fc / last_close - 1.0) * 100.0
                    width = float((fc["yhat_upper"].iloc[-1] - fc["yhat_lower"].iloc[-1]))
                    vol20 = float(dfp["close"].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252))
                    market_ctx = f"Base forecast: last_close={last_close:.2f}, end_forecast={end_fc:.2f}, change_to_end={pct:.2f}%, interval_width≈{width:.2f}, annualized_vol20≈{vol20:.2f}."
                    user_msg = {"role": "user", "content": (
                        f"Ticker: {st.session_state.get('strat_ticker')}\n"
                        f"Horizon: {periods} business days\n"
                        f"Conditioning prompt: {conditioning}\n\n"
                        f"{market_ctx}\n\n"
                        "Document snippets:\n" + "\n\n".join(doc_blurbs)
                    )}
                    st.session_state.strategy_chat = [system_msg, user_msg]

                with st.spinner("Reasoning with available context…"):
                    answer = call_ollama(st.session_state.strategy_chat, temperature=float(temperature))

                st.markdown("### Assisted Forecast — Model View")
                st.write(answer)

                if not disable_market:
                    fig3 = go.Figure()
                    dfp = st.session_state.strategy_last_md
                    fc = st.session_state.strategy_last_fc
                    fig3.add_trace(go.Scatter(x=dfp['date'], y=dfp['close'], mode='lines', name='History'))
                    fig3.add_trace(go.Scatter(x=fc['date'], y=fc['yhat'], mode='lines', name='Forecast'))
                    if show_pi:
                        fig3.add_trace(go.Scatter(
                            x=pd.concat([fc["date"], fc["date"][::-1]]),
                            y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
                            fill="toself", mode="lines", line=dict(width=0),
                            name="95% interval", opacity=0.2
                        ))
                    fig3.update_layout(title=f"{st.session_state.get('strat_ticker')} — Assisted Forecast Horizon ({periods}B)")
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("Docs-only mode: no price/forecast chart displayed.")

                if metas:
                    st.markdown("### Citations")
                    for i, m in enumerate(metas, start=1):
                        tag = f"{m.get('company','?')} | {m.get('report_date','?')} | p.{m.get('start_page','?')}-{m.get('end_page','?')}"
                        with st.expander(f"Source {i}: {tag}"):
                            st.json(m, expanded=False)

            except Exception as e:
                st.error(f"Assisted forecast error: {e}")

        if ask_followup:
            if not st.session_state.strategy_chat:
                st.warning("Run an Assisted Forecast first.")
            else:
                extra_where = {"report_date": {"$in": selected_dates}} if selected_dates else None
                try:
                    rag_tmp = RAGQuery(db_dir=db_dir, collection=st.session_state.collection, embedding_model="BAAI/bge-small-en-v1.5", ollama_url=ollama_url, model_name=model_name)
                    res_extra = rag_tmp.store.query(followup, int(topk_docs), where=extra_where)
                    docs2 = res_extra["documents"][0] if res_extra.get("documents") else []
                except Exception:
                    docs2 = []
                ctx_update = ""
                if docs2:
                    ctx_update = "\\n\\nAdditional context: " + " | ".join([d[:200].replace("\\n"," ") for d in docs2[:2]])
                st.session_state.strategy_chat.append({"role":"user", "content": followup + ctx_update})
                try:
                    with st.spinner("Thinking…"):
                        answer2 = call_ollama(st.session_state.strategy_chat, temperature=float(temperature))
                    st.markdown("### Follow-up")
                    st.write(answer2)
                except Exception as e:
                    st.error(f"Follow-up error: {e}")
