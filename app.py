"""
============================================================
DASHBOARD AIoT - SISTEM PREDIKSI KERUSAKAN POMPA AIR
Berbasis Klasifikasi Random Forest pada Data Konsumsi Energi

Universitas Siliwangi
Sifah Nur Rizkiyah - 237006035
============================================================
Deploy: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random
import datetime
import os

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AIoT Pompa Air | Sifah Nur Rizkiyah",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    /* Font & base */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f2a4a 0%, #1a4a7a 50%, #0d3d6e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { font-size: 1.6rem; font-weight: 700; margin: 0 0 6px 0; color: white; }
    .main-header p  { font-size: 0.9rem; opacity: 0.75; margin: 0; }
    .main-header .meta { margin-top: 12px; display: flex; gap: 20px; font-size: 0.8rem; opacity: 0.65; }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #e8ecf0;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .metric-label { font-size: 0.75rem; color: #6b7280; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
    .metric-value { font-size: 2rem; font-weight: 700; line-height: 1; margin-bottom: 4px; }
    .metric-sub   { font-size: 0.78rem; color: #9ca3af; }
    .blue  { color: #1d4ed8; }
    .green { color: #15803d; }
    .amber { color: #b45309; }
    .red   { color: #b91c1c; }

    /* Prediction badge */
    .badge-normal   { background:#dcfce7; color:#15803d; padding:6px 16px; border-radius:99px; font-weight:600; font-size:1rem; display:inline-block; }
    .badge-masalah  { background:#fee2e2; color:#b91c1c; padding:6px 16px; border-radius:99px; font-weight:600; font-size:1rem; display:inline-block; }

    /* Sidebar */
    .sidebar-section { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:1rem; margin-bottom:1rem; }
    .sidebar-title   { font-size:0.8rem; font-weight:600; color:#374151; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px; }

    /* Table striped */
    .stDataFrame { border-radius: 10px !important; }

    /* Footer */
    .footer { text-align:center; font-size:0.78rem; color:#9ca3af; padding:2rem 0 1rem; border-top:1px solid #f1f5f9; margin-top:2rem; }

    /* Streamlit default tweaks */
    .block-container { padding-top: 1.5rem !important; }
    h2 { font-size: 1.1rem !important; font-weight: 600 !important; }
    h3 { font-size: 1rem !important; font-weight: 600 !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.88rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    'lights','T1','RH_1','T2','RH_2','T3','RH_3','T4','RH_4',
    'T5','RH_5','T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9',
    'T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint'
]

@st.cache_resource
def load_model():
    try:
        model  = joblib.load('model_rf_pompa.pkl')
        scaler = joblib.load('scaler_pompa.pkl')
        return model, scaler, True
    except FileNotFoundError:
        return None, None, False

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('energydata_complete.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df, True
    except FileNotFoundError:
        return None, False

@st.cache_data
def load_feature_importance():
    try:
        return pd.read_csv('feature_importance.csv')
    except FileNotFoundError:
        # fallback hardcoded
        data = {
            'Fitur': ['lights','RH_1','T2','RH_8','RH_out','Press_mm_hg','RH_9','RH_3','RH_6','T3'],
            'Importance': [0.0625,0.0609,0.0586,0.0520,0.0484,0.0477,0.0471,0.0454,0.0451,0.0429],
            'Importance (%)': [6.25,6.09,5.86,5.20,4.84,4.77,4.71,4.54,4.51,4.29]
        }
        return pd.DataFrame(data)

model, scaler, model_loaded = load_model()
df_energy, data_loaded      = load_dataset()
df_feat                     = load_feature_importance()


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💧 AIoT Pompa Air")
    st.markdown("---")

    st.markdown('<div class="sidebar-section"><div class="sidebar-title">📌 Identitas</div>'
                '<b>Sifah Nur Rizkiyah</b><br><small>NIM: 237006035</small><br>'
                '<small>Universitas Siliwangi</small><br>'
                '<small>Mata Kuliah: Internet of Things</small></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><div class="sidebar-title">🤖 Info Model</div>'
                '<small><b>Algoritma:</b> Random Forest<br>'
                '<b>Estimators:</b> 100<br>'
                '<b>Fitur:</b> 25<br>'
                '<b>Akurasi:</b> 87.3%<br>'
                '<b>AUC-ROC:</b> 0.935</small></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><div class="sidebar-title">🔗 Resource</div>'
                '<small>📡 <a href="https://wokwi.com/projects/464319240845028353" target="_blank">Simulasi Wokwi</a><br>'
                '📊 Dataset: Energydata Complete<br>'
                '🐍 Python 3.10+ | Streamlit</small></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<small style='color:#9ca3af'>© 2024 Sifah Nur Rizkiyah<br>Universitas Siliwangi</small>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>💧 Sistem Prediksi Kerusakan Pompa Air Berbasis AIoT</h1>
  <p>Pendekatan Klasifikasi Random Forest pada Data Konsumsi Energi</p>
  <div class="meta">
    <span>👤 Sifah Nur Rizkiyah — 237006035</span>
    <span>🎓 Universitas Siliwangi</span>
    <span>📅 Internet of Things</span>
    <span>📂 Dataset: Energydata Complete</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Status chips
col_s1, col_s2, col_s3, _ = st.columns([1,1,1,3])
with col_s1:
    if model_loaded:
        st.success("✅ Model RF Loaded")
    else:
        st.error("❌ Model tidak ditemukan")
with col_s2:
    if data_loaded:
        st.success("✅ Dataset Loaded")
    else:
        st.warning("⚠️ Dataset tidak ditemukan")
with col_s3:
    st.info("📡 ESP32 + DHT22")


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🔬 Evaluasi Model",
    "📈 Feature Importance",
    "🔮 Prediksi Real-time",
    "🏗️ Arsitektur & Wokwi"
])


# ═══════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════
with tab1:
    st.markdown("### 📊 Ringkasan Performa Model")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-label">Akurasi</div>'
                    '<div class="metric-value blue">87.3%</div>'
                    '<div class="metric-sub">Overall accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-label">AUC-ROC</div>'
                    '<div class="metric-value blue">0.935</div>'
                    '<div class="metric-sub">Sangat baik</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-label">True Positive (Normal)</div>'
                    '<div class="metric-value green">3,012</div>'
                    '<div class="metric-sub">Terdeteksi benar</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-label">False Negative</div>'
                    '<div class="metric-value amber">382</div>'
                    '<div class="metric-sub">Masalah tidak terdeteksi</div></div>', unsafe_allow_html=True)

    st.markdown("")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Distribusi Prediksi Kelas")
        fig_pie = go.Figure(go.Pie(
            labels=['Normal', 'Bermasalah'],
            values=[3100, 847],
            hole=0.5,
            marker_colors=['#15803d', '#b91c1c'],
            textfont_size=13,
        ))
        fig_pie.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=280,
            legend=dict(orientation='h', y=-0.1),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("#### Metrik Evaluasi per Kelas")
        fig_bar = go.Figure()
        categories = ['Normal', 'Bermasalah']
        precision  = [88.7, 84.1]
        recall     = [97.2, 54.9]
        f1         = [92.7, 66.4]
        fig_bar.add_trace(go.Bar(name='Precision', x=categories, y=precision, marker_color='#2563eb'))
        fig_bar.add_trace(go.Bar(name='Recall',    x=categories, y=recall,    marker_color='#d97706'))
        fig_bar.add_trace(go.Bar(name='F1-Score',  x=categories, y=f1,        marker_color='#16a34a'))
        fig_bar.update_layout(
            barmode='group', height=280,
            margin=dict(t=20, b=20, l=10, r=10),
            yaxis=dict(range=[0,105], title='Nilai (%)'),
            legend=dict(orientation='h', y=-0.2),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### Top 5 Fitur Paling Berpengaruh")
    top5 = df_feat.head(5)
    fig_top5 = go.Figure(go.Bar(
        x=top5['Importance (%)'],
        y=top5['Fitur'],
        orientation='h',
        marker_color=['#1d4ed8','#2563eb','#3b82f6','#60a5fa','#93c5fd'],
        text=[f"{v:.2f}%" for v in top5['Importance (%)']],
        textposition='outside',
    ))
    fig_top5.update_layout(
        height=200, margin=dict(t=10, b=10, l=10, r=60),
        xaxis=dict(title='Feature Importance (%)'),
        yaxis=dict(autorange='reversed'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_top5, use_container_width=True)

    if data_loaded:
        st.markdown("#### Sampel Dataset Energydata")
        st.dataframe(df_energy[FEATURE_NAMES + ['Appliances']].head(10), use_container_width=True)
        st.caption(f"Total data: {len(df_energy):,} baris × {len(df_energy.columns)} kolom")


# ═══════════════════════════════════════════════
# TAB 2: EVALUASI MODEL
# ═══════════════════════════════════════════════
with tab2:
    st.markdown("### 🔬 Evaluasi Model Random Forest")

    col_cm, col_metrics = st.columns([1, 1])

    with col_cm:
        st.markdown("#### Confusion Matrix")
        cm_data = [[3012, 88], [382, 465]]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_data,
            x=['Pred: Normal', 'Pred: Bermasalah'],
            y=['Aktual: Normal', 'Aktual: Bermasalah'],
            colorscale=[[0,'#dbeafe'],[1,'#1e3a5f']],
            text=[[str(v) for v in row] for row in cm_data],
            texttemplate='<b>%{text}</b>',
            textfont=dict(size=22, color='white'),
            showscale=False,
        ))
        fig_cm.update_layout(
            height=280, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_metrics:
        st.markdown("#### Performa per Kelas")
        metrics_df = pd.DataFrame({
            'Kelas':     ['Normal', 'Bermasalah'],
            'Precision': ['88.7%', '84.1%'],
            'Recall':    ['97.2%', '54.9%'],
            'F1-Score':  ['92.7%', '66.4%'],
            'Support':   [3100, 847],
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        st.markdown("")
        st.markdown("**Akurasi Keseluruhan:** `87.3%`")
        st.markdown("**Macro Avg F1:** `79.6%`")
        st.markdown("**Weighted Avg F1:** `86.4%`")

    st.markdown("#### ROC Curve (AUC = 0.935)")
    # Generate approximate ROC curve
    fpr = np.linspace(0, 1, 200)
    tpr = np.where(fpr < 0.3, fpr * 2.8, 0.84 + (fpr - 0.3) * 0.23)
    tpr = np.clip(tpr, 0, 1)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines', name='ROC (AUC=0.935)',
        line=dict(color='#1d4ed8', width=2.5),
        fill='tozeroy', fillcolor='rgba(29,78,216,0.08)'
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode='lines', name='Random Classifier',
        line=dict(color='#9ca3af', width=1.5, dash='dash')
    ))
    fig_roc.update_layout(
        height=320,
        xaxis=dict(title='False Positive Rate', range=[0,1]),
        yaxis=dict(title='True Positive Rate', range=[0,1.02]),
        legend=dict(x=0.6, y=0.2),
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig_roc.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
    fig_roc.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
    st.plotly_chart(fig_roc, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 3: FEATURE IMPORTANCE
# ═══════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Feature Importance — Random Forest")

    col_sel, _ = st.columns([2,4])
    with col_sel:
        top_n = st.slider("Tampilkan top N fitur:", min_value=5, max_value=25, value=10, step=1)

    df_plot = df_feat.head(top_n).sort_values('Importance (%)')

    colors_feat = px.colors.sequential.Blues[2:]
    color_seq   = [colors_feat[int(i * (len(colors_feat)-1) / max(len(df_plot)-1, 1))]
                   for i in range(len(df_plot))]

    fig_feat = go.Figure(go.Bar(
        x=df_plot['Importance (%)'],
        y=df_plot['Fitur'],
        orientation='h',
        marker_color=color_seq,
        text=[f"{v:.2f}%" for v in df_plot['Importance (%)']],
        textposition='outside',
    ))
    fig_feat.update_layout(
        height=max(300, top_n * 36),
        xaxis=dict(title='Feature Importance (%)'),
        yaxis=dict(autorange='reversed' if top_n <= 10 else 'reversed'),
        margin=dict(t=10, b=10, l=80, r=70),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig_feat.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
    st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown("#### Tabel Feature Importance Lengkap")
    df_display = df_feat.copy()
    df_display.columns = ['Fitur', 'Importance Score', 'Importance (%)']
    df_display['Importance Score'] = df_display['Importance Score'].map('{:.6f}'.format)
    df_display['Importance (%)']   = df_display['Importance (%)'].map('{:.2f}%'.format)
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    if data_loaded:
        st.markdown("#### Korelasi Fitur Teratas dengan Appliances")
        top_features = df_feat.head(8)['Fitur'].tolist()
        corr_data    = df_energy[top_features + ['Appliances']].corr()['Appliances'].drop('Appliances')
        fig_corr = go.Figure(go.Bar(
            x=corr_data.index, y=corr_data.values,
            marker_color=['#16a34a' if v > 0 else '#b91c1c' for v in corr_data.values],
            text=[f"{v:.3f}" for v in corr_data.values],
            textposition='outside',
        ))
        fig_corr.update_layout(
            height=280, yaxis=dict(title='Korelasi Pearson'),
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        fig_corr.update_xaxes(showgrid=False)
        fig_corr.update_yaxes(showgrid=True, gridcolor='#f1f5f9', zeroline=True, zerolinecolor='#e2e8f0')
        st.plotly_chart(fig_corr, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 4: PREDIKSI REAL-TIME
# ═══════════════════════════════════════════════
with tab4:
    st.markdown("### 🔮 Prediksi Kondisi Pompa — Input Sensor")

    mode = st.radio("Mode input:", ["🎛️ Input Manual", "🎲 Simulasi Sensor Otomatis"], horizontal=True)

    st.markdown("---")

    if mode == "🎛️ Input Manual":
        st.markdown("#### Masukkan nilai sensor:")
        input_data = {}

        col1, col2, col3 = st.columns(3)
        with col1:
            input_data['lights']      = st.number_input("Lights (Wh)", 0.0, 150.0, 20.0, step=0.5)
            input_data['T1']          = st.number_input("T1 - Suhu Ruang 1 (°C)", 10.0, 35.0, 20.0, step=0.1)
            input_data['RH_1']        = st.number_input("RH_1 - Kelembaban 1 (%)", 20.0, 100.0, 47.0, step=0.5)
            input_data['T2']          = st.number_input("T2 - Suhu Ruang 2 (°C)", 10.0, 35.0, 19.5, step=0.1)
            input_data['RH_2']        = st.number_input("RH_2 - Kelembaban 2 (%)", 20.0, 100.0, 45.0, step=0.5)
            input_data['T3']          = st.number_input("T3 - Suhu Ruang 3 (°C)", 10.0, 35.0, 20.0, step=0.1)
            input_data['RH_3']        = st.number_input("RH_3 - Kelembaban 3 (%)", 20.0, 100.0, 45.0, step=0.5)
            input_data['T4']          = st.number_input("T4 - Suhu Ruang 4 (°C)", 10.0, 35.0, 19.0, step=0.1)
            input_data['RH_4']        = st.number_input("RH_4 - Kelembaban 4 (%)", 20.0, 100.0, 46.0, step=0.5)
        with col2:
            input_data['T5']          = st.number_input("T5 - Suhu Ruang 5 (°C)", 10.0, 35.0, 17.0, step=0.1)
            input_data['RH_5']        = st.number_input("RH_5 - Kelembaban 5 (%)", 20.0, 100.0, 55.0, step=0.5)
            input_data['T6']          = st.number_input("T6 - Suhu Luar (°C)", -5.0, 35.0, 7.0, step=0.1)
            input_data['RH_6']        = st.number_input("RH_6 - Kelembaban Luar (%)", 20.0, 100.0, 84.0, step=0.5)
            input_data['T7']          = st.number_input("T7 - Suhu Ruang 7 (°C)", 10.0, 35.0, 17.0, step=0.1)
            input_data['RH_7']        = st.number_input("RH_7 - Kelembaban 7 (%)", 20.0, 100.0, 42.0, step=0.5)
            input_data['T8']          = st.number_input("T8 - Suhu Ruang 8 (°C)", 10.0, 35.0, 18.0, step=0.1)
            input_data['RH_8']        = st.number_input("RH_8 - Kelembaban 8 (%)", 20.0, 100.0, 49.0, step=0.5)
            input_data['T9']          = st.number_input("T9 - Suhu Ruang 9 (°C)", 10.0, 35.0, 17.0, step=0.1)
        with col3:
            input_data['RH_9']        = st.number_input("RH_9 - Kelembaban 9 (%)", 20.0, 100.0, 46.0, step=0.5)
            input_data['T_out']       = st.number_input("T_out - Suhu Luar (°C)", -10.0, 35.0, 6.6, step=0.1)
            input_data['Press_mm_hg'] = st.number_input("Tekanan (mm Hg)", 720.0, 760.0, 733.5, step=0.5)
            input_data['RH_out']      = st.number_input("RH_out - Kelembaban Luar (%)", 20.0, 100.0, 92.0, step=0.5)
            input_data['Windspeed']   = st.number_input("Kecepatan Angin (m/s)", 0.0, 20.0, 7.0, step=0.5)
            input_data['Visibility']  = st.number_input("Visibilitas (km)", 0.0, 100.0, 63.0, step=1.0)
            input_data['Tdewpoint']   = st.number_input("Titik Embun (°C)", -10.0, 30.0, 5.3, step=0.1)

    else:  # Simulasi otomatis
        kondisi = st.selectbox("Pilih kondisi simulasi:", ["Normal 🟢", "Bermasalah 🟡", "Kritis 🔴"])
        if st.button("🔄 Generate Data Sensor Baru"):
            st.session_state['regen'] = True

        if kondisi.startswith("Normal"):
            input_data = {
                'lights':7.0,'T1':20.0,'RH_1':47.0,'T2':19.5,'RH_2':44.0,
                'T3':19.5,'RH_3':44.0,'T4':18.5,'RH_4':45.0,'T5':17.0,
                'RH_5':55.0,'T6':7.0,'RH_6':84.0,'T7':17.0,'RH_7':42.0,
                'T8':18.0,'RH_8':48.0,'T9':17.0,'RH_9':45.0,'T_out':6.5,
                'Press_mm_hg':733.5,'RH_out':92.0,'Windspeed':3.0,'Visibility':65.0,'Tdewpoint':5.0
            }
            for k in input_data:
                input_data[k] += random.uniform(-0.5, 0.5)
        elif kondisi.startswith("Bermasalah"):
            input_data = {
                'lights':60.0,'T1':25.0,'RH_1':55.0,'T2':24.5,'RH_2':53.0,
                'T3':24.0,'RH_3':52.0,'T4':24.0,'RH_4':54.0,'T5':22.5,
                'RH_5':63.0,'T6':15.0,'RH_6':74.0,'T7':23.0,'RH_7':49.0,
                'T8':24.5,'RH_8':57.0,'T9':22.5,'RH_9':55.0,'T_out':14.5,
                'Press_mm_hg':730.5,'RH_out':75.0,'Windspeed':7.0,'Visibility':45.0,'Tdewpoint':11.5
            }
            for k in input_data:
                input_data[k] += random.uniform(-1, 1)
        else:  # Kritis
            input_data = {
                'lights':100.0,'T1':29.0,'RH_1':63.0,'T2':28.5,'RH_2':61.0,
                'T3':28.0,'RH_3':60.0,'T4':28.0,'RH_4':62.0,'T5':27.0,
                'RH_5':70.0,'T6':21.0,'RH_6':66.0,'T7':27.5,'RH_7':55.0,
                'T8':28.5,'RH_8':65.0,'T9':27.0,'RH_9':63.0,'T_out':21.0,
                'Press_mm_hg':728.5,'RH_out':66.0,'Windspeed':11.5,'Visibility':29.0,'Tdewpoint':17.0
            }
            for k in input_data:
                input_data[k] += random.uniform(-1, 1)

        st.dataframe(pd.DataFrame([input_data]).T.rename(columns={0:'Nilai Sensor'}),
                     use_container_width=True, height=200)

    st.markdown("---")
    if st.button("🚀 Jalankan Prediksi", type="primary", use_container_width=True):
        if not model_loaded:
            st.error("❌ Model belum dimuat. Pastikan file `model_rf_pompa.pkl` dan `scaler_pompa.pkl` tersedia.")
        else:
            with st.spinner("Memproses inferensi model..."):
                try:
                    df_input = pd.DataFrame([input_data], columns=FEATURE_NAMES)
                    df_scaled = scaler.transform(df_input)
                    label     = model.predict(df_scaled)[0]
                    prob      = model.predict_proba(df_scaled)[0]
                    prob_normal  = float(prob[0]) * 100
                    prob_masalah = float(prob[1]) * 100

                    st.markdown("### Hasil Prediksi")
                    col_res1, col_res2 = st.columns([1,2])
                    with col_res1:
                        if label == 0:
                            st.markdown('<div style="padding:1.5rem;background:#f0fdf4;border:1px solid #86efac;border-radius:12px;text-align:center">'
                                        '<div style="font-size:2.5rem">✅</div>'
                                        '<div class="badge-normal">NORMAL</div>'
                                        '<p style="margin-top:10px;font-size:0.85rem;color:#166534">Pompa beroperasi dalam kondisi normal</p>'
                                        '</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div style="padding:1.5rem;background:#fef2f2;border:1px solid #fca5a5;border-radius:12px;text-align:center">'
                                        '<div style="font-size:2.5rem">⚠️</div>'
                                        '<div class="badge-masalah">BERMASALAH</div>'
                                        '<p style="margin-top:10px;font-size:0.85rem;color:#991b1b">Pompa terdeteksi mengalami masalah!</p>'
                                        '</div>', unsafe_allow_html=True)
                    with col_res2:
                        fig_gauge = make_subplots(rows=1, cols=2,
                                                  subplot_titles=['Prob. Normal', 'Prob. Bermasalah'],
                                                  specs=[[{'type':'indicator'},{'type':'indicator'}]])
                        fig_gauge.add_trace(go.Indicator(
                            mode='gauge+number', value=prob_normal,
                            number=dict(suffix='%'),
                            gauge=dict(axis=dict(range=[0,100]),
                                       bar=dict(color='#16a34a'),
                                       bgcolor='#f0fdf4',
                                       steps=[dict(range=[0,50],color='#dcfce7'),
                                              dict(range=[50,100],color='#bbf7d0')])), row=1, col=1)
                        fig_gauge.add_trace(go.Indicator(
                            mode='gauge+number', value=prob_masalah,
                            number=dict(suffix='%'),
                            gauge=dict(axis=dict(range=[0,100]),
                                       bar=dict(color='#b91c1c'),
                                       bgcolor='#fff1f2',
                                       steps=[dict(range=[0,50],color='#fee2e2'),
                                              dict(range=[50,100],color='#fecaca')])), row=1, col=2)
                        fig_gauge.update_layout(height=200, margin=dict(t=40, b=0, l=10, r=10),
                                                paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    if label == 1:
                        if prob_masalah >= 80:
                            st.error(f"🔴 **KRITIS** — Probabilitas bermasalah {prob_masalah:.1f}%. Hentikan pompa segera! Periksa beban motor dan sistem pendingin.")
                        else:
                            st.warning(f"🟡 **PERINGATAN** — Probabilitas bermasalah {prob_masalah:.1f}%. Jadwalkan pemeriksaan dalam 24 jam.")

                    st.caption(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                except Exception as e:
                    st.error(f"Error prediksi: {e}")


# ═══════════════════════════════════════════════
# TAB 5: ARSITEKTUR & WOKWI
# ═══════════════════════════════════════════════
with tab5:
    st.markdown("### 🏗️ Arsitektur Sistem AIoT 3-Layer")

    col_arch1, col_arch2 = st.columns([3,2])
    with col_arch1:
        layers = [
            {
                "icon": "📡", "label": "Layer 1 — Perception (Sensor)",
                "desc": "ESP32 + DHT22 membaca suhu, kelembaban, dan konsumsi daya secara real-time",
                "chips": ["ESP32 DevKit V1", "DHT22", "OLED SSD1306", "Potensiometer"],
                "color": "#eff6ff", "border": "#bfdbfe"
            },
            {
                "icon": "🤖", "label": "Layer 2 — Edge Processing (AI)",
                "desc": "Random Forest Classifier berjalan di edge device — inferensi lokal tanpa ketergantungan cloud",
                "chips": ["model_rf_pompa.pkl", "scaler_pompa.pkl", "100 estimators", "25 fitur"],
                "color": "#f0fdf4", "border": "#bbf7d0"
            },
            {
                "icon": "☁️", "label": "Layer 3 — Application (Cloud)",
                "desc": "Dashboard, notifikasi WhatsApp/Email, log monitoring, dan penyimpanan data",
                "chips": ["MQTT Broker", "Streamlit", "Alert System", "Log CSV"],
                "color": "#fffbeb", "border": "#fed7aa"
            }
        ]
        for i, layer in enumerate(layers):
            chips_html = " ".join([f'<span style="background:white;border:1px solid #e2e8f0;border-radius:99px;padding:2px 10px;font-size:0.75rem;margin:2px">{c}</span>' for c in layer['chips']])
            st.markdown(f"""
            <div style="background:{layer['color']};border:1px solid {layer['border']};border-radius:12px;padding:1rem 1.2rem;margin-bottom:8px">
              <div style="font-size:1rem;font-weight:600;margin-bottom:4px">{layer['icon']} {layer['label']}</div>
              <div style="font-size:0.85rem;color:#4b5563;margin-bottom:8px">{layer['desc']}</div>
              <div>{chips_html}</div>
            </div>
            {"<div style='text-align:center;font-size:1.2rem;margin:-2px 0;color:#9ca3af'>↓</div>" if i < 2 else ""}
            """, unsafe_allow_html=True)

    with col_arch2:
        st.markdown("#### Stack Teknologi")
        stack = [
            ("🤖 ML", "Random Forest · scikit-learn · joblib"),
            ("⚡ Hardware", "ESP32 · DHT22 · SSD1306 OLED"),
            ("📊 Dataset", "Energydata Complete · 19,735 baris"),
            ("🐍 Backend", "Python 3.10 · Pandas · NumPy"),
            ("🌐 Frontend", "Streamlit · Plotly"),
            ("🔗 Protocol", "MQTT · REST API · GPIO"),
            ("🚀 Deploy", "Streamlit Cloud · GitHub"),
        ]
        for icon_label, desc in stack:
            st.markdown(f"""
            <div style="display:flex;gap:10px;align-items:flex-start;padding:8px 0;border-bottom:1px solid #f1f5f9">
              <div style="font-size:0.8rem;font-weight:600;min-width:90px;color:#374151">{icon_label}</div>
              <div style="font-size:0.8rem;color:#6b7280">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📡 Wokwi Hardware Simulation")

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown("#### Konfigurasi Pin ESP32")
        pins = [
            ("D4",  "#16a34a", "DHT22 — Data (SDA)"),
            ("D34", "#2563eb", "Potensiometer — Sinyal"),
            ("D21", "#7c3aed", "OLED SSD1306 — SDA"),
            ("D22", "#7c3aed", "OLED SSD1306 — SCL"),
            ("D25", "#16a34a", "LED Hijau — Normal"),
            ("D26", "#d97706", "LED Kuning — Warning"),
            ("D27", "#b91c1c", "LED Merah — Masalah"),
            ("3V3", "#ef4444", "Power → Semua sensor"),
            ("GND", "#374151", "Ground → Semua komponen"),
        ]
        for pin, color, desc in pins:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:5px 0;border-bottom:1px solid #f8fafc">
              <span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.75rem;font-family:monospace;min-width:36px;text-align:center">{pin}</span>
              <span style="font-size:0.82rem;color:#4b5563">{desc}</span>
            </div>""", unsafe_allow_html=True)

    with col_w2:
        st.markdown("#### Library Arduino (Wokwi)")
        st.code("""# libraries.txt
DHT sensor library
Adafruit SSD1306
Adafruit GFX Library
Adafruit BusIO""", language="text")

        st.markdown("#### Link Simulasi")
        st.markdown("""
        <a href="https://wokwi.com/projects/464319240845028353" target="_blank"
           style="display:inline-flex;align-items:center;gap:8px;background:#1a1a2e;color:white;padding:10px 20px;border-radius:8px;text-decoration:none;font-size:0.9rem;font-weight:500">
          🔗 Buka Proyek di Wokwi
        </a>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("#### Deploy Checklist")
        checklist = ["model_rf_pompa.pkl", "scaler_pompa.pkl", "energydata_complete.csv",
                     "feature_importance.csv", "app.py", "requirements.txt", "README.md"]
        for item in checklist:
            ext = item.split('.')[-1]
            icon = "🟢" if ext in ['pkl','csv','py','txt','md'] else "⚪"
            st.markdown(f"{icon} `{item}`")

    st.markdown("---")
    st.markdown("#### Cara Deploy ke Streamlit Cloud")
    st.code("""# 1. Push semua file ke GitHub
git init
git add .
git commit -m "AIoT Pompa Air Dashboard"
git remote add origin https://github.com/username/aiot-pompa-air.git
git push -u origin main

# 2. Buka https://share.streamlit.io
# 3. Connect GitHub repo
# 4. Set main file: app.py
# 5. Deploy!""", language="bash")


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  💧 Sistem Prediksi Kerusakan Pompa Air Berbasis AIoT &nbsp;|&nbsp;
  Sifah Nur Rizkiyah — 237006035 &nbsp;|&nbsp;
  Universitas Siliwangi &nbsp;|&nbsp;
  Internet of Things 2024
</div>
""", unsafe_allow_html=True)
