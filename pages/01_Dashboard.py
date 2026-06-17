import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.markdown("# Dashboard")

# Access artifacts from session state
if "artifacts" in st.session_state:
    artifacts = st.session_state.artifacts
    df = artifacts.get("df")
else:
    # Fallback: try to load manually
    st.warning("Memuat data...")
    import joblib
    import os
    try:
        df = pd.read_csv("data/Distribusi_Pupuk_Jatim_2023-2025.csv")
        # Clean numeric columns
        for col in ["luas_panen_kedelai", "produksi_kedelai_ton"]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    except:
        df = None

if df is None:
    st.error("Data tidak dapat dimuat!")
else:
    # --- KPI Cards ---
    total_records = len(df)
    num_regions = df['kabupaten'].nunique()
    year_min = df['tahun'].min()
    year_max = df['tahun'].max()
    
    st.markdown("## Metrik Utama")
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Total Data", f"{total_records:,}", delta="3 tahun")
    with kpi2:
        st.metric("Jumlah Kabupaten/Kota", f"{num_regions}", delta="38 wilayah")
    with kpi3:
        st.metric("Periode Data", f"{year_min} - {year_max}")

    st.markdown("---")
    
    # --- Preprocess Data for Visualizations ---
    df['distribusi_total'] = df['distribusi_urea'] + df['distribusi_NPK']
    bulan_map = {
        "Januari":1, "Februari":2, "Maret":3, "April":4, 
        "Mei":5, "Juni":6, "Juli":7, "Agustus":8, 
        "September":9, "Oktober":10, "November":11, "Desember":12
    }
    df['bulan_num'] = df['bulan'].map(bulan_map)
    df['date'] = pd.to_datetime(
        df['tahun'].astype(str) + '-' + df['bulan_num'].astype(str),
        format='%Y-%m'
    )
    
    # --- Visualization 1: Fertilizer over time ---
    st.markdown("### Distribusi Pupuk Berdasarkan Waktu")
    fig_time = px.line(
        df.groupby('date')['distribusi_total'].sum().reset_index(),
        x='date',
        y='distribusi_total',
        title='Total Distribusi Pupuk (Urea + NPK) dari Waktu ke Waktu',
        color_discrete_sequence=['#2e7d32']
    )
    fig_time.update_layout(
        xaxis_title="Tanggal",
        yaxis_title="Total Distribusi (Ton)",
        template="plotly_white"
    )
    st.plotly_chart(fig_time, use_container_width=True)
    
    # --- Visualization 2: Fertilizer by Region ---
    st.markdown("### Distribusi Pupuk per Kabupaten/Kota")
    region_total = df.groupby('kabupaten')['distribusi_total'].sum().sort_values(ascending=False).reset_index()
    fig_region = px.bar(
        region_total.head(10),
        x='distribusi_total',
        y='kabupaten',
        orientation='h',
        title='Top 10 Kabupaten/Kota dengan Distribusi Pupuk Terbesar',
        color='distribusi_total',
        color_continuous_scale='Greens'
    )
    fig_region.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_region, use_container_width=True)
    
    # --- Visualization 3: Rainfall vs Fertilizer ---
    st.markdown("### Hubungan Curah Hujan dengan Distribusi Pupuk")
    fig_scatter = px.scatter(
        df,
        x='curah_hujan',
        y='distribusi_total',
        title='Curah Hujan vs Total Distribusi Pupuk',
        color_discrete_sequence=['#4caf50'],
        opacity=0.6
    )
    fig_scatter.update_layout(template="plotly_white")
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # --- Visualization 4: Harvest Area vs Fertilizer ---
    st.markdown("### Hubungan Luas Panen dengan Distribusi Pupuk")
    df['luas_panen_total'] = (
        df['luas_panen_padi'] + df['luas_panen_jagung'] + df['luas_panen_kedelai']
    )
    fig_area = px.scatter(
        df,
        x='luas_panen_total',
        y='distribusi_total',
        title='Luas Panen Total vs Total Distribusi Pupuk',
        color_discrete_sequence=['#8bc34a'],
        opacity=0.6
    )
    fig_area.update_layout(template="plotly_white")
    st.plotly_chart(fig_area, use_container_width=True)
