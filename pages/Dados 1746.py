import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# ============================================================
# CONFIGURAÇÃO STREAMLIT
# ============================================================
st.set_page_config(layout="wide", page_title="Chamados 1746 – Mapa Completo")

st.markdown("""
# Chamados 1746 – Visualização Completa em Mapa
Este painel mostra os chamados do 1746 e organiza visualmente tanto todos os registros quanto por categoria (subtipo).
""")

# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("dados/dados_1746.csv")

    # GeoJSON dos bairros (não usado diretamente neste mapa, mas mantido)
    bairros = gpd.read_file("dados/Limite_de_Bairros.geojson")
    return df, bairros


df_raw, bairros = load_data()

# ============================================================
# LIMPEZA E FILTRO (20% MAIS RECENTES)
# ============================================================

# Converter lat/lon para numérico
df_raw["latitude"] = pd.to_numeric(df_raw["latitude"], errors="coerce")
df_raw["longitude"] = pd.to_numeric(df_raw["longitude"], errors="coerce")

# Remover onde não há coordenadas válidas
df = df_raw.dropna(subset=["latitude", "longitude"]).copy()

# Se existir coluna data, ordenar
if "data" in df.columns:
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.sort_values("data")

# ============================================================
# MOSTRAR TABELA
# ============================================================
st.subheader("Base de Dados – 1746 (20% mais recentes)")
st.dataframe(df, use_container_width=True)
