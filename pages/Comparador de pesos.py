import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import branca.colormap as cm
from streamlit.components.v1 import html

import sys
sys.path.append("..")

# Importa exatamente as funções usadas no app principal
from utils import (
    carregar_dados,
    construir_indices,
    gerar_mapa_indices
)

COL_BAIRRO = "nome"

st.set_page_config(page_title="Comparador de Perfis de Peso — 1746", layout="wide")

# ============================================================
# CARREGAMENTO DOS DADOS (MESMA LÓGICA DO APP PRINCIPAL)
# ============================================================
bairros, df_1746_full, _ = carregar_dados()   # DD não usado aqui

# ============================================================
# DEFINIÇÃO DOS 5 PERFIS DE PESOS
# ============================================================
perfis = {
    "Perfil A – Balanced": {
        "Fiscalização de estacionamento irregular de veículo": 0.17,
        "Remoção de entulho e bens inservíveis": 0.17,
        "Remoção de resíduos no logradouro": 0.17,
        "Reparo de Luminária": 0.17,
        "Reparo de buraco, deformação ou afundamento na pista": 0.16,
        "Poda de árvore em logradouro": 0.16,
    },
    "Perfil B – Prioriza Buracos": {
        "Fiscalização de estacionamento irregular de veículo": 0.10,
        "Remoção de entulho e bens inservíveis": 0.10,
        "Remoção de resíduos no logradouro": 0.10,
        "Reparo de Luminária": 0.10,
        "Reparo de buraco, deformação ou afundamento na pista": 0.40,
        "Poda de árvore em logradouro": 0.20,
    },
    "Perfil C – Ambiental": {
        "Fiscalização de estacionamento irregular de veículo": 0.10,
        "Remoção de entulho e bens inservíveis": 0.25,
        "Remoção de resíduos no logradouro": 0.25,
        "Reparo de Luminária": 0.10,
        "Reparo de buraco, deformação ou afundamento na pista": 0.10,
        "Poda de árvore em logradouro": 0.20,
    },
    "Perfil D – Iluminação e Estacionamento": {
        "Fiscalização de estacionamento irregular de veículo": 0.30,
        "Remoção de entulho e bens inservíveis": 0.10,
        "Remoção de resíduos no logradouro": 0.10,
        "Reparo de Luminária": 0.30,
        "Reparo de buraco, deformação ou afundamento na pista": 0.10,
        "Poda de árvore em logradouro": 0.10,
    },
    "Perfil E – Urbanismo Clássico": {
        "Fiscalização de estacionamento irregular de veículo": 0.15,
        "Remoção de entulho e bens inservíveis": 0.15,
        "Remoção de resíduos no logradouro": 0.15,
        "Reparo de Luminária": 0.20,
        "Reparo de buraco, deformação ou afundamento na pista": 0.20,
        "Poda de árvore em logradouro": 0.15,
    },
}

# ============================================================
# TEXTO EXPLICATIVO COM PESOS DETALHADOS
# ============================================================

def gerar_texto_perfis(perfis):
    partes = []
    for nome, dpesos in perfis.items():
        partes.append(f"### {nome}")
        for classe, w in dpesos.items():
            partes.append(f"- **{classe}: {w*100:.1f}%**")
        partes.append("")  # linha vazia
    return "\n".join(partes)

texto_perfis = gerar_texto_perfis(perfis)

st.markdown(f"""
# Comparador de Perfis de Pesos – Chamados 1746

Este módulo permite **comparar diferentes visões de prioridade** sobre os chamados do 1746.

Cada perfil representa uma filosofia distinta de leitura do território —  
e abaixo você vê **exatamente quanto cada classe contribui (%)** para o índice final do perfil:

{texto_perfis}

O mapa abaixo contém **5 layers**, cada um representando o índice calculado com um perfil específico.
Ative e desative os layers no canto superior direito para comparar padrões, contrastes e mudanças estruturais
na distribuição territorial dos chamados.
""")

# ============================================================
# CALCULAR ÍNDICES PARA CADA PERFIL
# ============================================================
resultados = {}   # salva (gdf, df_freq) de cada perfil

for nome_perfil, pesos in perfis.items():
    gdf, df_freq = construir_indices(
        df_raw=df_1746_full,
        col_categoria="subtipo",
        pesos_custom=pesos,
        gdf_bairros=bairros,
        nome_col_total=f"total_{nome_perfil}"
    )
    resultados[nome_perfil] = (gdf, df_freq)

# ============================================================
# GERAR MAPA COM TODOS OS PERFIS COMO LAYERS
# ============================================================
centro = bairros.geometry.unary_union.centroid
mapa = folium.Map(location=[centro.y, centro.x], zoom_start=11, tiles="CartoDB Positron")

palette = ["#edf8fb", "#b3cde3", "#8c96c6", "#8856a7", "#810f7c"]

for nome_perfil, (gdfP, df_freqP) in resultados.items():

    col_indice = "indice_pop"  

    vmin, vmax = gdfP[col_indice].min(), gdfP[col_indice].max()
    colormap = cm.LinearColormap(
        colors=palette,
        vmin=vmin,
        vmax=vmax,
        caption=nome_perfil
    )

    layer = folium.FeatureGroup(name=nome_perfil, show=False)

    pesos_custom = perfis[nome_perfil]
    pesos_upper = {k.upper(): v for k, v in pesos_custom.items()}
    classes_upper = [k.upper() for k in pesos_custom.keys()]
    label_original = {k.upper(): k for k in pesos_custom.keys()}

    for _, row in gdfP.iterrows():

        valor = row[col_indice]
        color = colormap(valor) if valor > 0 else "#cccccc"

        # Soma total de registros do bairro no perfil
        total_reg = df_freqP[df_freqP[COL_BAIRRO] == row[COL_BAIRRO]]["count"].sum()

        # Construção das linhas da tabela
        linhas = []
        for c in classes_upper:

            sub = df_freqP[
                (df_freqP[COL_BAIRRO] == row[COL_BAIRRO]) &
                (df_freqP["subtipo"] == c)
            ]

            if not sub.empty:
                rec = sub.iloc[0]
                count_c = rec["count"]
                n_area = rec["norm_area"]
                n_pop = rec["norm_pop"]

                pop = rec["Total_de_pessoas_2022"]
                dom = rec["Total_de_domicilios_2022"]

                q_pop = count_c / pop if pop > 0 else 0
                q_dom = count_c / dom if dom > 0 else 0

            else:
                count_c = 0
                n_area = n_pop = q_pop = q_dom = 0

            linhas.append(f"""
            <tr>
                <td class='col-classe'>{label_original[c]}</td>
                <td class='col-num'>{pesos_upper[c]:.2f}</td>
                <td class='col-num'>{count_c}</td>
                <td class='col-num'>{n_area:.3f}</td>
                <td class='col-num'>{n_pop:.3f}</td>
                <td class='col-num'>{q_pop:.5f}</td>
                <td class='col-num'>{q_dom:.5f}</td>
            </tr>
            """)

        tabela = "\n".join(linhas)

        popup = f"""
        <div style="font-family:Arial; font-size:13px; padding:10px;">

            <h4 style="margin-bottom:8px;">{row[COL_BAIRRO]}</h4>

            <div style="
                background:#f7f7f7;
                border:1px solid #ddd;
                padding:8px;
                margin-bottom:12px;
                border-radius:4px;
            ">
                <b>Área:</b> {row['area_km2']:.2f} km²<br>
                <b>População:</b> {row['Total_de_pessoas_2022']:,}<br>
                <b>Domicílios:</b> {row['Total_de_domicilios_2022']:,}<br>
                <b>Total de registros (perfil):</b> {total_reg}<br>
                <b>Índice ({col_indice}):</b> {valor:.3f}
            </div>

            <b>Detalhamento por classe do perfil:</b><br><br>

            <table style="
                width:100%;
                border-collapse:collapse;
                font-size:12px;
                border:1px solid #ccc;
            ">

                <tr style="background:#eaeaea;">
                    <th class='th'>Classe</th>
                    <th class='th'>Peso</th>
                    <th class='th'>Qtd</th>
                    <th class='th'>Norm Área</th>
                    <th class='th'>Norm Pop</th>
                    <th class='th'>Qtd/Pop</th>
                    <th class='th'>Qtd/Dom</th>
                </tr>

                {tabela}

            </table>
        </div>

        <style>

            .th {{
                padding:6px;
                border-right:1px solid #ddd;
                text-align:center;
                font-weight:bold;
            }}

            td {{
                padding:6px 4px;
                border-bottom:1px solid #eee;
                border-right:1px solid #ddd;
            }}

            .col-classe {{
                text-align:left;
                padding-left:6px;
                width:190px;
            }}

            .col-num {{
                text-align:center;
                width:70px;
            }}

        </style>
        """

        folium.GeoJson(
            data=row["geometry"],
            style_function=lambda feat, color=color: {
                "fillColor": color,
                "color": "#222",
                "weight": 1,
                "fillOpacity": 0.6,
            },
            popup=folium.Popup(popup, max_width=550),
            tooltip=f"{row[COL_BAIRRO]} — {valor:.3f}"
        ).add_to(layer)

    colormap.add_to(mapa)
    layer.add_to(mapa)


folium.LayerControl(collapsed=False).add_to(mapa)

# ============================================================
# MOSTRAR MAPA
# ============================================================
html(mapa.get_root().render(), height=800)
