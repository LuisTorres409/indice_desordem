import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import numpy as np
import branca.colormap as cm
from streamlit.components.v1 import html

COL_BAIRRO = "nome"

# ============================================================
# MESES
# ============================================================
MESES_OPCOES = [
    "Ano inteiro",
    "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
    "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"
]
MAPA_MES_NOME_NUM = {
    "Janeiro": 1, "Fevereiro": 2, "Março": 3, "Abril": 4,
    "Maio": 5, "Junho": 6, "Julho": 7, "Agosto": 8,
    "Setembro": 9, "Outubro": 10, "Novembro": 11, "Dezembro": 12
}

# ============================================================
# NORMALIZAÇÃO
# ============================================================
def normalizar_intra(vals: pd.Series) -> pd.Series:
    vmax, vmin = vals.max(), vals.min()
    if vmax == vmin:
        return pd.Series([1.0] * len(vals), index=vals.index)
    return (vals - vmin) / (vmax - vmin)


# ============================================================
# CÁLCULO DOS ÍNDICES
# agora retorna também df_freq para montar tabelas detalhadas no popup
# ============================================================
def construir_indices(df_raw, col_categoria, pesos_custom, gdf_bairros, nome_col_total):
    """
    1746:
        - com coordenada: spatial join
        - sem coordenada: join por id_bairro

    Disque Denúncia:
        - sempre via latitude/longitude (spatial join)
        - registros sem coordenada são descartados
    """

    def _df_freq_vazio():
        return pd.DataFrame(
            columns=[
                COL_BAIRRO,
                col_categoria,
                "count",
                "alpha",
                "norm_intra",
                "norm_global",
                "norm_area",
                "norm_pop",
                "norm_dom",
            ]
        )

    # ============================================================
    # 0. CASOS SEM DADOS
    # ============================================================
    if df_raw is None or df_raw.empty:
        gdf_final = gdf_bairros.copy()
        gdf_final[nome_col_total] = 0
        for col in ["indice_intra", "indice_global", "indice_area", "indice_pop", "indice_dom"]:
            gdf_final[col] = 0

        gdf_final.attrs["log_info"] = {
            "is_1746": "id_bairro" in (df_raw.columns if df_raw is not None else []),
            "total_registros": 0,
            "total_com_coord": 0,
            "total_sem_coord": 0,
            "inseridos_por_id": 0,
            "sem_bairro_mesmo_com_id": 0,
        }
        return gdf_final, _df_freq_vazio()

    df = df_raw.copy()

    # mapeia pesos em upper para filtragem / cálculo
    pesos_upper = {k.upper(): v for k, v in pesos_custom.items()}

    # coluna de categoria em upper para consistência NO CÁLCULO
    df[col_categoria] = df[col_categoria].astype(str)
    df[col_categoria] = df[col_categoria].str.upper()

    # filtra apenas categorias que têm peso definido
    df = df[df[col_categoria].isin(pesos_upper.keys())].copy()

    if df.empty:
        gdf_final = gdf_bairros.copy()
        gdf_final[nome_col_total] = 0
        for col in ["indice_intra", "indice_global", "indice_area", "indice_pop", "indice_dom"]:
            gdf_final[col] = 0

        gdf_final.attrs["log_info"] = {
            "is_1746": "id_bairro" in df.columns,
            "total_registros": 0,
            "total_com_coord": 0,
            "total_sem_coord": 0,
            "inseridos_por_id": 0,
            "sem_bairro_mesmo_com_id": 0,
        }
        return gdf_final, _df_freq_vazio()

    # ============================================================
    # 1. IDENTIFICAR SE ESTE DATAFRAME É 1746 OU DISQUE DENÚNCIA
    # ============================================================
    is_1746 = "id_bairro" in df.columns  # Disque Denúncia não possui isso

    # ============================================================
    # 2. SEPARAR COM E SEM COORDENADAS
    # ============================================================
    total_registros = len(df)
    has_coords = df["latitude"].notna() & df["longitude"].notna()
    df_com_coord = df[has_coords].copy()
    df_sem_coord = df[~has_coords].copy()
    total_com_coord = len(df_com_coord)
    total_sem_coord = len(df_sem_coord)

    # ============================================================
    # 3. SPATIAL JOIN PARA REGISTROS COM COORDENADA
    # ============================================================
    if not df_com_coord.empty:
        gdf_com_coord = gpd.GeoDataFrame(
            df_com_coord,
            geometry=gpd.points_from_xy(df_com_coord.longitude, df_com_coord.latitude),
            crs="EPSG:4326",
        )

        gdf_joined_com_coord = gpd.sjoin(
            gdf_com_coord,
            gdf_bairros,
            how="left",
            predicate="within",
        )

        gdf_joined_com_coord["geometry"] = gdf_com_coord.geometry
        gdf_joined_com_coord["origem_bairro"] = "spatial"
    else:
        gdf_joined_com_coord = gpd.GeoDataFrame(columns=list(df.columns) + ["geometry", COL_BAIRRO])

    # ============================================================
    # 4. TRATAMENTO DOS SEM COORDENADA + CONTADORES
    # ============================================================
    inseridos_por_id = 0
    sem_bairro_mesmo_com_id = 0

    if is_1746:
        # -------- 1746 --------
        if not df_sem_coord.empty:
            # 4.1 Converter id_bairro para numérico e dropar NaN
            df_sem_coord["id_bairro"] = pd.to_numeric(df_sem_coord["id_bairro"], errors="coerce")
            n_sem_coord_total = len(df_sem_coord)
            df_sem_coord = df_sem_coord[df_sem_coord["id_bairro"].notna()].copy()
            n_sem_coord_sem_id = n_sem_coord_total - len(df_sem_coord)

            # 4.2 Normalizar id_bairro e codbairro como strings de 3 dígitos
            df_sem_coord["id_bairro"] = (
                df_sem_coord["id_bairro"].astype(int).astype(str).str.zfill(3)
            )

            gdf_bairros["codbairro"] = (
                pd.to_numeric(gdf_bairros["codbairro"], errors="coerce")
                .astype(int)
                .astype(str)
                .str.zfill(3)
            )

            # 4.3 Merge pelo ID padronizado
            df_sem_coord = df_sem_coord.merge(
                gdf_bairros[["codbairro", COL_BAIRRO]],
                left_on="id_bairro",
                right_on="codbairro",
                how="left",
            )

            # 4.4 marca origem
            df_sem_coord["geometry"] = None
            df_sem_coord["origem_bairro"] = "id"

            gdf_sem_coord = gpd.GeoDataFrame(df_sem_coord, geometry="geometry", crs="EPSG:4326")

            # contadores
            inseridos_por_id = len(df_sem_coord) - n_sem_coord_sem_id
            sem_bairro_mesmo_com_id = n_sem_coord_sem_id
        else:
            gdf_sem_coord = gpd.GeoDataFrame(columns=list(df.columns) + ["geometry", COL_BAIRRO])
    else:
        # -------- DISQUE DENÚNCIA --------
        inseridos_por_id = 0
        sem_bairro_mesmo_com_id = len(df_sem_coord)
        gdf_sem_coord = gpd.GeoDataFrame(columns=list(df.columns) + ["geometry", COL_BAIRRO])

    # ============================================================
    # 5. UNIR RESULTADOS
    # ============================================================
    joined = pd.concat([gdf_joined_com_coord, gdf_sem_coord], ignore_index=True)
    joined = gpd.GeoDataFrame(joined, geometry="geometry", crs="EPSG:4326")

    # ============================================================
    # 6. FREQUÊNCIAS
    # df_freq é o coração para o popup detalhado
    # ============================================================
    df_freq = (
        joined.groupby([COL_BAIRRO, col_categoria])
        .size()
        .reset_index(name="count")
    )

    df_freq = df_freq.merge(
        gdf_bairros[[COL_BAIRRO, "area_km2", "Total_de_pessoas_2022", "Total_de_domicilios_2022"]],
        on=COL_BAIRRO,
        how="left",
    )

    df_freq["alpha"] = df_freq[col_categoria].map(lambda x: pesos_upper.get(x, 0))

    # ============================================================
    # 7. NORMALIZAÇÕES (COM P95 RESTAURADO)
    # ============================================================

    # 7.1 Normalização Intra-Bairro (não usa p95)
    df_freq["norm_intra"] = df_freq.groupby(COL_BAIRRO)["count"].transform(normalizar_intra)

    # Helper p95
    def p95_por_categoria(s):
        return np.percentile(s, 95) if len(s) > 0 else 1

    # 7.2 GLOBAL
    p95_global = (
        df_freq.groupby(col_categoria)["count"].transform(p95_por_categoria).replace(0, 1)
    )
    df_freq["norm_global"] = (df_freq["count"] / p95_global).clip(0, 1)

    # 7.3 ÁREA
    df_freq["adj_area"] = df_freq["count"] / df_freq["area_km2"].replace(0, 1)
    p95_area = (
        df_freq.groupby(col_categoria)["adj_area"].transform(p95_por_categoria).replace(0, 1)
    )
    df_freq["norm_area"] = (df_freq["adj_area"] / p95_area).clip(0, 1)

    # 7.4 POPULAÇÃO
    df_freq["adj_pop"] = df_freq["count"] / df_freq["Total_de_pessoas_2022"].replace(0, 1)
    p95_pop = (
        df_freq.groupby(col_categoria)["adj_pop"].transform(p95_por_categoria).replace(0, 1)
    )
    df_freq["norm_pop"] = (df_freq["adj_pop"] / p95_pop).clip(0, 1)

    # 7.5 DOMICÍLIOS
    df_freq["adj_dom"] = df_freq["count"] / df_freq["Total_de_domicilios_2022"].replace(0, 1)
    p95_dom = (
        df_freq.groupby(col_categoria)["adj_dom"].transform(p95_por_categoria).replace(0, 1)
    )
    df_freq["norm_dom"] = (df_freq["adj_dom"] / p95_dom).clip(0, 1)

    # ============================================================
    # 7.x ÍNDICES PONDERADOS (para o shape final)
    # ============================================================
    df_freq["w_intra"] = df_freq["alpha"] * df_freq["norm_intra"]
    indice_intra = (
        df_freq.groupby(COL_BAIRRO)["w_intra"]
        .sum()
        .reset_index()
        .rename(columns={"w_intra": "indice_intra"})
    )

    df_freq["w_global"] = df_freq["alpha"] * df_freq["norm_global"]
    indice_global = (
        df_freq.groupby(COL_BAIRRO)["w_global"]
        .sum()
        .reset_index()
        .rename(columns={"w_global": "indice_global"})
    )

    df_freq["w_area"] = df_freq["alpha"] * df_freq["norm_area"]
    indice_area = (
        df_freq.groupby(COL_BAIRRO)["w_area"]
        .sum()
        .reset_index()
        .rename(columns={"w_area": "indice_area"})
    )

    df_freq["w_pop"] = df_freq["alpha"] * df_freq["norm_pop"]
    indice_pop = (
        df_freq.groupby(COL_BAIRRO)["w_pop"]
        .sum()
        .reset_index()
        .rename(columns={"w_pop": "indice_pop"})
    )

    df_freq["w_dom"] = df_freq["alpha"] * df_freq["norm_dom"]
    indice_dom = (
        df_freq.groupby(COL_BAIRRO)["w_dom"]
        .sum()
        .reset_index()
        .rename(columns={"w_dom": "indice_dom"})
    )

    # ============================================================
    # 8. TOTAL E CLASSES (pivot de contagens brutas)
    # ============================================================
    total = (
        df_freq.groupby(COL_BAIRRO)["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": nome_col_total})
    )

    tabela_classes = (
        df_freq.pivot_table(
            index=COL_BAIRRO,
            columns=col_categoria,
            values="count",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()
    )

    # ============================================================
    # 9. MERGE FINAL NO GDF
    # ============================================================
    gdf_final = (
        gdf_bairros
        .merge(total, on=COL_BAIRRO, how="left")
        .merge(indice_intra, on=COL_BAIRRO, how="left")
        .merge(indice_global, on=COL_BAIRRO, how="left")
        .merge(indice_area, on=COL_BAIRRO, how="left")
        .merge(indice_pop, on=COL_BAIRRO, how="left")
        .merge(indice_dom, on=COL_BAIRRO, how="left")
        .merge(tabela_classes, on=COL_BAIRRO, how="left")
        .fillna(0)
    )

    # ============================================================
    # 10. LOG DE QUALIDADE
    # ============================================================
    log_info = {
        "is_1746": is_1746,
        "total_registros": int(total_registros),
        "total_com_coord": int(total_com_coord),
        "total_sem_coord": int(total_sem_coord),
        "inseridos_por_id": int(inseridos_por_id),
        "sem_bairro_mesmo_com_id": int(sem_bairro_mesmo_com_id),
    }
    gdf_final.attrs["log_info"] = log_info

    fonte = "1746" if is_1746 else "Disque Denúncia"
    print(f"[LOG {fonte}] {log_info}")

    return gdf_final, df_freq

# ============================================================
# FUNÇÃO DE MAPA 1746 / DD (com popups ricos)
# AGORA: tabela completa por classe (peso + normalizações)
# ============================================================
def gerar_mapa_indices(gdf, df_freq, titulo, palette, prefixo_total, categoria_cols, col_categoria, pesos_custom):

    centro = gdf.geometry.unary_union.centroid
    mapa = folium.Map(location=[centro.y, centro.x], zoom_start=11, tiles="CartoDB Positron")

    # mapeia pesos em upper (para lookup) E nome original para exibir
    pesos_upper = {k.upper(): v for k, v in pesos_custom.items()}
    label_original = {k.upper(): k for k in pesos_custom.keys()}

    def add_layer(col_indice, nome_layer):
        vmin, vmax = gdf[col_indice].min(), gdf[col_indice].max()
        colormap = cm.LinearColormap(colors=palette, vmin=vmin, vmax=vmax, caption=nome_layer)

        # lógica: só População vem ligada
        if nome_layer.split("–")[-1].strip() == "População":
            layer = folium.FeatureGroup(name=nome_layer, show=True)
        else:
            layer = folium.FeatureGroup(name=nome_layer, show=False)

        for _, row in gdf.iterrows():
            valor = row[col_indice]
            color = colormap(valor) if valor > 0 else "#cccccc"

            # monta linhas da tabela por classe, usando df_freq
            linhas = []
            for c_upper in categoria_cols:
                filtro = (
                    (df_freq[COL_BAIRRO] == row[COL_BAIRRO]) &
                    (df_freq[col_categoria] == c_upper)
                )
                sub = df_freq.loc[filtro]

                if not sub.empty:
                    rec = sub.iloc[0]
                    count_c = rec["count"]
                    alpha_c = rec["alpha"]
                    n_intra = rec["norm_intra"]
                    n_glob = rec["norm_global"]
                    n_area = rec["norm_area"]
                    n_pop = rec["norm_pop"]
                    n_dom = rec["norm_dom"]
                else:
                    # sem registros daquela classe naquele bairro
                    count_c = 0
                    alpha_c = pesos_upper.get(c_upper, 0)
                    n_intra = n_glob = n_area = n_pop = n_dom = 0.0

                nome_classe = label_original.get(c_upper, c_upper)

                linhas.append(
                    f"""
                    <tr>
                        <td style="text-align:left;">{nome_classe}</td>
                        <td>{alpha_c:.2f}</td>
                        <td>{count_c:.0f}</td>
                        <td>{n_intra:.3f}</td>
                        <td>{n_glob:.3f}</td>
                        <td>{n_area:.3f}</td>
                        <td>{n_pop:.3f}</td>
                        <td>{n_dom:.3f}</td>
                    </tr>
                    """
                )

            if linhas:
                tabela_classes_html = "\n".join(linhas)
            else:
                tabela_classes_html = """
                <tr>
                    <td colspan="8">Sem registros para as classes ponderadas neste bairro.</td>
                </tr>
                """

            popup_html = f"""
            <div style="font-family: Arial; padding: 6px; font-size: 13px;">
                <h4 style="margin-bottom:5px;">{row[COL_BAIRRO]}</h4>

                <b>Área:</b> {row['area_km2']:.2f} km²<br>
                <b>População:</b> {row['Total_de_pessoas_2022']:.0f}<br>
                <b>Domicílios:</b> {row['Total_de_domicilios_2022']:.0f}<br>
                <b>Total:</b> {row[prefixo_total]:.0f}<br><br>

                <b>Índices (normalizações – {titulo}):</b>
                <ul style="margin-top:4px;">
                    <li><b>Intra-bairro:</b> {row['indice_intra']:.3f}</li>
                    <li><b>Global:</b> {row['indice_global']:.3f}</li>
                    <li><b>Área:</b> {row['indice_area']:.3f}</li>
                    <li><b>População:</b> {row['indice_pop']:.3f}</li>
                    <li><b>Domicílios:</b> {row['indice_dom']:.3f}</li>
                </ul>

                <b>Detalhamento por classe (pesos e normalizados):</b><br>
                <table border="1" style="width:100%; border-collapse: collapse; text-align:center; font-size:12px;">
                    <tr style="background:#f0f0f0; font-weight:bold;">
                        <th style="text-align:left;">Classe</th>
                        <th>Peso</th>
                        <th>Count</th>
                        <th>norm_intra</th>
                        <th>norm_global</th>
                        <th>norm_área</th>
                        <th>norm_pop</th>
                        <th>norm_dom</th>
                    </tr>
                    {tabela_classes_html}
                </table>
            </div>
            """

            folium.GeoJson(
                data=row["geometry"],
                style_function=lambda feat, color=color: {
                    "fillColor": color,
                    "color": "#333",
                    "weight": 1,
                    "fillOpacity": 0.6,
                },
                tooltip=f"{row[COL_BAIRRO]} — {valor:.3f}",
                popup=folium.Popup(popup_html, max_width=550),
            ).add_to(layer)

        colormap.add_to(mapa)
        layer.add_to(mapa)

    # layers
    add_layer("indice_intra",  f"{titulo} – Intra")
    add_layer("indice_global", f"{titulo} – Global")
    add_layer("indice_area",   f"{titulo} – Área")
    add_layer("indice_pop",    f"{titulo} – População")
    add_layer("indice_dom",    f"{titulo} – Domicílios")

    folium.LayerControl(collapsed=False).add_to(mapa)
    return mapa

# ============================================================
# CARREGAMENTO
# ============================================================
@st.cache_data
def carregar_dados():
    from pathlib import Path

    # ==========================
    # 1) CARREGAR GEOJSON
    # ==========================
    bairros = gpd.read_file("dados/Limite_de_Bairros.geojson")

    # ==========================
    # 2) CARREGAR 1746 MENSAL
    # ==========================
    pasta_mensal = Path("dados/1746_mensal")

    # Pega todos CSV dentro da pasta
    arquivos_1746_mensais = sorted(pasta_mensal.glob("*.csv"))

    if arquivos_1746_mensais:
        dfs = []
        for arq in arquivos_1746_mensais:
            df_mes = pd.read_csv(arq)
            dfs.append(df_mes)
        df_1746 = pd.concat(dfs, ignore_index=True)
    else:
        # fallback
        df_1746 = pd.read_csv("dados/dados_1746_large.csv")

    # ==========================
    # 3) DISQUE DENÚNCIA
    # ==========================
    df_dd = pd.read_parquet("dados/dados_dd.parquet")

    # ==========================
    # 4) CENSO + ÁREA
    # ==========================
    df_censo = pd.read_csv("dados/bairros_censo.csv")

    bairros_m = bairros.to_crs(3857)
    bairros["area_km2"] = bairros_m.geometry.area / 1e6

    mapa_equivalencia = {
        "Imperial de São Cristóvão": "São Cristóvão"
    }
    bairros["nome"] = bairros["nome"].replace(mapa_equivalencia)

    bairros = bairros.merge(
        df_censo[["nome", "Total_de_pessoas_2022", "Total_de_domicilios_2022"]],
        on="nome",
        how="left",
    ).fillna(0)

    return bairros, df_1746, df_dd

bairros, df_1746_full, df_dd_full = carregar_dados()