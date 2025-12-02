import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import numpy as np
import branca.colormap as cm
from streamlit.components.v1 import html

# ============================================================
# CONFIGURAÇÃO GERAL
# ============================================================
st.set_page_config(layout="wide", page_title="Mapa Índice 1746 + Disque Denúncia + Combinado")
COL_BAIRRO = "nome"

# ============================================================
# MESES
# ============================================================
MESES_OPCOES = [
    "Ano inteiro",
    "Janeiro","Fevereiro","Março","Abril","Maio","Junho",
    "Julho","Agosto","Setembro","Outubro","Novembro","Dezembro"
]
MAPA_MES_NOME_NUM = {
    "Janeiro":1, "Fevereiro":2, "Março":3, "Abril":4,
    "Maio":5, "Junho":6, "Julho":7, "Agosto":8,
    "Setembro":9, "Outubro":10, "Novembro":11, "Dezembro":12
}

# ============================================================
# NORMALIZAÇÃO
# ============================================================
def normalizar_intra(vals: pd.Series) -> pd.Series:
    vmax, vmin = vals.max(), vals.min()
    if vmax == vmin:
        return pd.Series([1.0]*len(vals), index=vals.index)
    return (vals - vmin) / (vmax - vmin)

# ============================================================
# CÁLCULO DOS ÍNDICES
# ============================================================
def construir_indices(df_raw, col_categoria, pesos_custom, gdf_bairros, nome_col_total):

    if df_raw is None or df_raw.empty:
        gdf_final = gdf_bairros.copy()
        gdf_final[nome_col_total] = 0
        for col in ["indice_intra","indice_global","indice_area","indice_pop","indice_dom"]:
            gdf_final[col] = 0
        return gdf_final

    df = df_raw.copy()
    df[col_categoria] = df[col_categoria].astype(str).str.upper()
    pesos_upper = {k.upper(): v for k, v in pesos_custom.items()}
    df = df[df[col_categoria].isin(pesos_upper.keys())].copy()

    if df.empty:
        gdf_final = gdf_bairros.copy()
        gdf_final[nome_col_total] = 0
        for col in ["indice_intra","indice_global","indice_area","indice_pop","indice_dom"]:
            gdf_final[col] = 0
        return gdf_final

    # GeoDataFrame dos pontos
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )

    joined = gpd.sjoin(gdf, gdf_bairros, how="left", predicate="within")

    # Frequência por bairro x categoria
    df_freq = (
        joined.groupby([COL_BAIRRO, col_categoria])
        .size()
        .reset_index(name="count")
    )

    df_freq = df_freq.merge(
        gdf_bairros[[COL_BAIRRO,"area_km2","Total_de_pessoas_2022","Total_de_domicilios_2022"]],
        on=COL_BAIRRO, how="left"
    )

    df_freq["alpha"] = df_freq[col_categoria].map(lambda x: pesos_upper.get(x, 0))

    # =========================
    # INTRA
    # =========================
    df_freq["norm_intra"] = df_freq.groupby(COL_BAIRRO)["count"].transform(normalizar_intra)
    df_freq["w_intra"] = df_freq["alpha"] * df_freq["norm_intra"]
    indice_intra = (
        df_freq.groupby(COL_BAIRRO)["w_intra"].sum()
        .reset_index()
        .rename(columns={"w_intra":"indice_intra"})
    )

    # =========================
    # GLOBAL (ajuste via p95 por categoria)
    # =========================
    df_freq["adj_global"] = df_freq["count"]
    p95 = df_freq.groupby(col_categoria)["adj_global"].transform(lambda s: s.quantile(0.95)).replace(0,1)
    df_freq["norm_global"] = (df_freq["adj_global"]/p95).clip(0,1)
    df_freq["w_global"] = df_freq["alpha"] * df_freq["norm_global"]
    indice_global = (
        df_freq.groupby(COL_BAIRRO)["w_global"].sum()
        .reset_index()
        .rename(columns={"w_global":"indice_global"})
    )

    # =========================
    # ÁREA
    # =========================
    df_freq["adj_area"] = df_freq["count"]/df_freq["area_km2"].replace(0,1)
    p95_area = df_freq.groupby(col_categoria)["adj_area"].transform(lambda s: s.quantile(0.95)).replace(0,1)
    df_freq["norm_area"] = (df_freq["adj_area"]/p95_area).clip(0,1)
    df_freq["w_area"] = df_freq["alpha"] * df_freq["norm_area"]
    indice_area = (
        df_freq.groupby(COL_BAIRRO)["w_area"].sum()
        .reset_index()
        .rename(columns={"w_area":"indice_area"})
    )

    # =========================
    # POPULAÇÃO
    # =========================
    df_freq["adj_pop"] = df_freq["count"]/df_freq["Total_de_pessoas_2022"].replace(0,1)
    p95_pop = df_freq.groupby(col_categoria)["adj_pop"].transform(lambda s: s.quantile(0.95)).replace(0,1)
    df_freq["norm_pop"] = (df_freq["adj_pop"]/p95_pop).clip(0,1)
    df_freq["w_pop"] = df_freq["alpha"] * df_freq["norm_pop"]
    indice_pop = (
        df_freq.groupby(COL_BAIRRO)["w_pop"].sum()
        .reset_index()
        .rename(columns={"w_pop":"indice_pop"})
    )

    # =========================
    # DOMICÍLIOS
    # =========================
    df_freq["adj_dom"] = df_freq["count"]/df_freq["Total_de_domicilios_2022"].replace(0,1)
    p95_dom = df_freq.groupby(col_categoria)["adj_dom"].transform(lambda s: s.quantile(0.95)).replace(0,1)
    df_freq["norm_dom"] = (df_freq["adj_dom"]/p95_dom).clip(0,1)
    df_freq["w_dom"] = df_freq["alpha"] * df_freq["norm_dom"]
    indice_dom = (
        df_freq.groupby(COL_BAIRRO)["w_dom"].sum()
        .reset_index()
        .rename(columns={"w_dom":"indice_dom"})
    )

    # TOTAL
    total = (
        df_freq.groupby(COL_BAIRRO)["count"].sum()
        .reset_index()
        .rename(columns={"count":nome_col_total})
    )

    # Tabela wide de classes p/ popups
    tabela_classes = (
        df_freq.pivot_table(
            index=COL_BAIRRO,
            columns=col_categoria,
            values="count",
            aggfunc="sum",
            fill_value=0
        ).reset_index()
    )

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

    return gdf_final

# ============================================================
# FUNÇÃO DE MAPA 1746 / DD (com popups ricos)
# ============================================================
def gerar_mapa_indices(gdf, titulo, palette, prefixo_total, categoria_cols):

    centro = gdf.geometry.unary_union.centroid
    mapa = folium.Map(location=[centro.y, centro.x], zoom_start=11, tiles="CartoDB Positron")

    def add_layer(col_indice, nome_layer):
        vmin, vmax = gdf[col_indice].min(), gdf[col_indice].max()
        colormap = cm.LinearColormap(colors=palette, vmin=vmin, vmax=vmax, caption=nome_layer)

        # --- SUA LÓGICA: só População vem ligada ---
        if nome_layer.split("–")[-1].strip() == "População":
            layer = folium.FeatureGroup(name=nome_layer, show=True)
        else:
            layer = folium.FeatureGroup(name=nome_layer, show=False)

        for _, row in gdf.iterrows():
            valor = row[col_indice]
            color = colormap(valor) if valor > 0 else "#cccccc"

            # Tabela de classes
            classe_html = "".join(
                f"<tr><td>{c}</td><td>{row.get(c,0):.0f}</td></tr>"
                for c in categoria_cols
            )

            popup_html = f"""
            <div style="font-family: Arial; padding: 6px;">
                <h4 style="margin-bottom:5px;">{row[COL_BAIRRO]}</h4>

                <b>Área:</b> {row['area_km2']:.2f} km²<br>
                <b>População:</b> {row['Total_de_pessoas_2022']:.0f}<br>
                <b>Domicílios:</b> {row['Total_de_domicilios_2022']:.0f}<br>
                <b>Total:</b> {row[prefixo_total]:.0f}<br><br>

                <b>Índices (normalizações):</b>
                <ul style="margin-top:4px;">
                    <li><b>Intra-bairro:</b> {row['indice_intra']:.3f}</li>
                    <li><b>Global:</b> {row['indice_global']:.3f}</li>
                    <li><b>Área:</b> {row['indice_area']:.3f}</li>
                    <li><b>População:</b> {row['indice_pop']:.3f}</li>
                    <li><b>Domicílios:</b> {row['indice_dom']:.3f}</li>
                </ul>

                <b>Contagem por Classe:</b><br>
                <table border="1" style="width:100%; border-collapse: collapse; text-align:center;">
                    <tr style="background:#f0f0f0;">
                        <th>Classe</th>
                        <th>Qtd</th>
                    </tr>
                    {classe_html}
                </table>
            </div>
            """

            folium.GeoJson(
                data=row["geometry"],
                style_function=lambda feat, color=color: {
                    "fillColor":color,"color":"#333","weight":1,"fillOpacity":0.6
                },
                tooltip=f"{row[COL_BAIRRO]} — {valor:.3f}",
                popup=folium.Popup(popup_html, max_width=450)
            ).add_to(layer)

        colormap.add_to(mapa)
        layer.add_to(mapa)

    # Layers com lógica de "População" ativa
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
    bairros = gpd.read_file("dados/Limite_de_Bairros.geojson")
    df_1746 = pd.read_csv("dados/dados_1746.csv")
    df_dd = pd.read_parquet("dados/dados_dd.parquet")
    df_censo = pd.read_csv("dados/bairros_censo.csv")

    bairros_m = bairros.to_crs(3857)
    bairros["area_km2"] = bairros_m.geometry.area / 1e6

    bairros = bairros.merge(
        df_censo[["nome","Total_de_pessoas_2022","Total_de_domicilios_2022"]],
        on="nome", how="left"
    ).fillna(0)

    return bairros, df_1746, df_dd

bairros, df_1746_full, df_dd_full = carregar_dados()

# ============================================================
# FILTROS DE PERÍODO
# ============================================================
def filtrar_1746_periodo(opcao, df):
    if opcao == "Ano inteiro":
        return df
    mes = MAPA_MES_NOME_NUM[opcao]
    df = df.copy()
    if "data_particao" in df.columns:
        df["data_particao"] = pd.to_datetime(df["data_particao"], errors="coerce")
        df = df[
            (df["data_particao"].dt.year == 2025) &
            (df["data_particao"].dt.month == mes)
        ]
    return df

def filtrar_dd_periodo(opcao, df):
    df = df.copy()
    if "data_denuncia" in df.columns:
        df["data_denuncia"] = pd.to_datetime(df["data_denuncia"], errors="coerce")
        df = df[df["data_denuncia"].dt.year == 2025]
        if opcao != "Ano inteiro":
            mes = MAPA_MES_NOME_NUM[opcao]
            df = df[df["data_denuncia"].dt.month == mes]
    return df

def preparar_dd(df_dd_raw):
    df_exp = df_dd_raw.explode("assuntos", ignore_index=True)
    def get_classe(a):
        if isinstance(a,dict):
            return a.get("classe")
        return None
    df_exp["classe"] = df_exp["assuntos"].apply(get_classe).astype(str).str.upper()
    classes = [
        "PERTURBAÇÃO DA ORDEM PÚBLICA",
        "CRIMES CONTRA O PATRIMÔNIO",
        "SUBSTÂNCIAS ENTORPECENTES"
    ]
    return df_exp[df_exp["classe"].isin(classes)].copy()

# ============================================================
# PESOS PADRÕES
# ============================================================
pesos_default_1746 = {
    "Fiscalização de estacionamento irregular de veículo":0.17,
    "Remoção de entulho e bens inservíveis":0.17,
    "Remoção de resíduos no logradouro":0.17,
    "Reparo de Luminária":0.17,
    "Reparo de buraco, deformação ou afundamento na pista":0.16,
    "Poda de árvore em logradouro":0.16
}
pesos_default_dd = {
    "Crimes contra o patrimônio":0.33,
    "Perturbação da ordem pública":0.33,
    "Substâncias entorpecentes":0.34
}

if "pesos_1746" not in st.session_state:
    st.session_state["pesos_1746"] = pesos_default_1746.copy()
if "pesos_dd" not in st.session_state:
    st.session_state["pesos_dd"] = pesos_default_dd.copy()

if "peso_combinado_1746" not in st.session_state:
    st.session_state["peso_combinado_1746"] = 0.8
if "peso_combinado_dd" not in st.session_state:
    st.session_state["peso_combinado_dd"] = 0.2

# ============================================================
# SIDEBAR
# ============================================================
opcao_periodo = st.sidebar.selectbox("Selecione o período", MESES_OPCOES)

# ============================================================
# FORM – AJUSTE DOS PESOS
# ============================================================
with st.form("frm"):
    st.markdown("## Ajuste de Pesos")

    with st.expander("Entenda como funcionam as normalizações do índice"):
        st.markdown("""
        ### Entendendo os Tipos de Normalização do Índice

        Para comparar bairros de forma justa, não basta olhar apenas para o número bruto de ocorrências.  
        Bairros têm **tamanhos diferentes**, **populações diferentes**, **características internas diferentes**.  
        Por isso usamos **cinco normalizações**, cada uma respondendo a uma pergunta específica.

        ---

        ## **1. Normalização Intra-Bairro (“Intra”)**
        **O que ela mede?**  
        Mostra **como cada tipo de ocorrência se destaca *dentro* do próprio bairro**.

        **Como funciona?**  
        Compara as categorias apenas dentro daquele bairro.  
        A categoria mais frequente tende a ficar próxima de 1; a menos frequente tende a ficar próxima de 0.

        **Para que serve?**  
        Revela **o perfil interno** do bairro e quais problemas são mais representativos ali.

        ---

        ## **2. Normalização Global**
        **O que ela mede?**  
        Compara **todos os bairros entre si**, categoria por categoria.

        **Como funciona?**  
        Ajusta os valores de forma que bairros muito acima da média não distorçam a escala.  
        Cada categoria é normalizada separadamente para permitir comparações equilibradas.

        **Para que serve?**  
        Mostra a **intensidade relativa** de cada categoria de ocorrência quando observada em toda a cidade.

        ---

        ## **3. Normalização por Área**
        **O que ela mede?**  
        A **quantidade de ocorrências por quilômetro quadrado**.

        **Como funciona?**  
        Divide as ocorrências pela área do bairro.  
        Assim, bairros grandes não parecem ter mais ocorrências só por serem extensos.

        **Para que serve?**  
        Identificar onde os problemas são mais **concentrados espacialmente**.

        ---

        ## **4. Normalização por População**
        **O que ela mede?**  
        A quantidade de ocorrências **por habitante**.

        **Como funciona?**  
        Divide as ocorrências pela população total do bairro.

        **Para que serve?**  
        Revela se um bairro é proporcionalmente mais afetado **por pessoa**, indo além da contagem absoluta.

        ---

        ## **5. Normalização por Domicílios**
        **O que ela mede?**  
        A quantidade de ocorrências **por residência**.

        **Como funciona?**  
        Divide as ocorrências pelo número de domicílios do bairro.

        **Para que serve?**  
        Mostra o impacto **na malha residencial**, especialmente útil em áreas com grande densidade habitacional.

        ---

        ## **Resumo Geral**
        Cada normalização responde a uma pergunta diferente:

        | Normalização | Pergunta respondida |
        |--------------|---------------------|
        | **Intra** | O que mais pesa **dentro do bairro**? |
        | **Global** | Como o bairro se compara com **o restante da cidade**? |
        | **Área** | Onde os problemas são mais **concentrados por km²**? |
        | **População** | Onde há mais impacto **por habitante**? |
        | **Domicílios** | Onde há mais impacto **por residência**? |

        Ao combinar essas visões, criamos um índice mais justo, completo e fiel à realidade urbana.
            """)


    c1, c2 = st.columns(2)

    novos_1746, novos_dd = {}, {}

    with c1:
        st.subheader("Pesos 1746")
        for cat,v in st.session_state["pesos_1746"].items():
            novos_1746[cat] = st.number_input(cat, value=float(v), step=0.01, key=f"p1746_{cat}")

    with c2:
        st.subheader("Pesos Disque Denúncia")
        for cat,v in st.session_state["pesos_dd"].items():
            novos_dd[cat] = st.number_input(cat, value=float(v), step=0.01, key=f"pdd_{cat}")

    st.subheader("Pesos do Índice Combinado")
    novo_peso_1746 = st.number_input("Peso 1746", value=float(st.session_state["peso_combinado_1746"]), step=0.05)
    novo_peso_dd   = st.number_input("Peso Disque Denúncia", value=float(st.session_state["peso_combinado_dd"]), step=0.05)

    submitted = st.form_submit_button("Recalcular")

if submitted:
    st.session_state["pesos_1746"] = novos_1746
    st.session_state["pesos_dd"] = novos_dd
    st.session_state["peso_combinado_1746"] = novo_peso_1746
    st.session_state["peso_combinado_dd"]   = novo_peso_dd

# ============================================================
# FILTRAR / CALCULAR
# ============================================================
df1746 = filtrar_1746_periodo(opcao_periodo, df_1746_full)
dfdd_raw = filtrar_dd_periodo(opcao_periodo, df_dd_full)
dfdd = preparar_dd(dfdd_raw)

gdf1746 = construir_indices(
    df_raw=df1746,
    col_categoria="subtipo",
    pesos_custom=st.session_state["pesos_1746"],
    gdf_bairros=bairros,
    nome_col_total="total_1746"
)

gdfdd = construir_indices(
    df_raw=dfdd,
    col_categoria="classe",
    pesos_custom=st.session_state["pesos_dd"],
    gdf_bairros=bairros,
    nome_col_total="total_dd"
)

# colunas de classes para popups
pesos_1746_upper = [k.upper() for k in st.session_state["pesos_1746"].keys()]
pesos_dd_upper   = [k.upper() for k in st.session_state["pesos_dd"].keys()]

class_cols_1746 = [c for c in gdf1746.columns if c in pesos_1746_upper]
class_cols_dd   = [c for c in gdfdd.columns   if c in pesos_dd_upper]

# ============================================================
# ÍNDICE COMBINADO
# ============================================================
peso1746 = st.session_state["peso_combinado_1746"]
pesoDD   = st.session_state["peso_combinado_dd"]

gdf_comb = bairros.copy()

for idx in ["intra","global","area","pop","dom"]:
    gdf_comb[f"indice_1746_{idx}"] = gdf1746[f"indice_{idx}"]
    gdf_comb[f"indice_dd_{idx}"]   = gdfdd[f"indice_{idx}"]
    gdf_comb[f"indice_{idx}"] = (
        gdf_comb[f"indice_1746_{idx}"] * peso1746 +
        gdf_comb[f"indice_dd_{idx}"]   * pesoDD
    )

gdf_comb["total_comb"] = (
    gdf1746["total_1746"] * peso1746 +
    gdfdd["total_dd"]     * pesoDD
)

# ============================================================
# MAPAS 1746 E DD
# ============================================================
mapa_1746 = gerar_mapa_indices(
    gdf1746,
    titulo="1746",
    palette=["#eff3ff","#bdd7e7","#6baed6","#3182bd","#08519c"],
    prefixo_total="total_1746",
    categoria_cols=class_cols_1746
)

mapa_dd = gerar_mapa_indices(
    gdfdd,
    titulo="Disque Denúncia",
    palette=["#ffffcc","#ffeda0","#feb24c","#f03b20","#bd0026"],
    prefixo_total="total_dd",
    categoria_cols=class_cols_dd
)

# ============================================================
# MAPA COMBINADO (mantendo sua lógica de População ligada)
# ============================================================
def gerar_mapa_combinado(gdf):

    centro = gdf.geometry.unary_union.centroid
    mapa = folium.Map(location=[centro.y, centro.x], zoom_start=11, tiles="CartoDB Positron")

    palette = ["#f2e5ff", "#d1b3ff", "#b080ff", "#8c4dff", "#661aff"]

    def add_layer(idx, nome):
        vmin, vmax = gdf[f"indice_{idx}"].min(), gdf[f"indice_{idx}"].max()
        colormap = cm.LinearColormap(colors=palette, vmin=vmin, vmax=vmax, caption=nome)

        # mesma lógica do nome_layer.split("–")...
        if nome.split("–")[-1].strip() == "População":
            layer = folium.FeatureGroup(name=nome, show=True)
        else:
            layer = folium.FeatureGroup(name=nome, show=False)

        for _, row in gdf.iterrows():
            valor = row[f"indice_{idx}"]
            color = colormap(valor) if valor > 0 else "#cccccc"

            popup = f"""
            <div style="font-family: Arial; padding: 6px;">
                <h4 style="margin-bottom:5px;">{row[COL_BAIRRO]}</h4>
                <b>Índice combinado ({idx}):</b> {valor:.3f}<br><br>
                <b>1746 ({idx}):</b> {row[f'indice_1746_{idx}']:.3f}<br>
                <b>DD ({idx}):</b> {row[f'indice_dd_{idx}']:.3f}<br>
                <b>Pesos combinação:</b> 1746={peso1746:.2f} | DD={pesoDD:.2f}<br>
            </div>
            """

            folium.GeoJson(
                data=row["geometry"],
                style_function=lambda feat, color=color: {
                    "fillColor":color,"color":"#333","weight":1,"fillOpacity":0.6
                },
                tooltip=f"{row[COL_BAIRRO]} — {valor:.3f}",
                popup=folium.Popup(popup, max_width=400)
            ).add_to(layer)

        colormap.add_to(mapa)
        layer.add_to(mapa)

    add_layer("intra",  "Combinado – Intra")
    add_layer("global", "Combinado – Global")
    add_layer("area",   "Combinado – Área")
    add_layer("pop",    "Combinado – População")
    add_layer("dom",    "Combinado – Domicílios")

    folium.LayerControl(collapsed=False).add_to(mapa)
    return mapa

mapa_comb = gerar_mapa_combinado(gdf_comb)

# ============================================================
# EXIBIÇÃO
# ============================================================
st.markdown("## Mapas de Índice")

st.subheader("Mapa 1746")
html(mapa_1746._repr_html_(), height=800)

st.subheader("Mapa Disque Denúncia")
html(mapa_dd._repr_html_(), height=800)

st.subheader("Mapa Combinado")
html(mapa_comb._repr_html_(), height=800)
