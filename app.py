"""Classificação de países por prioridade de ajuda."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Classificação de países", page_icon="🌍", layout="wide")

@st.cache_resource
def carregar_pacote():
    return joblib.load(Path(__file__).with_name("modelo_paises.pkl"))

pacote = carregar_pacote()
modelo = pacote["modelo"]
colunas = pacote["colunas"]
nomes = pacote["nomes"]
pca = modelo.named_steps["pca"]
clusterizador = modelo.steps[-1][1]

CAMPOS = {
    "child_mort": ("Mortalidade infantil (por 1.000 nascidos vivos)", 19.3),
    "exports": ("Exportações (% do PIB)", 35.0),
    "health": ("Gasto com saúde (% do PIB)", 6.3),
    "imports": ("Importações (% do PIB)", 43.3),
    "income": ("Renda líquida por pessoa (US$)", 9960.0),
    "inflation": ("Inflação anual (%)", 5.4),
    "life_expec": ("Expectativa de vida (anos)", 73.1),
    "total_fer": ("Filhos por mulher", 2.4),
    "gdpp": ("PIB per capita (US$)", 4660.0),
}

st.title("🌍 Classificação de países")
st.write("Informe os indicadores para estimar o grupo e comparar o perfil do país com o centro do grupo atribuído.")

with st.form("formulario_pais"):
    valores = {}
    esquerda, direita = st.columns(2)
    for i, coluna in enumerate(colunas):
        rotulo, inicial = CAMPOS.get(coluna, (coluna.replace("_", " ").title(), 0.0))
        with (esquerda if i % 2 == 0 else direita):
            valores[coluna] = st.number_input(rotulo, value=float(inicial), key=coluna)
    classificar = st.form_submit_button("Classificar país", type="primary")

if classificar:
    entrada = pd.DataFrame([valores], columns=colunas)
    grupo = int(modelo.predict(entrada)[0])
    nome_grupo = nomes[grupo] if isinstance(nomes, (list, tuple, np.ndarray)) else nomes.get(grupo, f"Grupo {grupo}")
    st.success(f"Grupo {grupo}: {nome_grupo}")

    # O pipeline reduz os dados a cinco dimensões; a projeção mostra apenas PC1 e PC2.
    projetado = modelo[:-1].transform(entrada)
    centros = clusterizador.cluster_centers_
    perfil_reduzido = centros[grupo:grupo + 1]
    # Reconstrução aproximada dos valores originais a partir dos cinco componentes retidos.
    perfil_padronizado = pca.inverse_transform(perfil_reduzido)
    scaler = modelo.steps[0][1]
    perfil_original = scaler.inverse_transform(perfil_padronizado)[0]

    st.subheader("Perfil médio do grupo")
    st.caption("Centro do grupo reconstruído pelo pipeline a partir dos cinco componentes PCA retidos; os valores são aproximados.")
    comparacao = pd.DataFrame({
        "Indicador": [CAMPOS.get(c, (c, 0))[0] for c in colunas],
        "País informado": [valores[c] for c in colunas],
        "Perfil médio estimado do grupo": perfil_original,
    }).set_index("Indicador")
    st.dataframe(comparacao.style.format({"País informado": "{:,.2f}", "Perfil médio estimado do grupo": "{:,.2f}"}), use_container_width=True)

    st.subheader("País no plano PC1 × PC2")
    pontos = pd.DataFrame(modelo[:-1].transform(entrada), columns=[f"PC{i+1}" for i in range(projetado.shape[1])])
    pontos["Tipo"] = "País informado"
    centros_df = pd.DataFrame(centros, columns=[f"PC{i+1}" for i in range(centros.shape[1])])
    centros_df["Tipo"] = [f"Centro do grupo {i}" for i in range(len(centros_df))]
    plano = pd.concat([pontos, centros_df], ignore_index=True)
    st.scatter_chart(plano, x="PC1", y="PC2", color="Tipo", size=100)

with st.expander("Sobre o modelo"):
    st.markdown("""
    **Objetivo:** agrupar países por indicadores socioeconômicos para apoiar a priorização de ajuda.

    **Origem dos dados:** conjunto [`Country-data.csv` no Kaggle](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data), associado ao estudo de agrupamento de países para a HELP International. Os arquivos fornecidos não identificam uma versão específica da base.

    **Data de treinamento:** não registrada nos arquivos do modelo. A data de modificação do arquivo não comprova quando o treinamento ocorreu.
    """)
    st.write(f"- Componentes do PCA retidos: **{pca.n_components_}** (a visualização usa somente PC1 e PC2).")
    st.write(f"- Número de grupos: **{clusterizador.n_clusters}**.")
    st.caption("Esta é uma documentação inicial do modelo; os indicadores e a projeção são estimativas do agrupamento, não uma recomendação oficial de política pública.")


