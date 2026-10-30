from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Segmentación de clientes", layout="wide")

CARPETA = Path(__file__).parent
NUM = ['ingreso_anual', 'gasto_total', 'compras_tienda', 'compras_web', 'compras_catalogo',
       'visitas_web_mes', 'recencia_dias', 'hijos_pequenos', 'edad']
CAT = ['educacion', 'estado_civil']


def limpiar(d):
    d.columns = [c.split('#')[-1] for c in d.columns]
    # Orange guarda 3 filas de encabezado: nombres, tipos y banderas (meta/class)
    if len(d) > 2 and d.iloc[0].astype(str).isin(['continuous', 'discrete', 'string', 'time']).any():
        d = d.iloc[2:].reset_index(drop=True)
    return d


@st.cache_data
def cargar():
    return limpiar(pd.read_csv(CARPETA / 'clientes_retail.csv'))


def con_grupo(base, d, col):
    g = d[col].astype(str).rename('grupo')
    if 'id_cliente' in d:
        ids = pd.to_numeric(d['id_cliente']).astype(int)
        return base.merge(pd.DataFrame({'id_cliente': ids, 'grupo': g}), on='id_cliente')
    return base.iloc[:len(d)].assign(grupo=g.values)


def perfil(m):
    p = m.groupby('grupo').agg(clientes=('id_cliente', 'size'), gasto_promedio=('gasto_total', 'mean'),
                               gasto=('gasto_total', 'sum'), respondio=('respondio_ultima_campana', 'mean'))
    p['% clientes'] = 100 * p.clientes / p.clientes.sum()
    p['% gasto'] = 100 * p.gasto / p.gasto.sum()
    p['% respondió'] = 100 * p.respondio
    return p[['clientes', '% clientes', 'gasto_promedio', '% gasto', '% respondió']].round(1)


@st.cache_data(show_spinner="Ajustando las cinco técnicas...")
def ajustar(k, eps, min_samples):
    c = cargar()
    Z = StandardScaler().fit_transform(c[NUM])
    X = np.column_stack([Z, c[CAT].values]).astype(object)
    etiquetas = {
        'k-means': KMeans(k, n_init=10, random_state=0).fit_predict(Z),
        'jerárquico': AgglomerativeClustering(k, linkage='ward').fit_predict(Z),
        'DBSCAN': DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z),
        'mezcla gaussiana': GaussianMixture(k, random_state=0, n_init=5).fit_predict(Z),
        'k-prototypes': KPrototypes(n_clusters=k, init='Cao', random_state=0, n_init=3)
                        .fit_predict(X, categorical=[len(NUM), len(NUM) + 1]),
    }
    return Z, pd.DataFrame(etiquetas)


def resumen(Z, et):
    filas = []
    for t, y in et.items():
        ok = y != -1
        grupos = sorted(set(y[ok]))
        sil = silhouette_score(Z[ok], y[ok]) if len(grupos) > 1 else np.nan
        tam = pd.Series(y[ok]).value_counts().sort_index()
        filas.append({'técnica': t, 'grupos': len(grupos), 'ruido': int((~ok).sum()),
                      'tamaños': ' / '.join(map(str, tam.values)), 'silueta': round(sil, 3)})
    return pd.DataFrame(filas).set_index('técnica')


base = cargar()
st.title("Segmentación de clientes")
visor, comparador = st.tabs(["Visor de grupos", "Comparador de técnicas"])

with visor:
    archivo = st.file_uploader("CSV con una columna de grupos (Orange o notebook)", type='csv')
    ejemplos = [p for p in ['clientes_con_cluster.csv', 'segmentos_notebook.csv'] if (CARPETA / p).exists()]
    if archivo is not None:
        d = limpiar(pd.read_csv(archivo))
    elif ejemplos:
        nombre = st.selectbox("Sin archivo: usar ejemplo", ejemplos)
        d = limpiar(pd.read_csv(CARPETA / nombre))
    else:
        st.info("Suba un CSV para empezar.")
        st.stop()

    candidatas = [c for c in d.columns if d[c].nunique() <= 20 and c not in NUM]
    col = st.selectbox("Columna de grupos", candidatas,
                       index=candidatas.index('Cluster') if 'Cluster' in candidatas else 0)
    m = con_grupo(base, d, col)

    p = perfil(m)
    a, b = st.columns(2)
    a.plotly_chart(px.bar(p.reset_index(), x='grupo', y='clientes', color='grupo', text='clientes',
                          title="Clientes por grupo"), width='stretch')
    b.plotly_chart(px.bar(p.reset_index(), x='grupo', y='% gasto', color='grupo', text='% gasto',
                          title="% del gasto total"), width='stretch')
    st.dataframe(p, width='stretch')

    x = st.selectbox("Eje x", NUM, index=0)
    y = st.selectbox("Eje y", NUM, index=1)
    st.plotly_chart(px.scatter(m, x=x, y=y, color='grupo', opacity=0.6), width='stretch')

with comparador:
    a, b, c = st.columns(3)
    k = a.slider("Número de grupos (k)", 2, 8, 3)
    eps = b.slider("eps de DBSCAN", 0.5, 4.0, 1.5, 0.1)
    ms = c.slider("Mínimo de vecinos (DBSCAN)", 3, 30, 5)
    Z, et = ajustar(k, eps, ms)

    st.subheader("Resumen")
    st.dataframe(resumen(Z, et), width='stretch')
    st.caption("Silueta sobre las 9 variables numéricas estandarizadas; en DBSCAN sin contar el ruido (-1).")

    st.subheader("¿Qué tanto coinciden? (índice de Rand ajustado, 1 = idénticas)")
    ari = pd.DataFrame([[adjusted_rand_score(et[i], et[j]) for j in et] for i in et], index=et.columns, columns=et.columns)
    st.plotly_chart(px.imshow(ari.round(2), text_auto=True, zmin=-0.1, zmax=1, color_continuous_scale='Blues'),
                    width='stretch')

    a, b = st.columns(2)
    t1 = a.selectbox("Técnica A", et.columns, index=0)
    t2 = b.selectbox("Técnica B", et.columns, index=4)
    st.dataframe(pd.crosstab(et[t1], et[t2], rownames=[t1], colnames=[t2]), width='stretch')

    t = st.selectbox("Ver grupos de", et.columns)
    vista = base.assign(grupo=et[t].astype(str))
    st.plotly_chart(px.scatter(vista, x='ingreso_anual', y='gasto_total', color='grupo', opacity=0.6),
                    width='stretch')
    st.dataframe(perfil(vista), width='stretch')

    st.download_button("Descargar asignaciones (CSV)", base[['id_cliente']].join(et).to_csv(index=False),
                       "asignaciones.csv", "text/csv")
