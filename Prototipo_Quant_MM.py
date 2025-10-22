import streamlit as st
import numpy as np
import pandas as pd
import openpyxl
import requests
from io import BytesIO
from datetime import date
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import math

# =============================== Configurações Streamlit ===============================
st.set_page_config(
    page_title="Quantitative Finance",
    page_icon="📈",
    layout="wide"
)
st.set_option('deprecation.showPyplotGlobalUse', False)

# =============================== Variáveis Globais =====================================
datafim = date.today()
datainicio = '2020-01-01'

lineColor = '#00CCCC'
MMlineColor = '#00FF00'
LineVol = MMlineColor
horizontalLineColor = '#CCCCFF'

dotColor1 = '#4C0099'
dotColor2 = '#CC0066'
dotColor3 = '#FFB266'

chartwidth = 800
chartheight = 225

# =============================== Funções de Carregamento ================================
@st.cache_data()
def carregar_excel(sheet_name):
    url = "https://raw.githubusercontent.com/diogoraucci/Prototipo_Quant_MM/main/DF_FormulaMagica.xlsx"
    response = requests.get(url)
    conteudo_excel = response.content
    df = pd.read_excel(BytesIO(conteudo_excel), sheet_name=sheet_name, index_col=0, engine='openpyxl')
    return df

@st.cache_data()
def carregar_descricao():
    return carregar_excel('descricaoBR')

@st.cache_data()
def carregar_rankFM():
    return carregar_excel('FM_acoesbancos')

@st.cache_data()
def carregar_resultado():
    df_resultadosAnual = carregar_excel('df_resultadoAnualBR')
    df_resultadosTrim = carregar_excel('df_resultadoTrimBR')
    return df_resultadosAnual, df_resultadosTrim

@st.cache_data()
def carregar_dataset():
    return carregar_excel('CompletoBR')

@st.cache_data()
def carregar_cotacoes():
    return carregar_excel('cotacoesBR')

# =============================== Carregando Dados =======================================
df_descricao = carregar_descricao()
df_rankFM = carregar_rankFM()
df_resultadosAnual, df_resultadosTrim = carregar_resultado()
df_dataset = carregar_dataset()
df_cotacoes = carregar_cotacoes()

# =============================== Função para Perfis de Investimento ===================
def calcular_residuos(ts, select_tickers, select_MM):
    # Média móvel
    ts['MM'] = ts[select_tickers].rolling(select_MM).mean()
    ts.fillna(method='bfill', inplace=True)
    
    # Retorno log da média móvel
    ts['mm%'] = ts.apply(lambda x: math.log(x[0] / x[1]), axis=1)
    ts['pct'] = np.log(ts.iloc[:, [0]].pct_change() + 1)
    ts['pct'].fillna(method='bfill', inplace=True)
    
    # Regressão linear
    X = ts['pct'].values.reshape(-1, 1)
    Y = ts['mm%'].values.reshape(-1, 1)
    reg = LinearRegression().fit(X, Y)
    ts['Resíduos'] = Y - reg.predict(X)
    
    # Estatísticas dos resíduos
    mean = ts['Resíduos'].mean()
    std = ts['Resíduos'].std()
    ts['zero'] = mean
    ts['1std+'], ts['1std-'] = std, -std
    ts['2std+'], ts['2std-'] = std*2, -std*2
    ts['3std+'], ts['3std-'] = std*3, -std*3
    
    # Sinalizando entradas
    ts['acima_3std'] = ts['Resíduos'] >= ts['3std+']
    ts['acima_2std'] = (ts['Resíduos'] >= ts['2std+']) & (ts['Resíduos'] < ts['3std+'])
    ts['acima_1std'] = (ts['Resíduos'] >= ts['1std+']) & (ts['Resíduos'] < ts['2std+'])
    ts['abaixo_3std'] = ts['Resíduos'] <= ts['3std-']
    ts['abaixo_2std'] = (ts['Resíduos'] <= ts['2std-']) & (ts['Resíduos'] > ts['3std-'])
    ts['abaixo_1std'] = (ts['Resíduos'] <= ts['1std-']) & (ts['Resíduos'] > ts['2std-'])
    
    return ts

def plot_cotacoes(ts, select_tickers, select_MM):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts[select_tickers], name=select_tickers, mode='lines', line=dict(color=lineColor, width=2)))
    fig.add_trace(go.Scatter(x=ts.index, y=ts['MM'], name=f'MM {select_MM}', mode='lines', line=dict(color=MMlineColor, width=2)))
    
    # Pontos
    for std_level, color, size in [(1, dotColor1, 5), (2, dotColor2, 10), (3, dotColor3, 15)]:
        fig.add_trace(go.Scatter(x=ts[ts[f'abaixo_{std_level}std']].index,
                                 y=ts.loc[ts[f'abaixo_{std_level}std'], select_tickers],
                                 mode='markers', marker=dict(color=color, size=size)))
        fig.add_trace(go.Scatter(x=ts[ts[f'acima_{std_level}std']].index,
                                 y=ts.loc[ts[f'acima_{std_level}std'], select_tickers],
                                 mode='markers', marker=dict(color=color, size=size)))
    
    fig.update_layout(title=f'Cotações de {select_tickers} Média Móvel de {select_MM} períodos',
                      legend=dict(x=0, y=1.0, orientation='h', traceorder='normal'),
                      autosize=True, height=500)
    
    # Eixo X
    num_dates = 5
    tick_values = ts.index[::max(1, len(ts.index) // num_dates)]
    tick_values = [ts.index[0]] + tick_values.tolist() + [ts.index[-1]]
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=tick_values, tickangle=0))
    
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

def plot_residuos(ts, select_tickers):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts['Resíduos'], mode='lines', line=dict(color=lineColor, width=2)))
    fig.add_trace(go.Scatter(x=ts.index, y=ts['zero'], mode='lines', line=dict(color=MMlineColor, width=1)))
    
    for std_level, color, size in [(1, dotColor1, 5), (2, dotColor2, 10), (3, dotColor3, 15)]:
        fig.add_trace(go.Scatter(x=ts[ts[f'abaixo_{std_level}std']].index,
                                 y=ts.loc[ts[f'abaixo_{std_level}std'], 'Resíduos'],
                                 mode='markers', marker=dict(color=color, size=size)))
        fig.add_trace(go.Scatter(x=ts[ts[f'acima_{std_level}std']].index,
                                 y=ts.loc[ts[f'acima_{std_level}std'], 'Resíduos'],
                                 mode='markers', marker=dict(color=color, size=size)))
    
    fig.update_layout(title=f'Gráfico Normalizado {select_tickers}',
                      showlegend=False,
                      yaxis=dict(showticklabels=False, showgrid=False, showline=False, zeroline=False),
                      autosize=True, height=400)
    
    num_dates = 5
    tick_values = ts.index[::max(1, len(ts.index) // num_dates)]
    tick_values = [ts.index[0]] + tick_values.tolist() + [ts.index[-1]]
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=tick_values, tickangle=0))
    
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# =============================== Perfil de Investimento ===============================
def main_perfil(select_MM, select_tickers):
    ts = pd.DataFrame(df_cotacoes[select_tickers]).copy()
    ts = calcular_residuos(ts, select_tickers, select_MM)
    plot_cotacoes(ts, select_tickers, select_MM)
    plot_residuos(ts, select_tickers)

# =============================== Página Streamlit =====================================
st.title("📊 Quantitative Finance - Fórmula Mágica BR")
select_tickers = st.selectbox("Selecione o ticker", df_cotacoes.columns)
select_MM = st.slider("Período da Média Móvel", min_value=2, max_value=60, value=20)

main_perfil(select_MM, select_tickers)
