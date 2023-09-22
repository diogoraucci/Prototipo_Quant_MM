import streamlit as st

import numpy as np
import pandas as pd
import openpyxl
import requests
from io import BytesIO
from io import StringIO

from datetime import date

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import altair as alt

from sklearn.linear_model import LinearRegression

from math import sqrt
import math

# Definindo layout ===================================
st.set_page_config(page_title="Quantitative Finance",
        page_icon="chart_with_upwards_trend",
        layout="wide",)

# Set the theme to dark mode =========================

# Remover Warning Mensage =============================
st.set_option('deprecation.showPyplotGlobalUse', False)
#======================================================

# Definir Variaveis

datafim = date.today() #'2023-06-01'
datainicio = '2020-01-01'

size_ChartBar = (10, 8)
font_title2 = 50
font_bar = 35
font_label = 40

lineColor = '#00CCCC'
MMlineColor = '#00FF00'
LineVol = MMlineColor
horizontalLineColor = '#CCCCFF'

dotColor1 = '#4C0099'
dotColor2 = '#CC0066'
dotColor3 = '#FFB266'

colorUp = lineColor
colorDown = '#CC0066'

chartwidth  = 800  # Aumentar a largura do gráfico de barras
chartheight = 225  # Aumentar a altura do gráfico de barras

# Definir Funçoes =====================================
@st.cache_data()
def carregar_descricao():
    # URL do arquivo Excel no GitHub
    url = "https://raw.githubusercontent.com/diogoraucci/Prototipo_Quant_MM/main/DF_Descricao.xlsx"
    # Faz o download do conteúdo do arquivo
    response = requests.get(url)
    conteudo_excel = response.content
    # Lê o conteúdo baixado como um arquivo Excel usando BytesIO
    df_descricao = pd.read_excel(BytesIO(conteudo_excel), sheet_name='DescricaoEmpresas_BR', index_col=0, engine='openpyxl')
    return df_descricao

@st.cache_data()
def carregar_rankFM():
    # URL do arquivo Excel no GitHub
    url = "https://raw.githubusercontent.com/diogoraucci/Prototipo_Quant_MM/main/DF_RankScoreFM.xlsx"
    # Faz o download do conteúdo do arquivo
    response = requests.get(url)
    conteudo_excel = response.content
    # Lê o conteúdo baixado como um arquivo Excel usando BytesIO
    df_rankFM = pd.read_excel(BytesIO(conteudo_excel), sheet_name='FM_RankScore', index_col=0, engine='openpyxl')
    return df_rankFM

@st.cache_data()
def carregar_resultado():
    # URL do arquivo Excel no GitHub
    url = "https://raw.githubusercontent.com/diogoraucci/Prototipo_Quant_MM/main/DF_Resultados.xlsx"
    # Faz o download do conteúdo do arquivo
    response = requests.get(url)
    conteudo_excel = response.content
    # Lê o conteúdo baixado como um arquivo Excel usando BytesIO
    df_resultadosAnual = pd.read_excel(BytesIO(conteudo_excel), sheet_name='ResAnualBR', index_col=0, engine='openpyxl')
    df_resultadosTrim = pd.read_excel(BytesIO(conteudo_excel), sheet_name='ResTrimBR', index_col=0, engine='openpyxl')
    return df_resultadosAnual, df_resultadosTrim

@st.cache_data ()
def carregar_dataset():
    # URL do arquivo Excel no GitHub
    url = "https://raw.githubusercontent.com/diogoraucci/Prototipo_Quant_MM/main/DF_completo.xlsx"
    # Faz o download do conteúdo do arquivo
    response = requests.get(url)
    conteudo_excel = response.content
    # Lê o conteúdo baixado como um arquivo Excel usando BytesIO
    df_dataset = pd.read_excel(BytesIO(conteudo_excel), sheet_name='DataSet', index_col=0, engine='openpyxl')
    # Verifique o DataFrame
    return df_dataset

@st.cache_data ()
def carregar_cotacoes():
     # URL do arquivo Excel no GitHub
    url = "https://raw.githubusercontent.com/diogoraucci/Prototipo_Quant_MM/main/DF_completo.xlsx"
    # Faz o download do conteúdo do arquivo
    response = requests.get(url)
    conteudo_excel = response.content
    # Lê o conteúdo baixado como um arquivo Excel usando BytesIO
    df_cotacoes = pd.read_excel(BytesIO(conteudo_excel), sheet_name='Cotacoes', index_col=0, engine='openpyxl')
    # Verifique o DataFrame
    return df_cotacoes

# Definir Gráfico por perfil de investimento
def main_conservador(select_MM, select_tickers):
    # Coleta das cotações
    ts = pd.DataFrame(df_cotacoes.loc[:, select_tickers]).copy()

    # Cálculo da média móvel
    ts['MM'] = ts[select_tickers].rolling(select_MM).mean()
    ts.fillna(method='bfill', inplace=True)

    # Calcular Retorno Logaritmico da Média Móvel
    ts['mm%'] = ts.apply(lambda x: math.log(x[0] / x[1]), axis=1)  # /mm

    # Calcuular Variação Diária
    ts['pct'] = np.log(ts.iloc[:, [0]].pct_change() + 1)
    ts['pct'].fillna(method='bfill', inplace=True)

    # REGRESSÃO LINEAR
    X_independent = ts['pct'].values.reshape(-1, 1)
    Y_dependent = ts['mm%'].values.reshape(-1, 1)

    reg = LinearRegression().fit(X_independent, Y_dependent)

    # Gerando Reta da regressao------------------------------------------------------->>>>>>>>>>>
    Y_predict = reg.predict(X_independent);
    # Calculando residuos
    ts['Resíduos'] = (Y_dependent - Y_predict)

    # Gráfico dos Resíduos
    mean = ts['Resíduos'].mean()
    std = ts['Resíduos'].std()

    ts['1std+'] = std
    ts['1std-'] = std * -1
    ts['2std+'] = std * 2
    ts['2std-'] = std * -2
    ts['3std+'] = std * 3
    ts['3std-'] = std * -3
    ts['zero'] = mean

    # Plot do gráfico Precos
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts[select_tickers], name=select_tickers, mode='lines',
                             line=dict(color=lineColor, width=2)))
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['MM'], name=f'Média Móvel {select_MM} períodos', mode='lines',
                   line=dict(color=MMlineColor, width=2)))
    # Incluir Pontos

    fig.add_trace(go.Scatter(
        x=ts[np.logical_and(ts['Resíduos'] <= ts['1std-'], ts['Resíduos'] > ts['2std-'])].index,
        y=ts.loc[np.logical_and(ts['Resíduos'] <= ts['1std-'], ts['Resíduos'] > ts['2std-']), select_tickers],
        mode='markers', marker=dict(color=dotColor1, size=5), showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=ts[np.logical_and(ts['Resíduos'] <= ts['2std-'], ts['Resíduos'] > ts['3std-'])].index,
        y=ts.loc[np.logical_and(ts['Resíduos'] <= ts['2std-'], ts['Resíduos'] > ts['3std-']), select_tickers],
        mode='markers', marker=dict(color=dotColor2, size=10), showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=ts[np.logical_and(ts['Resíduos'] <= ts['3std-'], ts['Resíduos'] <= ts['3std-'])].index,
        y=ts.loc[np.logical_and(ts['Resíduos'] <= ts['3std-'], ts['Resíduos'] <= ts['3std-']), select_tickers],
        mode='markers', marker=dict(color=dotColor3, size=15), showlegend=False
    ))

    fig.update_layout(title=f'Cotações de {select_tickers} Média Móvel de {select_MM} períodos')

    # Define a legenda na parte interna do gráfico
    fig.update_layout(legend=dict(x=0, y=1.0, orientation='h', traceorder='normal'), autosize=True,
                      height=500)

    # Exibir 8 datas no eixo x
    num_dates = 5
    tick_values = ts.index[::max(1, len(ts.index) // num_dates)]
    tick_values = [ts.index[0]] + tick_values.tolist() + [
        ts.index[-1]]  # Adiciona o primeiro e o último valor do DataFrame
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=tick_values, tickangle=0))

    # Exibição do gráfico no Streamlit com largura de 100% e altura igual a 50% da largura
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    # Sinalizando Entradas ============================================================================
    # VENDAS
    # Criando uma nova coluna que indica se o valor é maior que o limite superior de 3 desvios padrão
    ts['acima_3std'] = ts['Resíduos'] >= ts['3std+']
    ts['acima_2std'] = (ts['Resíduos'] >= ts['2std+']) & (ts['Resíduos'] < ts['3std+'])
    ts['acima_1std'] = (ts['Resíduos'] >= ts['1std+']) & (ts['Resíduos'] < ts['2std+'])
    # COMPRAS
    ts['abaixo_3std'] = ts['Resíduos'] <= ts['3std-']
    ts['abaixo_2std'] = (ts['Resíduos'] <= ts['2std-']) & (ts['Resíduos'] > ts['3std-'])
    ts['abaixo_1std'] = (ts['Resíduos'] <= ts['1std-']) & (ts['Resíduos'] > ts['2std-'])

    # ==================================================
    # Plot do gráfico de Resíduos
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['Resíduos'], name='Resíduos', mode='lines',
                   line=dict(color=lineColor, width=2)))
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['zero'], mode='lines', line=dict(color=MMlineColor, width=1, dash='solid')))

    fig.add_trace(go.Scatter(x=ts.index, y=ts['1std+'], mode='lines',
                             line=dict(color=horizontalLineColor, width=1.3, dash='dot')))
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['2std+'], mode='lines',
                   line=dict(color=horizontalLineColor, width=0.5, dash='dot')))
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['3std+'], mode='lines',
                   line=dict(color=horizontalLineColor, width=0.2, dash='dot')))
    fig.add_trace(go.Scatter(x=ts.index, y=ts['1std-'], mode='lines',
                             line=dict(color=horizontalLineColor, width=1.3, dash='dot')))
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['2std-'], mode='lines',
                   line=dict(color=horizontalLineColor, width=0.5, dash='dot')))
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['3std-'], mode='lines',
                   line=dict(color=horizontalLineColor, width=0.2, dash='dot')))


    fig.add_trace(go.Scatter(x=ts[ts['abaixo_1std']].index, y=ts.loc[ts['abaixo_1std'], 'Resíduos'], mode='markers',
                             marker=dict(color=dotColor1, size=5)))

    fig.add_trace(go.Scatter(x=ts[ts['abaixo_2std']].index, y=ts.loc[ts['abaixo_2std'], 'Resíduos'], mode='markers',
                             marker=dict(color=dotColor2, size=10)))

    fig.add_trace(go.Scatter(x=ts[ts['abaixo_3std']].index, y=ts.loc[ts['abaixo_3std'], 'Resíduos'], mode='markers',
                             marker=dict(color=dotColor3, size=15)))

    fig.update_layout(title=f'Gráfico Normalizado {select_tickers} Média Móvel de {select_MM} períodos')
    fig.update_layout(showlegend=False)  # Remove as legendas
    # Remover o eixo Y
    fig.update_layout(yaxis=dict(showticklabels=False, showgrid=False))
    fig.update_layout(yaxis=dict(showline=False, zeroline=False))
    # Define a legenda na parte interna do gráfico
    fig.update_layout(legend=dict(x=0, y=1.1, orientation='h', traceorder='normal'), autosize=True, height=400)

    # Exibir 8 datas no eixo x
    num_dates = 5
    tick_values = ts.index[::max(1, len(ts.index) // num_dates)]
    tick_values = [ts.index[0]] + tick_values.tolist() + [
        ts.index[-1]]  # Adiciona o primeiro e o último valor do DataFrame
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=tick_values, tickangle=0))

    # Exibição do gráfico no Streamlit com largura de 100% e altura igual a 50% da largura
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

def main_moderado(select_MM, select_tickers):
    # Coleta das cotações
    ts = pd.DataFrame(df_cotacoes.loc[:, select_tickers]).copy()

    # Cálculo da média móvel
    ts['MM'] = ts[select_tickers].rolling(select_MM).mean()
    ts.dropna(inplace=True)

    # Calcular Retorno Logaritmico da Média Móvel
    ts['mm%'] = ts.apply(lambda x: math.log(x[0] / x[1]), axis=1)  # /mm

    # Calcuular Variação Diária
    ts['pct'] = np.log(ts.iloc[:, [0]].pct_change() + 1)
    ts['pct'].fillna(method='bfill', inplace=True)

    # REGRESSÃO LINEAR
    X_independent = ts['pct'].values.reshape(-1, 1)
    Y_dependent = ts['mm%'].values.reshape(-1, 1)

    reg = LinearRegression().fit(X_independent, Y_dependent)

    # Gerando Reta da regressao------------------------------------------------------->>>>>>>>>>>
    Y_predict = reg.predict(X_independent);
    # Calculando residuos
    ts['Resíduos'] = (Y_dependent - Y_predict)

    # Gráfico dos Resíduos
    mean = ts['Resíduos'].mean()
    std = ts['Resíduos'].std()

    ts['1std+'] = std
    ts['1std-'] = std * -1
    ts['2std+'] = std * 2
    ts['2std-'] = std * -2
    ts['3std+'] = std * 3
    ts['3std-'] = std * -3
    ts['zero'] = 0

    # Plot do gráfico Precos
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts[select_tickers], name=select_tickers, mode='lines',
                             line=dict(color=lineColor, width=2)))
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['MM'], name=f'Média Móvel {select_MM} períodos', mode='lines',
                   line=dict(color=MMlineColor, width=2)))
    # Incluir Pontos

    fig.add_trace(go.Scatter(
        x=ts[np.logical_and(ts['Resíduos'] >= ts['2std+'], ts['Resíduos'] < ts['3std+'])].index,
        y=ts.loc[np.logical_and(ts['Resíduos'] >= ts['2std+'], ts['Resíduos'] < ts['3std+']), select_tickers],
        mode='markers', marker=dict(color=dotColor2, size=10), showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=ts[np.logical_and(ts['Resíduos'] >= ts['3std+'], ts['Resíduos'] >= ts['3std+'])].index,
        y=ts.loc[np.logical_and(ts['Resíduos'] >= ts['3std+'], ts['Resíduos'] >= ts['3std+']), select_tickers],
        mode='markers', marker=dict(color=dotColor3, size=15), showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=ts[np.logical_and(ts['Resíduos'] <= ts['1std-'], ts['Resíduos'] > ts['2std-'])].index,
        y=ts.loc[np.logical_and(ts['Resíduos'] <= ts['1std-'], ts['Resíduos'] > ts['2std-']), select_tickers],
        mode='markers', marker=dict(color=dotColor1, size=5), showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=ts[np.logical_and(ts['Resíduos'] <= ts['2std-'], ts['Resíduos'] > ts['3std-'])].index,
        y=ts.loc[np.logical_and(ts['Resíduos'] <= ts['2std-'], ts['Resíduos'] > ts['3std-']), select_tickers],
        mode='markers', marker=dict(color=dotColor2, size=10), showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=ts[np.logical_and(ts['Resíduos'] <= ts['3std-'], ts['Resíduos'] <= ts['3std-'])].index,
        y=ts.loc[np.logical_and(ts['Resíduos'] <= ts['3std-'], ts['Resíduos'] <= ts['3std-']), select_tickers],
        mode='markers', marker=dict(color=dotColor3, size=15), showlegend=False
    ))

    fig.update_layout(title=f'Cotações de {select_tickers} Média Móvel de {select_MM} períodos')

    # Define a legenda na parte interna do gráfico
    fig.update_layout(legend=dict(x=0, y=1.0, orientation='h', traceorder='normal'), autosize=True,
                      height=500)

    # Exibir 8 datas no eixo x
    num_dates = 5
    tick_values = ts.index[::max(1, len(ts.index) // num_dates)]
    tick_values = [ts.index[0]] + tick_values.tolist() + [
        ts.index[-1]]  # Adiciona o primeiro e o último valor do DataFrame
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=tick_values, tickangle=0))

    # Exibição do gráfico no Streamlit com largura de 100% e altura igual a 50% da largura
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    # Sinalizando Entradas ============================================================================
    # VENDAS
    # Criando uma nova coluna que indica se o valor é maior que o limite superior de 3 desvios padrão
    ts['acima_3std'] = ts['Resíduos'] >= ts['3std+']
    ts['acima_2std'] = (ts['Resíduos'] >= ts['2std+']) & (ts['Resíduos'] < ts['3std+'])
    ts['acima_1std'] = (ts['Resíduos'] >= ts['1std+']) & (ts['Resíduos'] < ts['2std+'])
    # COMPRAS
    ts['abaixo_3std'] = ts['Resíduos'] <= ts['3std-']
    ts['abaixo_2std'] = (ts['Resíduos'] <= ts['2std-']) & (ts['Resíduos'] > ts['3std-'])
    ts['abaixo_1std'] = (ts['Resíduos'] <= ts['1std-']) & (ts['Resíduos'] > ts['2std-'])

    # ==================================================
    # Plot do gráfico de Resíduos
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['Resíduos'], name='Resíduos', mode='lines',
                   line=dict(color=lineColor, width=2)))
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['zero'], mode='lines', line=dict(color=MMlineColor, width=1, dash='solid')))
    fig.add_trace(go.Scatter(x=ts.index, y=ts['1std+'], mode='lines',
                             line=dict(color=horizontalLineColor, width=1.3, dash='dot')))
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['2std+'], mode='lines',
                   line=dict(color=horizontalLineColor, width=0.5, dash='dot')))
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['3std+'], mode='lines',
                   line=dict(color=horizontalLineColor, width=0.2, dash='dot')))

    fig.add_trace(go.Scatter(x=ts.index, y=ts['1std-'], mode='lines',
                             line=dict(color=horizontalLineColor, width=1.3, dash='dot')))
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['2std-'], mode='lines',
                   line=dict(color=horizontalLineColor, width=0.5, dash='dot')))
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['3std-'], mode='lines',
                   line=dict(color=horizontalLineColor, width=0.2, dash='dot')))

    fig.add_trace(go.Scatter(x=ts[ts['acima_2std']].index, y=ts.loc[ts['acima_2std'], 'Resíduos'], mode='markers',
                             marker=dict(color=dotColor2, size=10)))

    fig.add_trace(go.Scatter(x=ts[ts['acima_3std']].index, y=ts.loc[ts['acima_3std'], 'Resíduos'], mode='markers',
                             marker=dict(color=dotColor3, size=15)))

    fig.add_trace(go.Scatter(x=ts[ts['abaixo_1std']].index, y=ts.loc[ts['abaixo_1std'], 'Resíduos'], mode='markers',
                             marker=dict(color=dotColor1, size=5)))

    fig.add_trace(go.Scatter(x=ts[ts['abaixo_2std']].index, y=ts.loc[ts['abaixo_2std'], 'Resíduos'], mode='markers',
                             marker=dict(color=dotColor2, size=10)))

    fig.add_trace(go.Scatter(x=ts[ts['abaixo_3std']].index, y=ts.loc[ts['abaixo_3std'], 'Resíduos'], mode='markers',
                             marker=dict(color=dotColor3, size=15)))

    fig.update_layout(title=f'Gráfico Normalizado {select_tickers} Média Móvel de {select_MM} períodos')
    fig.update_layout(showlegend=False)  # Remove as legendas
    # Remover o eixo Y
    fig.update_layout(yaxis=dict(showticklabels=False, showgrid=False))
    fig.update_layout(yaxis=dict(showline=False, zeroline=False))
    # Define a legenda na parte interna do gráfico
    fig.update_layout(legend=dict(x=0, y=1.1, orientation='h', traceorder='normal'), autosize=True, height=400)

    # Exibir 8 datas no eixo x
    num_dates = 5
    tick_values = ts.index[::max(1, len(ts.index) // num_dates)]
    tick_values = [ts.index[0]] + tick_values.tolist() + [
        ts.index[-1]]  # Adiciona o primeiro e o último valor do DataFrame
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=tick_values, tickangle=0))

    # Exibição do gráfico no Streamlit com largura de 100% e altura igual a 50% da largura
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

def main_arrojado(select_MM, select_tickers):
    # Coleta das cotações
    ts = pd.DataFrame(df_cotacoes.loc[:, select_tickers]).copy()

    # Cálculo da média móvel
    ts['MM'] = ts[select_tickers].rolling(select_MM).mean()
    ts.fillna(method='bfill', inplace=True)

    # Calcular Retorno Logaritmico da Média Móvel
    ts['mm%'] = ts.apply(lambda x: math.log(x[0] / x[1]), axis=1)  # /mm

    # Calcuular Variação Diária
    ts['pct'] = np.log(ts.iloc[:, [0]].pct_change() + 1)
    ts['pct'].fillna(method='bfill', inplace=True)

    # REGRESSÃO LINEAR
    X_independent = ts['pct'].values.reshape(-1, 1)
    Y_dependent = ts['mm%'].values.reshape(-1, 1)

    reg = LinearRegression().fit(X_independent, Y_dependent)

    # Gerando Reta da regressao------------------------------------------------------->>>>>>>>>>>
    Y_predict = reg.predict(X_independent);
    # Calculando residuos
    ts['Resíduos'] = (Y_dependent - Y_predict)

    # Gráfico dos Resíduos
    mean = ts['Resíduos'].mean()
    std = ts['Resíduos'].std()

    ts['1std+'] = std
    ts['1std-'] = std * -1
    ts['2std+'] = std * 2
    ts['2std-'] = std * -2
    ts['3std+'] = std * 3
    ts['3std-'] = std * -3
    ts['zero'] = mean

    # Plot do gráfico Precos
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts[select_tickers], name=select_tickers, mode='lines',
                             line=dict(color=lineColor, width=2)))
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['MM'], name=f'Média Móvel {select_MM} períodos', mode='lines',
                   line=dict(color=MMlineColor, width=2)))
    # Incluir Pontos
    fig.add_trace(go.Scatter(
        x=ts[np.logical_and(ts['Resíduos'] >= ts['1std+'], ts['Resíduos'] < ts['2std+'])].index,
        y=ts.loc[np.logical_and(ts['Resíduos'] >= ts['1std+'], ts['Resíduos'] < ts['2std+']), select_tickers],
        mode='markers', marker=dict(color=dotColor1, size=5), showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=ts[np.logical_and(ts['Resíduos'] <= ts['1std-'], ts['Resíduos'] > ts['2std-'])].index,
        y=ts.loc[np.logical_and(ts['Resíduos'] <= ts['1std-'], ts['Resíduos'] > ts['2std-']), select_tickers],
        mode='markers', marker=dict(color=dotColor1, size=5), showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=ts[np.logical_and(ts['Resíduos'] >= ts['2std+'], ts['Resíduos'] < ts['3std+'])].index,
        y=ts.loc[np.logical_and(ts['Resíduos'] >= ts['2std+'], ts['Resíduos'] < ts['3std+']), select_tickers],
        mode='markers', marker=dict(color=dotColor2, size=10), showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=ts[np.logical_and(ts['Resíduos'] <= ts['2std-'], ts['Resíduos'] > ts['3std-'])].index,
        y=ts.loc[np.logical_and(ts['Resíduos'] <= ts['2std-'], ts['Resíduos'] > ts['3std-']), select_tickers],
        mode='markers', marker=dict(color=dotColor2, size=10), showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=ts[np.logical_and(ts['Resíduos'] >= ts['3std+'], ts['Resíduos'] >= ts['3std+'])].index,
        y=ts.loc[np.logical_and(ts['Resíduos'] >= ts['3std+'], ts['Resíduos'] >= ts['3std+']), select_tickers],
        mode='markers', marker=dict(color=dotColor3, size=15), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=ts[np.logical_and(ts['Resíduos'] <= ts['3std-'], ts['Resíduos'] <= ts['3std-'])].index,
        y=ts.loc[np.logical_and(ts['Resíduos'] <= ts['3std-'], ts['Resíduos'] <= ts['3std-']), select_tickers],
        mode='markers', marker=dict(color=dotColor3, size=15), showlegend=False
    ))

    fig.update_layout(title=f'Cotações de {select_tickers} Média Móvel de {select_MM} períodos')

    # Define a legenda na parte interna do gráfico
    fig.update_layout(legend=dict(x=0, y=1.0, orientation='h', traceorder='normal'), autosize=True,
                      height=500)

    # Exibir 8 datas no eixo x
    num_dates = 5
    tick_values = ts.index[::max(1, len(ts.index) // num_dates)]
    tick_values = [ts.index[0]] + tick_values.tolist() + [
        ts.index[-1]]  # Adiciona o primeiro e o último valor do DataFrame
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=tick_values, tickangle=0))

    # Exibição do gráfico no Streamlit com largura de 100% e altura igual a 50% da largura
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    # Sinalizando Entradas ============================================================================
    # VENDAS
    # Criando uma nova coluna que indica se o valor é maior que o limite superior de 3 desvios padrão
    ts['acima_3std'] = ts['Resíduos'] >= ts['3std+']
    ts['acima_2std'] = (ts['Resíduos'] >= ts['2std+']) & (ts['Resíduos'] < ts['3std+'])
    ts['acima_1std'] = (ts['Resíduos'] >= ts['1std+']) & (ts['Resíduos'] < ts['2std+'])
    # COMPRAS
    ts['abaixo_3std'] = ts['Resíduos'] <= ts['3std-']
    ts['abaixo_2std'] = (ts['Resíduos'] <= ts['2std-']) & (ts['Resíduos'] > ts['3std-'])
    ts['abaixo_1std'] = (ts['Resíduos'] <= ts['1std-']) & (ts['Resíduos'] > ts['2std-'])

    # ==================================================
    # Plot do gráfico de Resíduos
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['Resíduos'], name='Resíduos', mode='lines',
                   line=dict(color=lineColor, width=2)))
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['zero'], mode='lines', line=dict(color=MMlineColor, width=1, dash='solid')))
    fig.add_trace(go.Scatter(x=ts.index, y=ts['1std+'], mode='lines',
                             line=dict(color=horizontalLineColor, width=1.3, dash='dot')))
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['2std+'], mode='lines',
                   line=dict(color=horizontalLineColor, width=0.5, dash='dot')))
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['3std+'], mode='lines',
                   line=dict(color=horizontalLineColor, width=0.2, dash='dot')))

    fig.add_trace(go.Scatter(x=ts.index, y=ts['1std-'], mode='lines',
                             line=dict(color=horizontalLineColor, width=1.3, dash='dot')))
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['2std-'], mode='lines',
                   line=dict(color=horizontalLineColor, width=0.5, dash='dot')))
    fig.add_trace(
        go.Scatter(x=ts.index, y=ts['3std-'], mode='lines',
                   line=dict(color=horizontalLineColor, width=0.2, dash='dot')))


    fig.add_trace(go.Scatter(x=ts[ts['acima_1std']].index, y=ts.loc[ts['acima_1std'], 'Resíduos'], mode='markers',
                             marker=dict(color=dotColor1, size=5)))

    fig.add_trace(go.Scatter(x=ts[ts['acima_2std']].index, y=ts.loc[ts['acima_2std'], 'Resíduos'], mode='markers',
                             marker=dict(color=dotColor2, size=10)))

    fig.add_trace(go.Scatter(x=ts[ts['acima_3std']].index, y=ts.loc[ts['acima_3std'], 'Resíduos'], mode='markers',
                             marker=dict(color=dotColor3, size=15)))

    fig.add_trace(go.Scatter(x=ts[ts['abaixo_1std']].index, y=ts.loc[ts['abaixo_1std'], 'Resíduos'], mode='markers',
                             marker=dict(color=dotColor1, size=5)))

    fig.add_trace(go.Scatter(x=ts[ts['abaixo_2std']].index, y=ts.loc[ts['abaixo_2std'], 'Resíduos'], mode='markers',
                             marker=dict(color=dotColor2, size=10)))

    fig.add_trace(go.Scatter(x=ts[ts['abaixo_3std']].index, y=ts.loc[ts['abaixo_3std'], 'Resíduos'], mode='markers',
                             marker=dict(color=dotColor3, size=15)))

    fig.update_layout(title=f'Gráfico Normalizado {select_tickers} Média Móvel de {select_MM} períodos')
    fig.update_layout(showlegend=False)  # Remove as legendas
    # Remover o eixo Y
    fig.update_layout(yaxis=dict(showticklabels=False, showgrid=False))
    fig.update_layout(yaxis=dict(showline=False, zeroline=False))
    # Define a legenda na parte interna do gráfico
    fig.update_layout(legend=dict(x=0, y=1.1, orientation='h', traceorder='normal'), autosize=True, height=400)

    # Exibir 8 datas no eixo x
    num_dates = 5
    tick_values = ts.index[::max(1, len(ts.index) // num_dates)]
    tick_values = [ts.index[0]] + tick_values.tolist() + [
        ts.index[-1]]  # Adiciona o primeiro e o último valor do DataFrame
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=tick_values, tickangle=0))

    # Exibição do gráfico no Streamlit com largura de 100% e altura igual a 50% da largura
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

def chart_vol(select_tickers):
    # Coleta das cotações
    df_chart = pd.DataFrame(df_cotacoes[select_tickers]).copy()

    # Cálculo da volatilidade em janela móvel
    df_chart['Vol'] = df_chart[select_tickers].pct_change().rolling(window=21).std() * np.sqrt(252)
    df_chart.dropna(inplace=True)

    df_chart['Baixa'] = np.percentile(df_chart['Vol'], 20)
    df_chart['Alta'] = np.percentile(df_chart['Vol'], 80)
    df_chart['Mean'] = df_chart['Vol'].mean()

    Mean = df_chart['Vol'].mean()
    std = df_chart['Vol'].std()

    df_chart['+1Desv'] = Mean + (std * 1)
    df_chart['-1Desv'] = Mean + (std * -1)
    df_chart['+2Desv'] = Mean + (std * 2)
    df_chart['-2Desv'] = Mean + (std * -2)

    if df_chart['Vol'].iloc[-1] < df_chart['-1Desv'].iloc[-1] and df_chart['Vol'].iloc[-1] > df_chart['-2Desv'].iloc[-1]:
        classVol = 'BAIXA'
    elif df_chart['Vol'].iloc[-1] < df_chart['-2Desv'].iloc[-1]:
        classVol = 'MUITO BAIXA'
    elif df_chart['Vol'].iloc[-1] > df_chart['+1Desv'].iloc[-1] and df_chart['Vol'].iloc[-1] < df_chart['+2Desv'].iloc[-1]:
        classVol = 'ALTA'
    elif df_chart['Vol'].iloc[-1] > df_chart['+2Desv'].iloc[-1]:
        classVol = 'MUITO ALTA'
    else:
        classVol = 'NEUTRA'

    # Grafico de Volatilidade
    # Plot do gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Vol'], name=select_tickers, mode='lines',
                             line=dict(color=LineVol, width=1)))
    # fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Mean'], name='Vol Média', mode='lines'))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['-1Desv'], name='-1', mode='lines',
                             line=dict(color=horizontalLineColor, width=0.5, dash='dot')))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['+1Desv'], name='+1', mode='lines',
                             line=dict(color=horizontalLineColor, width=0.5, dash='dot')))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['-2Desv'], name='-2', mode='lines',
                             line=dict(color=horizontalLineColor, width=0.5, dash='dot')))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['+2Desv'], name='+2', mode='lines',
                             line=dict(color=horizontalLineColor, width=0.5, dash='dot')))
    fig.update_layout(title=f'Volatildiade Atual: {classVol}')

    # Incluir um círculo branco no último valor de 'df_chart['Vol']'
    ultimo_valor_vol = df_chart['Vol'].iloc[-1]
    fig.add_trace(
        go.Scatter(x=[df_chart.index[-1]], y=[ultimo_valor_vol], mode='markers', marker=dict(color='white', size=10)))

    # Define a legenda na parte interna do gráfico
    fig.update_layout(legend=dict(x=0, y=1.0, orientation='h', traceorder='normal'), showlegend=False, autosize=True,
                      height=400)

    # Exibir 8 datas no eixo x
    num_dates = 3
    tick_values = df_chart.index[::max(1, len(df_chart.index) // num_dates)]
    tick_values = [df_chart.index[0]] + tick_values.tolist() + [
        df_chart.index[-1]]  # Adiciona o primeiro e o último valor do DataFrame
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=tick_values, tickangle=0))
    # Remover o eixo Y
    fig.update_layout(yaxis=dict(showticklabels=False, showgrid=False))
    fig.update_layout(yaxis=dict(showline=False, zeroline=False))

    # Exibição do gráfico no Streamlit com largura de 100% e altura igual a 50% da largura
    st.plotly_chart(fig, use_container_width=True)

# Coletar Dados ==========================================================
df_descricao = carregar_descricao()
df_rankFM = carregar_rankFM()

# Verificador Cache
url = "https://raw.githubusercontent.com/diogoraucci/Prototipo_Quant_MM/main/dataControl.csv"
# Faz o download do conteúdo do arquivo
response = requests.get(url)
conteudo_csv = response.content
# Lê o conteúdo baixado como um DataFrame do pandas
dataControl = pd.read_csv(BytesIO(conteudo_csv))#, index_col=0)
teteData = pd.DataFrame([[dataControl.iat[0, 0]]], columns=['0'])
#teteData = pd.DataFrame({'Data': [teteData]})
#st.dataframe(teteData)

teteData2 = pd.DataFrame([[dataControl.iat[0, 0]]], columns=['0'])
#teteData = pd.DataFrame({'Data': [teteData]})
#st.dataframe(teteData2)

#df_cotacoes.index[-1]
#df_cotacoes.index[-1]
teste2 = pd.DataFrame(['Diferentes'])  # ================================
teste1 = pd.DataFrame(['Iguais'])  # ================================
if teteData2.equals(teteData):
    # Os DataFrames são iguais
    st.dataframe(teste1)
    st.dataframe(pd.DataFram(df_cotacoes))
else:
    # Os DataFrames são diferentes
    st.dataframe(teste2)  # ================================
    st.cache_data.clear()










# CORPO DA PÁGINA # =========================================================================
st.title('Análise Fundamentalista e Quantitativa de Ações')

# Menu Lateral
add_selectbox = st.sidebar.selectbox(
    "!!!EM DESENVOLVIMENTO!!!",
    ("Ações B3", "Opções  !!EM BREVE!! ", "Long&Short Forex  !!EM BREVE!! ", "Crypto  !!EM BREVE!! ")
)

with st.sidebar:
    add_radio = st.radio(
        "Vencimento de Opções !! EM DESENVOLVIMENTO !!",
        ("Data de Montagem 01/01/2222", "Data do Exercício 01/01/2222")
    )

# Criando o container com três partes
container = st.container()

# Menu ==================================
with container:
    fail = False
    col0,col1, col2, col3, col4 = st.columns(5)

    df_resultadosAnual, df_resultadosTrim = carregar_resultado()
    df_cotacoes = carregar_cotacoes()


    # Verificardor Limpeza de Cache
    dataControl_cache = pd.DataFrame([df_cotacoes.index[-1]])
    # Verificar se a última data do df_cotacoes é maior que a primeira data do dataControl_cache
    if df_cotacoes.index[-1] > dataControl_cache.iat[0, 0]:
        # Se for maior, limpar a memória cache do DataFrame df_cotacoes
        st.experimental_memo.clear(df_cotacoes)
        st.experimental_memo.clear(df_dataset)
        st.cache.clear()
    else:
        # Caso contrário, não fazer nada
        pass

    # Seletor Sinal de Entrada
    select_sinalEntrada = col0.selectbox("Modo de Visualização", ['Analizar Ativos', 'Rastreador de Entrada'])

    # Seletor Perfil de Risco
    select_PerfilRisco = col1.selectbox("Perfil de Risco", ['Conservador', 'Moderado', 'Arrojado'])
    if select_PerfilRisco == 'Conservador':
        filtroScore = 9
        df_dataset = carregar_dataset()
        df_dataset = df_dataset[(df_dataset['ADF'] >= 0.95)]
        df_dataset = df_dataset[(df_dataset['Score_AnoTrim'] >= filtroScore) & (df_dataset['MM'] >= 50) & (df_dataset['Angle_1y'] > 0)]


        if select_PerfilRisco == 'Conservador' and select_sinalEntrada == 'Rastreador de Entrada':
            df_dataset = df_dataset[(df_dataset['Sinal'] == 'Rastreador de Entrada') & (df_dataset['STD'] <= -1)]

            # Seletor Ações
            tickers = list(df_dataset.index.drop_duplicates())
            select_tickers = col2.selectbox('Selecione um Ticker', sorted(tickers))
            #df_cotacoes = carregar_cotacoes()

            # Seletor Média Móvel
            MM = tuple(df_dataset['MM'][df_dataset.index == select_tickers])
            mm_limit = int(len(MM))
            if mm_limit < 1:
                MM = MM[-1:]
            else:
                MM = MM[len(MM) - mm_limit:]

            select_MM = col3.selectbox('Média Móvel', MM)

        elif select_PerfilRisco == 'Conservador' and select_sinalEntrada == 'Analizar Ativos':

            # Seletor Ações
            tickers = list(df_dataset.index.drop_duplicates())
            select_tickers = col2.selectbox('Selecione um Ticker', sorted(tickers))
            #df_cotacoes = carregar_cotacoes()

            # Seletor Média Móvel
            MM = tuple(df_dataset['MM'][df_dataset.index == select_tickers])
            mm_limit = int(len(MM))
            if mm_limit < 1:
                MM = MM[-1:]
            else:
                MM = MM[len(MM) - mm_limit:]
            select_MM = col3.selectbox('Média Móvel', MM)

    elif select_PerfilRisco == 'Moderado':
        filtroScore = 8
        #df_dataset = carregar_dataset()
        #df_dataset = df_dataset[(df_dataset['ADF'] >= 0.95)]
        #df_dataset = df_dataset[(df_dataset['Score_AnoTrim'] >= filtroScore) & (df_dataset['MM'] >= 50) & (df_dataset['Angle_1y'] > 0)]

        if select_PerfilRisco == 'Moderado' and select_sinalEntrada == 'Rastreador de Entrada':
            df_dataset = carregar_dataset()
            df_dataset = df_dataset[(df_dataset['ADF'] >= 0.95) & (df_dataset['Score_AnoTrim'] >= filtroScore) & (df_dataset['MM'] >= 50) & (df_dataset['Angle_1y'] > 0) & (df_dataset['Sinal'] == 'Yes') & ((df_dataset['STD'] <= -1) | (df_dataset['STD'] >= 2))]

            #df_dataset = df_dataset[(df_dataset['Sinal'] == 'Yes') | (df_dataset['STD'] != 1)]

            # Seletor Ações
            tickers = list(df_dataset.index.drop_duplicates())
            select_tickers = col2.selectbox('Selecione um Ticker', sorted(tickers))
            #df_cotacoes = carregar_cotacoes()

            # Seletor Média Móvel
            MM = tuple(df_dataset['MM'][df_dataset.index == select_tickers])
            mm_limit = int(len(MM))
            if mm_limit < 1:
                MM = MM[-1:]
            else:
                MM = MM[len(MM) - mm_limit:]

            select_MM = col3.selectbox('Média Móvel', MM)

        elif select_PerfilRisco == 'Moderado' and select_sinalEntrada == 'Analizar Ativos':

            # Seletor Ações
            df_dataset = carregar_dataset()
            tickers = list(df_dataset.index.drop_duplicates())
            select_tickers = col2.selectbox('Selecione um Ticker', sorted(tickers))
            #df_cotacoes = carregar_cotacoes()

            # Seletor Média Móvel
            MM = tuple(df_dataset['MM'][df_dataset.index == select_tickers])
            mm_limit = int(len(MM))
            if mm_limit < 1:
                MM = MM[-1:]
            else:
                MM = MM[len(MM) - mm_limit:]
            select_MM = col3.selectbox('Média Móvel', MM)

    elif select_PerfilRisco == 'Arrojado':
        # Seletor Ranck Ações
        select_RankAcoes = col4.selectbox('Score das Ações', ('Todas as Ações', '5 Pts', '6 Pts', '7 Pts', '8 Pts', '9 Pts', '10 Pts'))
        if select_RankAcoes == 'Todas as Ações':
            filtroScore = 0
        elif select_RankAcoes == '5 Pts':
            filtroScore = 5
        elif select_RankAcoes == '6 Pts':
            filtroScore = 6
        elif select_RankAcoes == '7 Pts':
            filtroScore = 7
        elif select_RankAcoes == '8 Pts':
            filtroScore = 8
        elif select_RankAcoes == '9 Pts':
            filtroScore = 9
        elif select_RankAcoes == '10 Pts':
            filtroScore = 10

        df_dataset = carregar_dataset()
        df_dataset = df_dataset[(df_dataset['ADF'] >= 0.90)]
        df_dataset = df_dataset[(df_dataset['Score_AnoTrim'] >= filtroScore)]

        if select_PerfilRisco == 'Arrojado' and select_sinalEntrada == 'Rastreador de Entrada':
            df_dataset = df_dataset[(df_dataset['Sinal'] == 'Yes')]

            # Seletor Ações
            tickers = list(df_dataset.index.drop_duplicates())
            select_tickers = col2.selectbox('Selecione um Ticker', sorted(tickers))
            #df_cotacoes = carregar_cotacoes()

            # Seletor Média Móvel
            MM = tuple(df_dataset['MM'][df_dataset.index == select_tickers])
            select_MM = col3.selectbox('Média Móvel', MM)

        elif select_PerfilRisco == 'Arrojado' and select_sinalEntrada == 'Analizar Ativos':

            # Seletor Ações
            tickers = list(df_dataset.index.drop_duplicates())
            select_tickers = col2.selectbox('Selecione um Ticker', sorted(tickers))

            # Seletor Média Móvel
            MM = tuple(df_dataset['MM'][df_dataset.index == select_tickers])
            select_MM = col3.selectbox('Média Móvel', MM)

 # ----------------------------------------------------------------------------------
# Segunda parte com gráficos de linhas e de barras horizontais
# Plota o gráfico Trimestral e Anual de barras ==============================================
res_trim = df_resultadosTrim[df_resultadosTrim['TICKER'] == select_tickers ]
res_anual = df_resultadosAnual[df_resultadosAnual['TICKER'] == select_tickers ]

with container:
    col1, col2 = st.columns([6, 2])

    # Gráfico de linhas na coluna 1
    with col1:
        if __name__ == '__main__':
            #main(select_MM, select_tickers)
            try:
                if select_PerfilRisco == 'Conservador':
                   main_conservador(select_MM, select_tickers)

                elif select_PerfilRisco == 'Moderado':
                    main_moderado(select_MM, select_tickers)

                elif select_PerfilRisco == 'Arrojado':
                    main_arrojado(select_MM, select_tickers)

            except:
                st.markdown(f'No momento, não há nenhuma oportunidade no Perfil De Risco {select_PerfilRisco}')

    with col2:
        # Criar o gráfico de barras horizontais Resultado Trimestral com Altair
        res_trim['Data'] = res_trim.index.strftime('%Y-%m')
        res_trim.rename(columns={'Net Income': 'NetIncome'}, inplace=True)

        chart_trim = (
            alt.Chart(res_trim)
                .mark_bar()
                .encode(
                y=alt.Y('Data:O', title='', sort=alt.EncodingSortField('Data', order='ascending')),
                x=alt.X('NetIncome:Q', title=''),
                color=alt.condition(alt.datum.NetIncome > 0, alt.value(colorUp), alt.value(colorDown))
            )
        )

        # Adicionar valores das barras
        text_trim = (
            chart_trim.mark_text(
                align='left',
                baseline='middle',
                dx=5  # Espaço para o rótulo
            )
        )

        # Configurações adicionais
        chart_trim = (
                chart_trim + text_trim
        ).properties(
            title='Resultado Trimestral',
            width=chartwidth,  # Aumentar a largura do gráfico
            height=chartheight  # Aumentar a altura do gráfico
        ).configure_axis(
            labelFontSize=14,  # Aumentar o tamanho da fonte do eixo
            titleFontSize=16  # Aumentar o tamanho da fonte do título
        ).configure_axisX(
            labels=False  # Remover rótulos do eixo x
        )

        # Exibir o gráfico trimestral no Streamlit
        try:
            st.altair_chart(chart_trim, use_container_width=True)
        except:
            # st.markdown(f'Não há nenhuma oportunidade no Perfil De Risco {select_PerfilRisco}')
            print()

        # Criar o gráfico de barras horizontais Resultado Anual ===============================
        res_anual['Data'] = res_anual.index.strftime('    --   %Y')
        res_anual.rename(columns={'Net Income': 'NetIncome'}, inplace=True)

        chart_anual = (
            alt.Chart(res_anual)
                .mark_bar()
                .encode(
                y=alt.Y('Data:O', title='', sort=alt.EncodingSortField('Data', order='ascending')),
                x=alt.X('NetIncome:Q', title=''),
                color=alt.condition(alt.datum.NetIncome > 0, alt.value(colorUp), alt.value(colorDown))
            )
        )

        # Adicionar valores das barras
        text_anual = (
            chart_anual.mark_text(
                align='left',
                baseline='middle',
                dx=10,  # Espaço para o rótulo
                dy=0  # Ajuste vertical dos rótulos
            )
        )

        # Configurações adicionais
        chart_anual = (
                chart_anual + text_anual
        ).properties(
            title='Resultado Anual',
            width=chartwidth,  # Aumentar a largura do gráfico
            height=chartheight  # Aumentar a altura do gráfico
        ).configure_axis(
            labelFontSize=14,  # Aumentar o tamanho da fonte do eixo
            titleFontSize=16  # Aumentar o tamanho da fonte do título
        ).configure_axisX(
            labels=False  # Remover rótulos do eixo x
        )

        # Exibir o gráfico anual no Streamlit
        try:
            st.altair_chart(chart_anual, use_container_width=True)
            print('Dados Indisponíveis')

            # Grafico de Volatildiade
            chart_vol(select_tickers)

            # Descrição ========================================================================
            descricao = df_descricao.at[select_tickers, 'Descricao']
            print('Dados Indisponíveis')
        except:
            # st.markdown(f'Não há nenhuma oportunidade no Perfil De Risco {select_PerfilRisco}')
            print()

        try:
            alturaTextBox = 200
            st.text_area("Descrição da Empresa", value=descricao, height=alturaTextBox, max_chars=None)
        except:
            # st.markdown(f'Não há nenhuma oportunidade no Perfil De Risco {select_PerfilRisco}')
            None