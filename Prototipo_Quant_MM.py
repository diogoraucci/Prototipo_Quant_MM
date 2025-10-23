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
    url = "https://raw.githubusercontent.com/diogoraucci/Prototipo_Quant_MM/main/DF_FormulaMagica.xlsx"
    # Faz o download do conteúdo do arquivo
    response = requests.get(url)
    conteudo_excel = response.content
    # Lê o conteúdo baixado como um arquivo Excel usando BytesIO
    df_descricao = pd.read_excel(BytesIO(conteudo_excel), sheet_name='descricaoBR', index_col=1, engine='openpyxl')
    return df_descricao

@st.cache_data()
def carregar_rankFM():
    # URL do arquivo Excel no GitHub
    url = "https://raw.githubusercontent.com/diogoraucci/Prototipo_Quant_MM/main/DF_FormulaMagica.xlsx"
    # Faz o download do conteúdo do arquivo
    response = requests.get(url)
    conteudo_excel = response.content
    # Lê o conteúdo baixado como um arquivo Excel usando BytesIO
    df_rankFM = pd.read_excel(BytesIO(conteudo_excel), sheet_name='FM_acoesbancos', index_col=0, engine='openpyxl')
    return df_rankFM

@st.cache_data()
def carregar_resultado():
    # URL do arquivo Excel no GitHub
    url = "https://raw.githubusercontent.com/diogoraucci/Prototipo_Quant_MM/main/DF_FormulaMagica.xlsx"
    # Faz o download do conteúdo do arquivo
    response = requests.get(url)
    conteudo_excel = response.content
    # Lê o conteúdo baixado como um arquivo Excel usando BytesIO
    df_resultadosAnual = pd.read_excel(BytesIO(conteudo_excel), sheet_name='df_resultadoAnualBR', index_col=0, engine='openpyxl')
    df_resultadosTrim = pd.read_excel(BytesIO(conteudo_excel), sheet_name='df_resultadoTrimBR', index_col=0, engine='openpyxl')
    return df_resultadosAnual, df_resultadosTrim

@st.cache_data ()
def carregar_dataset():
    # URL do arquivo Excel no GitHub
    url = "https://raw.githubusercontent.com/diogoraucci/Prototipo_Quant_MM/main/DF_FormulaMagica.xlsx"
    # Faz o download do conteúdo do arquivo
    response = requests.get(url)
    conteudo_excel = response.content
    # Lê o conteúdo baixado como um arquivo Excel usando BytesIO
    df_dataset = pd.read_excel(BytesIO(conteudo_excel), sheet_name='CompletoBR', index_col=0, engine='openpyxl')
    # Verifique o DataFrame
    return df_dataset

@st.cache_data ()
def carregar_cotacoes():
     # URL do arquivo Excel no GitHub
    url = "https://raw.githubusercontent.com/diogoraucci/Prototipo_Quant_MM/main/DF_FormulaMagica.xlsx"
    # Faz o download do conteúdo do arquivo
    response = requests.get(url)
    conteudo_excel = response.content
    # Lê o conteúdo baixado como um arquivo Excel usando BytesIO
    df_cotacoes = pd.read_excel(BytesIO(conteudo_excel), sheet_name='cotacoesBR', index_col=0, engine='openpyxl')
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

    fig.update_layout(title=f'Gráfico