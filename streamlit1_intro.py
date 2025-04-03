import streamlit as st
import Funciones as MCF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #Visualizacion de datos
import yfinance as yf #Api de Yahoo Finanzas
from scipy.stats import kurtosis, skew, shapiro ,norm,t #Funciones estadísticas
from datetime import date
import proyecto1_mercados as CR

ticker = CR.ticker
st.title(f'Análisis del activo {ticker} desde 2010')

st.header('Visualización de datos')


with st.spinner('Descargando data...'):
    df_precios = MCF.obtener_datos(ticker)
    df_rendimientos = MCF.calcular_rendimientos_Log(df_precios)

#Datos inciso b)

st.subheader(f"Gráfico de precio: {ticker}")

fig,ax = plt.subplots(figsize = (10,5))
ax.plot(df_precios.index,df_precios)
ax.axhline(y = 0,linestyle = '-',alpha = 0.7)
ax.set_xlabel("Fecha")
ax.set_ylabel("Precio Diario")
st.pyplot(fig)

st.subheader("Medidas de los rendimientos logarítmicos diarios")

promedio_rendi_diario = df_rendimientos.mean()
kurtosis = kurtosis(df_rendimientos)
skew = skew(df_rendimientos)
col1,col2,col3 = st.columns(3)

col1.metric("Rendimiento Medio Diario", f"{promedio_rendi_diario:.4%}")
col2.metric("Kurtosis",f"{kurtosis:.4}")
col3.metric("Sesgo",f"{skew:.3}")


st.subheader(f'Gráfico de Rendimientos : {ticker}')

fig,ax = plt.subplots(figsize = (10,5))
ax.plot(df_rendimientos.index,df_rendimientos)
ax.axhline(y = 0,linestyle = '-',alpha = 0.7)
ax.set_xlabel("Fecha")
ax.set_ylabel("Rendimiento Diario")
st.pyplot(fig)

 # Histograma 
st.subheader(f'Histograma de Rendimientos : {ticker}')
fig2,ax = plt.subplots(figsize =(10,5))
ax.hist(df_rendimientos,bins=50,alpha=0.5,color = 'green')
ax.set_title('Histograma')
ax.set_xlabel('Rendimiento Diario')
ax.set_ylabel('Frecuencia')
st.pyplot(fig2)

#VaR y ES inciso c)
st.header(f"Medidas de Riesgo de {ticker} ")
st.subheader("VaR a 95%, 97.5% y 99%")

tabla1= pd.DataFrame(
    {
        "% Confianza": ["95%", "99%", "99%"],
        "Paramétrico N": [CR.VaR_95, CR.VaR_99, CR.VaR_99],
        "Paramétrico T": [CR.VaR_95_t,CR.VaR_99_t, CR.VaR_99_t],
        "Histórico" : [CR.hVaR_95, CR.hVaR_99,CR.hVaR_99],
        "Monte Carlo" : [CR.MCVaR_95, CR.MCVaR_99,CR.MCVaR_99],
    }
)

st.dataframe(tabla1)

st.subheader("ES a 95%, 97.5% y 99%")

tabla2= pd.DataFrame(
    {
        "% Confianza": ["95%", "99%", "99%"],
        "Paramétrico N": [CR.ES_95, CR.ES_99, CR.ES_99],
        "Paramétrico T": [CR.ES_95_t,CR.ES_99_t, CR.ES_99_t],
        "Histórico" : [CR.hES_95, CR.hES_99,CR.hES_99],
        "Monte Carlo" : [CR.MCES_95, CR.MCES_99,CR.MCES_99],
    }
)

st.dataframe(tabla2)

#Histograma donde se muestra VaR y ES-----------------------------------------------------------------
st.subheader("Histogramas de las Medidas de Riesgo")

st.text("Se marcarán los rendimientos que superen al VaR histórico")

fig3,ax = plt.subplots(figsize =(10,5))
n, bins, patches = plt.hist(df_rendimientos, bins=50, color='blue', alpha=0.7, label='Returns')


# Identificamos datos menores al VaR histórico y los marcamos de rojo
for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
    if bin_left < CR.hVaR_95:
        patch.set_facecolor('red')

# Mark the different VaR and CVaR values on the histogram
plt.axvline(x=CR.VaR_95_t, color='skyblue', linestyle='--', label='VaR 95% (Paramétrico t)')
plt.axvline(x=CR.MCVaR_95, color='grey', linestyle='--', label='VaR 95% (Monte Carlo)')
plt.axvline(x=CR.hVaR_95, color='green', linestyle='--', label='VaR 95% (Histórico)')
plt.axvline(x=CR.ES_95_t, color='purple', linestyle='-.', label='ES (Paramétrico t) 95%')

# Add a legend and labels to make the chart more informative
plt.title('Histograma de Rendimientos con VaR y ES a 95%')
plt.xlabel('Rendimientos')
plt.ylabel('Frecuencia')
plt.legend()

st.pyplot(fig3)

#---------------------------------------------------------------------------------------

fig4,ax = plt.subplots(figsize =(10,5))
n, bins, patches = plt.hist(df_rendimientos, bins=50, color='blue', alpha=0.7, label='Returns')


# Identify bins to the left of hVaR_99 and color them differently
for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
    if bin_left < CR.hVaR_975:
        patch.set_facecolor('red')

# Mark the different VaR and CVaR values on the histogram
plt.axvline(x=CR.VaR_975_t, color='skyblue', linestyle='--', label='VaR 97.5% (Paramétrico t)')
plt.axvline(x=CR.MCVaR_975, color='grey', linestyle='--', label='VaR 97.5% (Monte Carlo)')
plt.axvline(x=CR.hVaR_975, color='green', linestyle='--', label='VaR 97.5% (Histórico)')
plt.axvline(x=CR.ES_975_t, color='purple', linestyle='-.', label='ES (Paramétrico t) 97.5%')

# Nombres de ejes
plt.title('Histograma de Rendimientos con VaR y ES a 97.5%')
plt.xlabel('Rendimientos')
plt.ylabel('Frecuencia')
plt.legend()

# Display the chart
st.pyplot(fig4)


fig5,ax = plt.subplots(figsize =(10,5))
n, bins, patches = plt.hist(df_rendimientos, bins=50, color='blue', alpha=0.7, label='Returns')


# Identify bins to the left of hVaR_99 and color them differently
for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
    if bin_left < CR.hVaR_99:
        patch.set_facecolor('red')

# Mark the different VaR and CVaR values on the histogram
plt.axvline(x=CR.VaR_99_t, color='skyblue', linestyle='--', label='VaR 99% (Paramétrico t)')
plt.axvline(x=CR.MCVaR_99, color='grey', linestyle='--', label='VaR 99% (Monte Carlo)')
plt.axvline(x=CR.hVaR_99, color='green', linestyle='--', label='VaR 99% (Histórico)')
plt.axvline(x=CR.ES_99_t, color='purple', linestyle='-.', label='ES (Paramétrico t) 99%')

# Nombres de ejes
plt.title('Histograma de Rendimientos con VaR y ES a 99%')
plt.xlabel('Rendimientos')
plt.ylabel('Frecuencia')
plt.legend()

# Display the chart
st.pyplot(fig5)

#d)) Rolling windows VaR------------------------------------------------------------------------------

# Graficamos rendimientos
fig6,ax = plt.subplots(figsize = (10,5))
ax.plot(df_rendimientos.index,df_rendimientos * 100, label='Daily Returns (%)', color='blue', alpha=0.5)
ax.axhline(y = 0,linestyle = '-',alpha = 0.7)
ax.set_xlabel("Fecha")
ax.set_ylabel("Rendimiento Diario")

# Plot the 95% Rolling VaR
plt.plot(df_rendimientos.index,CR.VaR_95_rolling_df['95% VaR Rolling'], label='95% Rolling VaR', color='red')

# Add a title and axis labels
plt.title('Daily Returns and 95% Rolling VaR')
plt.xlabel('Date')
plt.ylabel('Values (%)')

# Add a legend
plt.legend()

# Show the plot
#plt.tight_layout()
#plt.show()
st.pyplot(fig6)

st.subheader("Eficiencia de estimaciones")
st.dataframe(CR.df_final)