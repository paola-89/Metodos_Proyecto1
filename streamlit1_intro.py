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


st.subheader(f'Gráfico de Rendimientos Logarítmicos: {ticker}')

fig,ax = plt.subplots(figsize = (10,5))
ax.plot(df_rendimientos.index,df_rendimientos)
ax.axhline(y = 0,linestyle = '-',alpha = 0.7)
ax.set_xlabel("Fecha")
ax.set_ylabel("Rendimiento Diario")
st.pyplot(fig)

 # Histograma 
st.subheader(f'Histograma de LogRendimientos : {ticker}')
fig2,ax = plt.subplots(figsize =(10,5))
ax.hist(df_rendimientos,bins=50,alpha=0.5,color = 'blue')
ax.set_title('Histograma')
ax.set_xlabel('Rendimiento Diario')
ax.set_ylabel('Frecuencia')
st.pyplot(fig2)

#VaR y ES inciso c)
st.header(f"Medidas de Riesgo de {ticker} ")
st.subheader("VaR a 95%, 97.5% y 99%")

tabla1= pd.DataFrame(
    {
        "% Confianza": ["95%", "97.5%", "99%"],
        "Paramétrico N": [CR.VaR_95, CR.VaR_975, CR.VaR_99],
        "Paramétrico T": [CR.VaR_95_t,CR.VaR_975_t, CR.VaR_99_t],
        "Histórico" : [CR.hVaR_95, CR.hVaR_975,CR.hVaR_99],
        "Monte Carlo" : [CR.MCVaR_95, CR.MCVaR_975,CR.MCVaR_99],
    }
)

st.dataframe(tabla1,hide_index = True)

st.subheader("ES a 95%, 97.5% y 99%")

tabla2= pd.DataFrame(
    {
        "% Confianza": ["95%", "99%", "99%"],
        "Paramétrico N": [CR.ES_95, CR.ES_975, CR.ES_99],
        "Paramétrico T": [CR.ES_95_t,CR.ES_975_t, CR.ES_99_t],
        "Histórico" : [CR.hES_95, CR.hES_975,CR.hES_99],
        "Monte Carlo" : [CR.MCES_95, CR.MCES_975,CR.MCES_99],
    }
)

st.dataframe(tabla2,hide_index = True)

#Histograma donde se muestra VaR y ES-----------------------------------------------------------------
st.subheader("Histogramas de las Medidas de Riesgo")

st.text("Se marcarán los rendimientos que superen al VaR histórico con color rojo")

fig3,ax = plt.subplots(figsize =(10,5))
n, bins, patches = plt.hist(df_rendimientos, bins=50, color='blue', alpha=0.7, label='Returns')


# Identificamos datos menores al VaR histórico y los marcamos de rojo
for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
    if bin_left < CR.hVaR_95:
        patch.set_facecolor('red')

# Graficamos las diferentes medidas
plt.axvline(x=CR.VaR_95_t, color='skyblue', linestyle='--', label='VaR 95% (Paramétrico t)')
plt.axvline(x=CR.MCVaR_95, color='grey', linestyle='--', label='VaR 95% (Monte Carlo)')
plt.axvline(x=CR.hVaR_95, color='green', linestyle='--', label='VaR 95% (Histórico)')
plt.axvline(x=CR.ES_95_t, color='purple', linestyle='-.', label='ES (Paramétrico t) 95%')

#Agregamos leyenda
plt.title('Histograma de Rendimientos con VaR y ES a 95%')
plt.xlabel('Rendimientos')
plt.ylabel('Frecuencia')
plt.legend()

st.pyplot(fig3)

#---------------------------------------------------------------------------------------

fig4,ax = plt.subplots(figsize =(10,5))
n, bins, patches = plt.hist(df_rendimientos, bins=50, color='blue', alpha=0.7, label='Returns')


for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
    if bin_left < CR.hVaR_975:
        patch.set_facecolor('red')

# Graficamos las medidas
plt.axvline(x=CR.VaR_975_t, color='skyblue', linestyle='--', label='VaR 97.5% (Paramétrico t)')
plt.axvline(x=CR.MCVaR_975, color='grey', linestyle='--', label='VaR 97.5% (Monte Carlo)')
plt.axvline(x=CR.hVaR_975, color='green', linestyle='--', label='VaR 97.5% (Histórico)')
plt.axvline(x=CR.ES_975_t, color='purple', linestyle='-.', label='ES (Paramétrico t) 97.5%')

# Nombres de ejes
plt.title('Histograma de Rendimientos con VaR y ES a 97.5%')
plt.xlabel('Rendimientos')
plt.ylabel('Frecuencia')
plt.legend()

st.pyplot(fig4)


fig5,ax = plt.subplots(figsize =(10,5))
n, bins, patches = plt.hist(df_rendimientos, bins=50, color='blue', alpha=0.7, label='Returns')

for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
    if bin_left < CR.hVaR_99:
        patch.set_facecolor('red')

#Graficamos los datos
plt.axvline(x=CR.VaR_99_t, color='skyblue', linestyle='--', label='VaR 99% (Paramétrico t)')
plt.axvline(x=CR.MCVaR_99, color='grey', linestyle='--', label='VaR 99% (Monte Carlo)')
plt.axvline(x=CR.hVaR_99, color='green', linestyle='--', label='VaR 99% (Histórico)')
plt.axvline(x=CR.ES_99_t, color='purple', linestyle='-.', label='ES (Paramétrico t) 99%')

# Nombres de ejes
plt.title('Histograma de Rendimientos con VaR y ES a 99%')
plt.xlabel('Rendimientos')
plt.ylabel('Frecuencia')
plt.legend()

st.pyplot(fig5)

#d)) Rolling windows VaR----------------------------------------------------------------------------------------

st.header("VaR y ES con Rolling Windows")
st.subheader("VaR histórico y paramétrico")

#poner eleccion 
intervalos = ["95","99"]
VAR_seleccionado = st.selectbox('Selecciona (%) de confianza ',intervalos)

if VAR_seleccionado:

    columna1 = f'{VAR_seleccionado}% VaR Rolling'
    columna2 = f'ES Rolling {VAR_seleccionado}%'
    dfc_name = f"VaR_{VAR_seleccionado}_rolling_df"  # "VaR_95_rolling_df"

    # Obtener el DataFrame dinámicamente
    dfc = getattr(CR, dfc_name)

    # Graficamos rendimientos
    fig6,ax = plt.subplots(figsize = (10,5))
    ax.plot(df_rendimientos.index,df_rendimientos * 100, label='Rendimientos diarios (%)', color='blue', alpha=0.5)
    ax.axhline(y = 0,linestyle = '-',alpha = 0.7)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Rendimiento Diario")

    # Graficamos los Rolling VaR
    plt.plot(df_rendimientos.index,dfc[columna1], label=columna1, color='red')
    #plt.plot(df_rendimientos.index,CR.VaR_99_rolling_df[columna], label= columna, color='purple')
    plt.plot(df_rendimientos.index, dfc[columna2], label=columna2, color='purple')
    #plt.plot(df_rendimientos.index, CR.hES_95_rolling_df['hES Rolling 95%'], label='hES Rolling 95%', color='black')


    # Add a title and axis labels
    plt.title('Rendimientos diarios y and 95% Rolling VaR')
    plt.xlabel('Date')
    plt.ylabel('Values (%)')

    # Add a legend
    plt.legend()

    # Show the plot
    #plt.tight_layout()
    #plt.show()
    st.pyplot(fig6)

#inciso e) ------------------------------------------------------------------------

st.subheader("Eficiencia de estimaciones")
st.dataframe(CR.df_final.style.background_gradient(cmap='Reds',subset=['Proporción 95%','Proporción 99%']),hide_index = True)

#inciso f) ------------------------------------------------------------------------------------------
st.subheader("Eficiencia de aproximación")
st.dataframe(CR.tabla_violaciones.style.applymap(MCF.highlight_high_values, subset=['Porcentaje Violaciones (%)']),hide_index = True)