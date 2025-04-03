
#Librerías y/o paquetrías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #Visualizacion de datos
import yfinance as yf #Api de Yahoo Finanzas
from scipy.stats import kurtosis, skew, shapiro ,norm,t #Funciones estadísticas
from datetime import date
import Funciones as MCF


"""#*Código principal*#"""

#Código
#inciso a) Descarga de información de un activo
ticker = "NVDA"
df_precios = MCF.obtener_datos(ticker)
df_precios.iloc[:,0]
empresa = yf.Ticker(ticker)
nombre = empresa.info['longName']

#b) Calcular rendimientos
df_rendimientos_log = MCF.calcular_rendimientos_Log(df_precios)
#df_rendimientos_log = calcular_rendimientos(df_precios)
print(df_rendimientos_log)

#media
media = df_rendimientos_log.mean()
print(media)

#sesgo
sesgo = skew(df_rendimientos_log)
print(sesgo)

#curtosis
curtosis = kurtosis(df_rendimientos_log)
print(curtosis)

#VaR método paramétrico Normal--------------------------------------------------------------------------------------------------------------------------
mu = media
sigma = np.std(df_rendimientos_log)

VaR_95 = norm.ppf(1-0.95,mu,sigma)

print(type(VaR_95))
print(type(mu))
print(type(sigma))

'''
c)
Calcula el VaR y ES para la serie completa de datos a los siguientes intervalos de confianza:
α = 0,95, 0,975, y 0,99 bajo una aproximación param ́etrica asumiendo una distribuci ́on normal y t-student,
además bajo una aproximación histórica y Monte Carlo.
'''
#VaR método paramétrico Normal--------------------------------------------------------------------------------------------------------------------------
mu = media
sigma = np.std(df_rendimientos_log)

VaR_95 = norm.ppf(1-0.95,mu,sigma)
VaR_975 = norm.ppf(1-0.975,mu,sigma)
VaR_99 = norm.ppf(1-0.99,mu,sigma)

print("El 95% VaR de", nombre , "es:", round(VaR_95*100,4))
print("El 975% VaR de", nombre , "es:", round(VaR_975*100,4))
print("El 99% VaR de", nombre , "es:", round(VaR_99*100,4))

#VaR método paramétrico T-student
gl = len(df_rendimientos_log)-1
VaR_95_t = t.ppf(1-0.95,gl,mu,sigma)
VaR_975_t = t.ppf(1-0.975,gl,mu,sigma)
VaR_99_t = t.ppf(1-0.99,gl,mu,sigma)

print("\nEl 95% VaR de", nombre , "es:", round(VaR_95_t*100,4))
print("El 975% VaR de", nombre , "es:", round(VaR_975_t*100,4))
print("El 99% VaR de", nombre , "es:", round(VaR_99_t*100,4))

#VaR histórico
hVaR_95 = df_rendimientos_log.quantile(0.05)
hVaR_975 = df_rendimientos_log.quantile(0.025)
hVaR_99 = df_rendimientos_log.quantile(0.001)

print("\nEl 95% VaR de", nombre , "es:", round(hVaR_95*100,4))
print("El 975% VaR de", nombre , "es:", round(hVaR_975*100,4))
print("El 99% VaR de", nombre , "es:", round(hVaR_99*100,4))

#VaR Monte Carlo         Está en normal pero chance podemos cambiarla a una t
n_sim = 1000000
sim_returns = np.random.normal(mu,sigma ,n_sim)

MCVaR_95 = np.percentile(sim_returns, 5)
MCVaR_975 = np.percentile(sim_returns, 2.5)
MCVaR_99 = np.percentile(sim_returns, 0.1)

print("\nEl 95% VaR de", nombre , "es:", round(MCVaR_95*100,4))
print("El 975% VaR de", nombre , "es:", round(MCVaR_975*100,4))
print("El 99% VaR de", nombre , "es:", round(MCVaR_99*100,4))

#ES paramétrico normal
ES_95 = df_rendimientos_log[df_rendimientos_log<= VaR_95].mean()
ES_975 = df_rendimientos_log[df_rendimientos_log<= VaR_975].mean()
ES_99 = df_rendimientos_log[df_rendimientos_log<= VaR_99].mean()

print("\nEl 95% Expected Shorfall de", nombre , "es:", round(ES_95*100,4))
print("El 975% Expected Shorfall de", nombre , "es:", round(ES_975*100,4))
print("El 99% Expected Shorfall de", nombre , "es:", round(ES_99*100,4))

#ES histórico
hES_95 = df_rendimientos_log[df_rendimientos_log<= hVaR_95].mean()
hES_975 = df_rendimientos_log[df_rendimientos_log<= hVaR_975].mean()
hES_99 = df_rendimientos_log[df_rendimientos_log<= hVaR_99].mean()

print("\nEl 95% Expected Shorfall de", nombre , "es:", round(hES_95*100,4))
print("El 975% Expected Shorfall de", nombre , "es:", round(hES_975*100,4))
print("El 99% Expected Shorfall de", nombre , "es:", round(hES_99*100,4))

#ES parámetrico T-student
ES_95_t = df_rendimientos_log[df_rendimientos_log<= VaR_95_t].mean()
ES_975_t = df_rendimientos_log[df_rendimientos_log<= VaR_975_t].mean()
ES_99_t = df_rendimientos_log[df_rendimientos_log<= VaR_99_t].mean()

print("\nEl 95% Expected Shorfall de", nombre , "es:", round(ES_95_t*100,4))
print("El 975% Expected Shorfall de", nombre , "es:", round(ES_975_t*100,4))
print("El 99% Expected Shorfall de", nombre , "es:", round(ES_99_t*100,4))

#ES Monte Carlo
MCES_95 = df_rendimientos_log[df_rendimientos_log<= MCVaR_95].mean()
MCES_975 = df_rendimientos_log[df_rendimientos_log<= MCVaR_975].mean()
MCES_99 = df_rendimientos_log[df_rendimientos_log<= MCVaR_99].mean()

print("\nEl 95% Expected Shorfall de", nombre , "es:", round(MCES_95*100,4))
print("El 975% Expected Shorfall de", nombre , "es:", round(MCES_975*100,4))
print("El 99% Expected Shorfall de", nombre , "es:", round(MCES_99*100,4))

"""##Inciso d)##
En una sola gráfica muestra las ganancias y ṕerdidas además del VaR y el ES con α = 0,95 y 0,99 con una rolling window de 252 retornos.
"""

# VaR Rolling método paramétrico Normal
mu_roll = df_rendimientos_log.rolling(window=252).mean()
sigma_roll = df_rendimientos_log.rolling(window=252).std()

VaR_95_rolling = norm.ppf(1-0.95, mu_roll, sigma_roll)
VaR_99_rolling = norm.ppf(1-0.99, mu_roll, sigma_roll)

VaR_95_rolling_porcentaje = (VaR_95_rolling * 100).round(4)
VaR_99_rolling_porcentaje = (VaR_99_rolling * 100).round(4)

VaR_95_rolling_df = pd.DataFrame({'Fecha': df_rendimientos_log.index, '95% VaR Rolling': VaR_95_rolling_porcentaje.squeeze()})
VaR_99_rolling_df = pd.DataFrame({'Fecha': df_rendimientos_log.index, '99% VaR Rolling': VaR_99_rolling_porcentaje.squeeze()})

print(VaR_95_rolling_df.tail())
### print(vaR_99_rolling_df.tail()) ###

#VaR Rolling histórico
hVaR_95_r = df_rendimientos_log.rolling(window=252).quantile(0.05)
hVaR_99_r = df_rendimientos_log.rolling(window=252).quantile(0.001)

hVaR_95_r_porcentaje = (hVaR_95_r * 100).round(4)
hVaR_99_r_porcentaje = (hVaR_99_r * 100).round(4)

hVaR_95_r_df = pd.DataFrame({'Fecha': df_rendimientos_log.index, '95% hVaR Rolling': hVaR_95_r_porcentaje.squeeze()})
hVaR_99_r_df = pd.DataFrame({'Fecha': df_rendimientos_log.index, '99% hVaR Rolling': hVaR_99_r_porcentaje.squeeze()})

print(hVaR_95_r_df.tail())
### print(hVaR_99_r_df.tail()) ###

#ES Rolling paramétrico normal
ES_95_rolling = df_rendimientos_log[df_rendimientos_log<= VaR_95_rolling].mean()
ES_99_rolling = df_rendimientos_log[df_rendimientos_log<= VaR_99_rolling].mean()

ES_95_roll_porcentaje = (ES_95_rolling * 100).round(4)
ES_99_roll_porcentaje = (ES_99_rolling * 100).round(4)

ES_95_roll_df = pd.DataFrame({'Fecha': df_rendimientos_log.index, '95% ES Rolling': ES_95_roll_porcentaje.squeeze()})
ES_99_roll_df = pd.DataFrame({'Fecha': df_rendimientos_log.index, '99% ES Rolling': ES_99_roll_porcentaje.squeeze()})

print(ES_95_roll_df.tail())
### print(ES_99_roll_df.tail()) ###

#ES Rolling histórico
hES_95_rolling = df_rendimientos_log[df_rendimientos_log<= hVaR_95_r].mean()
hES_99_rolling = df_rendimientos_log[df_rendimientos_log<= hVaR_99_r].mean()

hES_95_r_porcentaje = (hES_95_rolling * 100).round(4)
hES_99_r_porcentaje = (hES_99_rolling * 100).round(4)

hES_95_r_df = pd.DataFrame({'Fecha': df_rendimientos_log.index, '95% hES Rolling': hES_95_r_porcentaje.squeeze()})
hES_99_r_df = pd.DataFrame({'Fecha': df_rendimientos_log.index, '99% hES Rolling': hES_99_r_porcentaje.squeeze()})

print(hES_95_r_df.tail())
### print(hVaR_99_r_df.tail()) ###

#e) Eficiencia de estimaciones-----------------------------------------------------------------------------

# Desplazar los rendimientos del stock al día siguiente (t+1)
rendimientos_back = df_rendimientos_log.shift(-1)*100


series_95 = [VaR_95_rolling_df, hVaR_95_r_df, ES_95_roll_df, hES_95_r_df]
series_99 = [VaR_99_rolling_df, hVaR_99_r_df, ES_99_roll_df, hES_99_r_df]
alfas = [series_95, series_99]


# Crear una lista para almacenar los resultados y las medidas
tabla_resultados = []

# Procesar las series de alfas
for i, series in enumerate(alfas):
    for j, df in enumerate(series):
        print('\n Medida de Riesgo: ', df.columns[1])

        # Cambiamos el índice de los rolling para poder concatenar después
        df.index = pd.to_datetime(df['Fecha'])
        df2 = pd.concat([df, rendimientos_back], axis=1)

        # Crea un nuevo DF solamente con las excedencias al VaR
        comparison = df2[ df2[df2.columns[2]] < df2[df2.columns[1]] ]
        # Cuenta cuántas excedencias hay
        excede = comparison.shape[0]
        excede_prop = 100 * excede / df.shape[0]

        print('Número de Excedencias: ', excede)
        print('Proporción de Excedencias (%): ', round(excede_prop, 4))

        # Agregar los resultados a la lista
        medida = df.columns[1]
        if i == 0:  # Para 95%
            tabla_resultados.append([medida, excede, round(excede_prop, 4), None, None])
        else:  # Para 99%
            tabla_resultados[-len(series) + j][3] = excede
            tabla_resultados[-len(series) + j][4] = round(excede_prop, 4)

# Crear el DataFrame directamente desde la lista de resultados
df_final = pd.DataFrame(tabla_resultados, columns=['Medida de Riesgo', 'Excedencias 95%', 'Proporción 95%', 'Excedencias 99%', 'Proporción 99%'])

df_final
