
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

VaR_95_rolling_df.set_index('Fecha', inplace=True)
VaR_99_rolling_df.set_index('Fecha', inplace=True)

print(VaR_95_rolling_df.tail())
### print(vaR_99_rolling_df.tail()) ###

#VaR Rolling histórico
hVaR_95_r = df_rendimientos_log.rolling(window=252).quantile(0.05)
hVaR_99_r = df_rendimientos_log.rolling(window=252).quantile(0.001)

hVaR_95_r_porcentaje = (hVaR_95_r * 100).round(4)
hVaR_99_r_porcentaje = (hVaR_99_r * 100).round(4)

hVaR_95_r_df = pd.DataFrame({'Fecha': df_rendimientos_log.index, '95% hVaR Rolling': hVaR_95_r_porcentaje.squeeze()})
hVaR_99_r_df = pd.DataFrame({'Fecha': df_rendimientos_log.index, '99% hVaR Rolling': hVaR_99_r_porcentaje.squeeze()})

hVaR_95_r_df.set_index('Fecha', inplace=True)
hVaR_99_r_df.set_index('Fecha', inplace=True)

print(hVaR_95_r_df.tail())
### print(hVaR_99_r_df.tail()) ###

tam_ventana = 252
ES_95_rolling = [np.nan] * tam_ventana
ES_99_rolling = [np.nan] * tam_ventana

for i in range(tam_ventana, len(df_rendimientos_log)):
    ventana = df_rendimientos_log.iloc[i-tam_ventana : i]

    mu_for = ventana.mean()
    sigma_for = np.std(ventana)

    var_95 = norm.ppf(1-0.95,mu_for,sigma_for)
    var_99 = norm.ppf(1-0.99,mu_for,sigma_for)

    es_95 = ventana[ventana <= var_95].mean()
    es_99 = ventana[ventana <= var_99].mean()

    ES_95_rolling.append((es_95 * 100).round(4).squeeze())
    ES_99_rolling.append((es_99 * 100))


ES_95_rolling_df = pd.DataFrame({'Fecha': df_rendimientos_log.index, 'ES Rolling 95%': ES_95_rolling})
ES_95_rolling_df.set_index('Fecha', inplace=True)

ES_99_rolling_df = pd.DataFrame({'Fecha': df_rendimientos_log.index, 'ES Rolling 99%': ES_99_rolling})
ES_99_rolling_df.set_index('Fecha', inplace=True)


#Histórico
tam_ventana = 252
hES_95_rolling = [np.nan] * tam_ventana
hES_99_rolling = [np.nan] * tam_ventana

for i in range(tam_ventana, len(df_rendimientos_log)):
    ventana = df_rendimientos_log.iloc[i-tam_ventana : i]

    hvar_95 = ventana.quantile(0.05)
    hvar_99 = ventana.quantile(0.001)

    es_95 = ventana[ventana <= hvar_95].mean()
    es_99 = ventana[ventana <= hvar_99].mean()

    hES_95_rolling.append((es_95 * 100).round(4).squeeze())
    hES_99_rolling.append((es_99 * 100))


hES_95_rolling_df = pd.DataFrame({'Fecha': df_rendimientos_log.index, 'hES Rolling 95%': hES_95_rolling})
hES_95_rolling_df.set_index('Fecha', inplace=True)

hES_99_rolling_df = pd.DataFrame({'Fecha': df_rendimientos_log.index, 'hES Rolling 99%': hES_99_rolling})
hES_99_rolling_df.set_index('Fecha', inplace=True)

print(hES_95_rolling_df)
print(hES_99_rolling_df)

ES_95_rolling = df_rendimientos_log[df_rendimientos_log<= VaR_95_rolling].mean()
ES_99_rolling = df_rendimientos_log[df_rendimientos_log<= VaR_99_rolling].mean()

ES_95_roll_porcentaje = (ES_95_rolling * 100).round(4)
ES_99_roll_porcentaje = (ES_99_rolling * 100).round(4)

ES_95_roll_df = pd.DataFrame({'Fecha': df_rendimientos_log.index, 'ES Rolling 95%': ES_95_roll_porcentaje.squeeze()})
ES_99_roll_df = pd.DataFrame({'Fecha': df_rendimientos_log.index, 'ES Rolling 99%': ES_99_roll_porcentaje.squeeze()})

ES_95_roll_df.set_index('Fecha', inplace=True)
ES_99_roll_df.set_index('Fecha', inplace=True)

print(ES_99_roll_df.tail())
### print(ES_99_roll_df.tail()) ###

#ES Rolling histórico
hES_95_rolling = df_rendimientos_log[df_rendimientos_log<= hVaR_95_r].mean()
hES_99_rolling = df_rendimientos_log[df_rendimientos_log<= hVaR_99_r].mean()

hES_95_r_porcentaje = (hES_95_rolling * 100).round(4)
hES_99_r_porcentaje = (hES_99_rolling * 100).round(4)

hES_95_r_df = pd.DataFrame({'Fecha': df_rendimientos_log.index, 'hES Rolling 95%': hES_95_r_porcentaje.squeeze()})
hES_99_r_df = pd.DataFrame({'Fecha': df_rendimientos_log.index, 'hES Rolling 99%': hES_99_r_porcentaje.squeeze()})

hES_95_r_df.set_index('Fecha', inplace=True)
hES_99_r_df.set_index('Fecha', inplace=True)

print(hES_95_r_df.tail())
### print(hVaR_99_r_df.tail()) ###

plt.figure(figsize=(12,8))

# Graficamos los retornos diarios
plt.plot(df_rendimientos_log.index, df_rendimientos_log * 100, label='Retornos Diarios (%)', color='blue', alpha=0.5)

# Graficamos Rolling VaR, Rolling hVaR, Rolling ES y Rolling hES al 95%
plt.plot(VaR_95_rolling_df.index, VaR_95_rolling_df['95% VaR Rolling'], label='VaR Rolling 95%', color='red')
plt.plot(hVaR_95_r_df.index, hVaR_95_r_df['95% hVaR Rolling'], label='hVaR Rolling 95%', color='green')
plt.plot(ES_95_rolling_df.index, ES_95_rolling_df['ES Rolling 95%'], label='ES Rolling 95%', color='purple')
plt.plot(hES_95_rolling_df.index, hES_95_rolling_df['hES Rolling 95%'], label='hES Rolling 95%', color='black')

plt.title('Retornos Diarios (%) y VaR Rolling 95%')
plt.xlabel('Fecha')
plt.ylabel('Valores (%)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,8))

# Graficamos los retornos diarios
plt.plot(df_rendimientos_log.index, df_rendimientos_log * 100, label='Retornos Diarios (%)', color='blue', alpha=0.5)

# Graficamos Rolling VaR, Rolling hVaR, Rolling ES y Rolling hES al 99%
plt.plot(VaR_99_rolling_df.index, VaR_99_rolling_df['99% VaR Rolling'], label='VaR Rolling 99%', color='red')
plt.plot(hVaR_99_r_df.index, hVaR_99_r_df['99% hVaR Rolling'], label='hVaR Rolling 95%', color='green')
plt.plot(ES_99_rolling_df.index, ES_99_rolling_df['ES Rolling 99%'], label='ES Rolling 99%', color='purple')
plt.plot(hES_99_rolling_df.index, hES_99_rolling_df['hES Rolling 99%'], label='hES Rolling 99%', color='black')

plt.title('Retornos Diarios (%) y VaR Rolling 99%')
plt.xlabel('Fecha')
plt.ylabel('Valores (%)')
plt.legend()
plt.tight_layout()
plt.show()

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
        print('\n Medida de Riesgo: ', df.columns[0])

        # Cambiamos el índice de los rolling para poder concatenar después
        #df.index = pd.to_datetime(df['Fecha'])
        df2 = pd.concat([df, rendimientos_back], axis=1)

        # Crea un nuevo DF solamente con las excedencias al VaR
        comparison = df2[ df2[df2.columns[2]] < df2[df2.columns[1]] ]
        # Cuenta cuántas excedencias hay
        excede = comparison.shape[0]
        excede_prop = 100 * excede / df.shape[0]

        print('Número de Excedencias: ', excede)
        print('Proporción de Excedencias (%): ', round(excede_prop, 4))

        # Agregar los resultados a la lista
        medida = df.columns[0]
        if i == 0:  # Para 95%
            tabla_resultados.append([medida, excede, round(excede_prop, 4), None, None])
        else:  # Para 99%
            tabla_resultados[-len(series) + j][3] = excede
            tabla_resultados[-len(series) + j][4] = round(excede_prop, 4)

# Crear el DataFrame directamente desde la lista de resultados
df_final = pd.DataFrame(tabla_resultados, columns=['Medida de Riesgo', 'Excedencias 95%', 'Proporción 95%', 'Excedencias 99%', 'Proporción 99%'])

df_final

#Inciso f)

# Dados los datos en el enunciado del inciso, tenemos que:

#Calculamos el rolling mean y la desviación estándar, considerando la ventana de 252 días
rolling_std = df_rendimientos_log.rolling(window=252).std()
rolling_mean = df_rendimientos_log.rolling(window=252).mean()

#Calculamos el VaR con volatilidad móvil:
VaR_95_rolling = norm.ppf(1-0.95, rolling_mean, rolling_std)
VaR_99_rolling = norm.ppf(1-0.99, rolling_mean, rolling_std)

#Convertimos a porcentaje y redondeamos con 4 décimas
VaR_95_rolling_percent = (VaR_95_rolling * 100).round(4)
VaR_99_rolling_percent = (VaR_99_rolling * 100).round(4)

#DataFrame:
VaR_rolling_df = pd.DataFrame({'Date': df_rendimientos_log.index,'95% VaR Rolling': VaR_95_rolling_percent,'99% VaR Rolling': VaR_99_rolling_percent})
VaR_rolling_df.set_index('Date', inplace=True)

print(VaR_rolling_df.tail())

plt.figure(figsize=(14, 7))

#plot con retornos diarios:
plt.plot(df_rendimientos_log.index, df_rendimientos_log * 100, label='Rendimientos diarios (%)', color='cadetblue', alpha=0.6)

#plot con 95% y 99% rolling VaR con volatilidad móvil
plt.plot(VaR_rolling_df.index, VaR_rolling_df['95% VaR Rolling'], label='VaR 95% móvil', color='red')
plt.plot(VaR_rolling_df.index, VaR_rolling_df['99% VaR Rolling'], label='VaR 99% móvil', color='purple')

#Títulos:
plt.title(f'Rendimientos diarios y VaR Móvil de {nombre} (ventana 252 retornos)')
plt.xlabel('Fecha')
plt.ylabel('Rendimiento/VaR (%)')


plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#vamos a considerar solo los días con ventana completa (sin los primeros 252 días)
valid_returns = df_rendimientos_log.iloc[251:]

# Convert VaR_95_rolling and VaR_99_rolling to pandas Series with the same index as df_rendimientos_log
VaR_95_rolling = pd.Series(VaR_95_rolling, index=df_rendimientos_log.index)
VaR_99_rolling = pd.Series(VaR_99_rolling, index=df_rendimientos_log.index)

VaR_95_valid = VaR_95_rolling.iloc[251:]
VaR_99_valid = VaR_99_rolling.iloc[251:]

#contamos violaciones:
violaciones_95 = (valid_returns < VaR_95_valid).sum()
violaciones_99 = (valid_returns < VaR_99_valid).sum()
total_dias_validos = len(valid_returns)

#porcentaje de violaciones:
porcentaje_violaciones_95 = (violaciones_95 / total_dias_validos) * 100
porcentaje_violaciones_99 = (violaciones_99 / total_dias_validos) * 100

#DataFrame:
tabla_violaciones = pd.DataFrame({
    'Nivel de Confianza': ['95%', '99%'],
    'Días Totales': [total_dias_validos, total_dias_validos],
    'Violaciones Esperadas': [0.05 * total_dias_validos, 0.01 * total_dias_validos],
    'Violaciones Observadas': [violaciones_95, violaciones_99],
    'Porcentaje Violaciones (%)': [porcentaje_violaciones_95, porcentaje_violaciones_99]
})

#Le anexamos la evaluación que conlleva el criterio: Una buena estimación genera
#un porcentaje de violaciones menores al 2.5 %
tabla_violaciones['Evaluación Modelo'] = [
    "Adecuado" if abs(violaciones_95 - 0.05*total_dias_validos) <= 2*np.sqrt(0.05*total_dias_validos) else "Revisar",
    "Adecuado" if abs(violaciones_99 - 0.01*total_dias_validos) <= 2*np.sqrt(0.01*total_dias_validos) else "Revisar"
]




#Título:
print("\n" + "="*42)
print("Violaciones del VaR con volatilidad móvil")
print("="*42)

#periodo:
print(f"\nPeríodo: {valid_returns.index[0].date()} a {valid_returns.index[-1].date()}")
print()
print(tabla_violaciones.to_string(index=False))