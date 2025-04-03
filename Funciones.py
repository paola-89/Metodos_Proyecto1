
#Librerías y/o paquetrías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #Visualizacion de datos
import yfinance as yf #Api de Yahoo Finanzas
#! pip install streamlit -q #Visualización en streamlit
from scipy.stats import kurtosis, skew, shapiro ,norm,t #Funciones estadísticas
#import Funciones_MCF as MCF #Utilizar solo si se estás trabajando en VSC y tienes las funciones en un archivo a parte
from datetime import date

'''
Para usar streamlit en google colab es un poco más de desmadre
pq no usas la terminal directamente como en VSC
tons no lo haré ahorita, pero de acá sale
https://medium.com/@yash.kavaiya3/running-streamlit-code-in-google-colab-involves-a-few-steps-c43ea0e8c0d9#:~:text=La%20segunda%20l%C3%ADnea%20(%20!,externa%20usando%20el%20comando%20wget%20.&text=La%20l%C3%ADnea%20%25%25writefile%20app,contactarnos%20si%20tienes%20alguna%20duda.
Chance pa eso sí conviene más el VSC, como q siento se entiende mejor
'''

#Funciones a ultilizar
def obtener_datos(stocks):
    '''
    El objetivo de esta funcion es descargar el precio
    de cierre de uno o varios activos en una ventana de 2010 a dia de hoy

    Input = Ticker del activo en string
    Output = DataFrame del precio del activo

    '''
    df = yf.download(stocks, start = "2010-01-01" , end = date.today())['Close']
    return df


def calcular_rendimientos(df): #Función de rendimientos simples
    '''
    Funcion de calcula los rendimientos de un activo

    Input = Data Frame de precios por activo

    Output = Data Frame de  rendimientos

    '''
    return df.pct_change().dropna()


def calcular_rendimientos_Log(df): #Función de rendimientos logarítmicos
    '''
    Funcion que calcula los rendimientos de un activo

    Input = Data Frame de precios por activo

    Output = Data Frame de  rendimientos

    '''
    rendimientos = np.log(df.iloc[:,0]) - np.log(df.iloc[:,0].shift(1))

    return rendimientos.dropna()
