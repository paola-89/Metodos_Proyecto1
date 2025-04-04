
#Librerías y/o paquetrías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #Visualizacion de datos
import yfinance as yf #Api de Yahoo Finanzas
from scipy.stats import kurtosis, skew, shapiro ,norm,t #Funciones estadísticas
from datetime import date
import streamlit as st


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


def highlight_high_values(val):
    """Resalta en rojo si el valor de la columna 'Porcentaje Violaciones (%)' es mayor a 2.5"""
    color = "lightcoral" if val > 2.5 else "white"
    return f"background-color: {color}"

def generarMenu():
    with st.sidebar:
        st.page_link('proyecto1_mercados.py',label="Análisis")
        st.page_link('nombres.py',label="Equipo")
