import numpy as np

######## PROCESAMIENTO DE SEÑAL

def normalizar_señales(matriz):
    """
    Normaliza una matriz de señales ECG. Para cada señal (columna), 
    resta la media y divide por la desviación estándar, excepto la primera columna (tiempo).
    
    Parámetros:
    matriz : numpy.ndarray - Matriz de tamaño (n, 13), donde cada columna representa una señal.
        
        
    Retorna:
    numpy.ndarray - Matriz normalizada donde la primera columna queda intacta 
                    y las demás tienen media 0 y desviación estándar 1.
    """
    # Copiar la matriz original para no modificarla directamente
    matriz_normalizada = matriz.copy()

    # Calcular la media y la desviación estándar para cada columna (excepto la primera)
    media = np.mean(matriz[:, 1:], axis=0)
    desviacion_estandar = np.std(matriz[:, 1:], axis=0)

    # Evitar división por cero: si la desviación estándar es 0, reemplazar por 1
    desviacion_estandar[desviacion_estandar == 0] = 1  

    # Normalizar todas las columnas excepto la primera (tiempo)
    matriz_normalizada[:, 1:] = (matriz[:, 1:] - media) / desviacion_estandar

    # Calcular offsets solo para las columnas de señal (omitimos la primera)
    # Centrar cada columna para que su media sea 0 (excepto tiempo)
    media_col = np.mean(matriz_normalizada[:, 1:], axis=0)
    matriz_normalizada[:, 1:] -= media_col
    
    return matriz_normalizada

def ecg_calcul_normalized(ecg_aiso):
    """
    Calcula las 12 derivaciones del ECG a partir del fichero ecg_aiso con los potenciales
    proporcionados por elvira y normaliza las señales resultantes.
    
    Parámetros:
    ecg_aiso : str - Ruta al archivo de potenciales
    
    Retorna:
    numpy.ndarray - Matriz con las 12 derivaciones normalizadas del ECG
    """
    # Cargar el archivo de datos
    ecg_elvira = np.loadtxt(ecg_aiso, skiprows=1)

    # Crear un array vacío para almacenar las derivaciones
    ECG = np.zeros((ecg_elvira.shape[0], 13))

    # Asignar la primera columna como el tiempo
    ECG[:, 0] = ecg_elvira[:, 0]

    # Cálculo de las derivaciones estándar
    ECG[:, 1] = ecg_elvira[:, 7] - ecg_elvira[:, 8]  # I = LA - RA
    ECG[:, 2] = ecg_elvira[:, 9] - ecg_elvira[:, 8]  # II = LL - RA
    ECG[:, 3] = ecg_elvira[:, 9] - ecg_elvira[:, 7]  # III = LL - LA

    # Cálculo de las derivaciones aumentadas
    ECG[:, 4] = ecg_elvira[:, 8] - 0.5 * (ecg_elvira[:, 7] + ecg_elvira[:, 9])  # aVR = RA - (1/2)(LA + LL)
    ECG[:, 5] = ecg_elvira[:, 7] - 0.5 * (ecg_elvira[:, 8] + ecg_elvira[:, 9])  # aVL = LA - (1/2)(RA + LL)
    ECG[:, 6] = ecg_elvira[:, 9] - 0.5 * (ecg_elvira[:, 8] + ecg_elvira[:, 7])  # aVF = LL - (1/2)(RA + LA)

    # Cálculo de las derivaciones precordiales
    ECG[:, 7] = ecg_elvira[:, 1] - (1/3) * (ecg_elvira[:, 8] + ecg_elvira[:, 7] + ecg_elvira[:, 9])  # V1
    ECG[:, 8] = ecg_elvira[:, 2] - (1/3) * (ecg_elvira[:, 8] + ecg_elvira[:, 7] + ecg_elvira[:, 9])  # V2
    ECG[:, 9] = ecg_elvira[:, 3] - (1/3) * (ecg_elvira[:, 8] + ecg_elvira[:, 7] + ecg_elvira[:, 9])  # V3
    ECG[:, 10] = ecg_elvira[:, 4] - (1/3) * (ecg_elvira[:, 8] + ecg_elvira[:, 7] + ecg_elvira[:, 9])  # V4
    ECG[:, 11] = ecg_elvira[:, 5] - (1/3) * (ecg_elvira[:, 8] + ecg_elvira[:, 7] + ecg_elvira[:, 9])  # V5
    ECG[:, 12] = ecg_elvira[:, 6] - (1/3) * (ecg_elvira[:, 8] + ecg_elvira[:, 7] + ecg_elvira[:, 9])  # V6

    # Normalizar las señales resultantes
    ECG = normalizar_señales(ECG)
    return ECG
 
def ecg_calcul(ecg_aiso):
    """
    Calcula las 12 derivaciones del ECG a partir del fichero ecg_aiso con los potenciales
    proporcionados por elvira, sin normalizar los resultados.
    
    Parámetros:
    ecg_aiso : str - Ruta al archivo de potenciales
    
    Retorna:
    numpy.ndarray - Matriz con las 12 derivaciones del ECG sin normalizar
    """
    # Cargar el archivo de datos
    ecg_elvira = np.loadtxt(ecg_aiso, skiprows=1)

    # Crear un array vacío para almacenar las derivaciones
    ECG = np.zeros((ecg_elvira.shape[0], 13))

    # Asignar la primera columna como el tiempo
    ECG[:, 0] = ecg_elvira[:, 0]

    # Cálculo de las derivaciones estándar
    ECG[:, 1] = ecg_elvira[:, 7] - ecg_elvira[:, 8]  # I = LA - RA
    ECG[:, 2] = ecg_elvira[:, 9] - ecg_elvira[:, 8]  # II = LL - RA
    ECG[:, 3] = ecg_elvira[:, 9] - ecg_elvira[:, 7]  # III = LL - LA

    # Cálculo de las derivaciones aumentadas
    ECG[:, 4] = ecg_elvira[:, 8] - 0.5 * (ecg_elvira[:, 7] + ecg_elvira[:, 9])  # aVR = RA - (1/2)(LA + LL)
    ECG[:, 5] = ecg_elvira[:, 7] - 0.5 * (ecg_elvira[:, 8] + ecg_elvira[:, 9])  # aVL = LA - (1/2)(RA + LL)
    ECG[:, 6] = ecg_elvira[:, 9] - 0.5 * (ecg_elvira[:, 8] + ecg_elvira[:, 7])  # aVF = LL - (1/2)(RA + LA)

    # Cálculo de las derivaciones precordiales
    ECG[:, 7] = ecg_elvira[:, 1] - (1/3) * (ecg_elvira[:, 8] + ecg_elvira[:, 7] + ecg_elvira[:, 9])  # V1
    ECG[:, 8] = ecg_elvira[:, 2] - (1/3) * (ecg_elvira[:, 8] + ecg_elvira[:, 7] + ecg_elvira[:, 9])  # V2
    ECG[:, 9] = ecg_elvira[:, 3] - (1/3) * (ecg_elvira[:, 8] + ecg_elvira[:, 7] + ecg_elvira[:, 9])  # V3
    ECG[:, 10] = ecg_elvira[:, 4] - (1/3) * (ecg_elvira[:, 8] + ecg_elvira[:, 7] + ecg_elvira[:, 9])  # V4
    ECG[:, 11] = ecg_elvira[:, 5] - (1/3) * (ecg_elvira[:, 8] + ecg_elvira[:, 7] + ecg_elvira[:, 9])  # V5
    ECG[:, 12] = ecg_elvira[:, 6] - (1/3) * (ecg_elvira[:, 8] + ecg_elvira[:, 7] + ecg_elvira[:, 9])  # V6
    
    return ECG
    