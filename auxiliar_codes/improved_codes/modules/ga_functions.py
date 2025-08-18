import numpy as np
from modules.signal_processing import ecg_calcul_normalized , ecg_calcul
from modules.simulation_utils import modify_stimulation_nodes, limpiar_directorio
from modules.results_handler import save_manager_state
from modules.visualization import graficar_best
import pygad
import os
import sys
import numba
import math
import shutil
import multiprocessing
from multiprocessing import Manager
import subprocess
from carputils import tools
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pickle
from dtwParallel import dtw_functions
from scipy.spatial import distance as d

from datetime import (date, datetime)
import vtk
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.stats import pearsonr

from scipy.io import savemat
from scipy.spatial import cKDTree
import json
from vtk.util import numpy_support

from carputils import settings

from carputils import mesh
from carputils.carpio import txt

from modules.stimOpts import stimBlock
from modules.simulation import simulation_eik
from modules.functions import (create_vtk_from_nodes, smart_reader, extract_surface, threshold, 
						get_cobivecoaux, vtkWrite, addGlobalIds, farthest_point_sampling_global_ids,get_closest_global_ids, get_closest_global_ids_sorted_by_label,modify_json_entry,
						)
from modules.leadfield import (compute_leadfield, compute_ecg)

######## FUNCIONES PERSONALIZADAS PARA EL GA


def comparation_func(simulatedSignal, referenceSignal):
	"""
	Calcula la distancia euclidiana entre dos señales.
	
	Parámetros:
	simulatedSignal : numpy.ndarray - Señal simulada
	referenceSignal : numpy.ndarray - Señal de referencia
	
	Retorna:
	float - Distancia euclidiana entre las señales
	"""
	return np.linalg.norm(simulatedSignal - referenceSignal)  # Norma euclidiana

def dtw_camps(simulatedSignal, referenceSignal, derivaciones, meshVolume):
	"""
	Implementa el algoritmo Dynamic Time Warping (DTW) para calcular
	la similitud entre señales ECG, considerando restricciones específicas.
	
	Parámetros:
	simulatedSignal : numpy.ndarray - Señal ECG simulada
	referenceSignal : numpy.ndarray - Señal ECG de referencia
	derivaciones : str - 'total' para todas las derivaciones, 'precordiales' para solo V1-V6
	
	Retorna:
	tuple - (part_dtw, res) donde part_dtw es la suma de DTW por derivación y res es el error ponderado
	"""
	# Determinar qué derivaciones usar
	if derivaciones == 'total':
		first_lead_index = 1
		final_lead_index = referenceSignal.shape[1]  # 13
		nLeads = 12
	
	if derivaciones == 'precordiales':
		first_lead_index = 8
		final_lead_index = referenceSignal.shape[1]  # 13
		nLeads = 6
	
	# Parámetros del DTW
	w_max = 10.0  # Peso máximo para penalizar desvíos temporales
	max_slope = 2  # Pendiente máxima permitida para el camino de warping
	n_timestamps_1 = referenceSignal.shape[0]  # Número de instantes temporales de la señal de referencia
	n_timestamps_2 = simulatedSignal.shape[0]  # Número de instantes temporales de la señal simulada
	
	# Computes the region (in-window area using a trianglogram)
	# WARNING: versión simplificada de la función trianglogram
	max_slope_ = max_slope
	min_slope_ = 1 / max_slope_
	scale_max = (n_timestamps_2 - 1) / (n_timestamps_1 - 2)
	max_slope_ *= scale_max
	scale_min = (n_timestamps_2 - 2) / (n_timestamps_1 - 1)
	min_slope_ *= scale_min
	
	centered_scale = np.arange(n_timestamps_1) - n_timestamps_1 + 1
	lower_bound = min_slope_ * np.arange(n_timestamps_1)
	lower_bound = np.round(lower_bound, 2)
	lower_bound = np.floor(lower_bound)  # Garantiza al menos un píxel disponible
	
	upper_bound = max_slope_ * np.arange(n_timestamps_1) + 1
	upper_bound = np.round(upper_bound, 2)
	upper_bound = np.ceil(upper_bound)  # Garantiza al menos un píxel disponible
	
	region_original = np.asarray([lower_bound, upper_bound]).astype('int64')
	region_original = np.clip(region_original[:, :n_timestamps_1], 0, n_timestamps_2)  # Proyecta la región en el conjunto factible
	
	# Constantes para la función de costo
	
	part_dtw = 0.
	small_c = 0.05 * 171 / meshVolume
	
	# Calcular DTW para cada derivación independiente
	for lead_i in range(first_lead_index, final_lead_index):
		region = np.copy(region_original)
		
		x = referenceSignal[:, lead_i]  # Señal de referencia para esta derivación
		y = simulatedSignal[:, lead_i]  # Señal simulada para esta derivación
		
		# Función de distancia (cuadrática)
		dist_ = lambda x, y: (x - y) ** 2
	
		# Matriz de costos con ventana (infinito fuera de la región permitida)
		cost_mat = np.full((n_timestamps_1, n_timestamps_2), np.inf)
		m = np.amax(cost_mat.shape)
		
		# Calcular costos dentro de la región permitida
		for i in numba.prange(n_timestamps_1):
			for j in numba.prange(region[0, i], region[1, i]):
				# Costo ponderado: penaliza más las deformaciones temporales al inicio de la señal
				cost_mat[i, j] = dist_(x[i], y[j]) * (w_max * abs(i-j)/max(1., (i+j))+1.)
		
		# Matriz de costos acumulados
		acc_cost_mat = np.full((n_timestamps_1, n_timestamps_2), np.inf) 
		
		# Inicializar primera fila y columna
		acc_cost_mat[0, 0: region[1, 0]] = np.cumsum(cost_mat[0, 0: region[1, 0]])
		acc_cost_mat[0: region[1, 0], 0] = np.cumsum(cost_mat[0: region[1, 0], 0])
		
		# Ajustar región para evitar índices negativos
		region_ = np.copy(region)
		region_[0] = np.maximum(region_[0], 1)
		
		# Calcular matriz de costos acumulados con restricciones de pendiente
		for i in range(1, n_timestamps_1):
			for j in range(region_[0, i], region_[1, i]):
				# Implementación de restricción de pendiente como patrón de pasos
				# Corresponde a la implementación simétrica de Sakoe y Chiba (1978) con P = 0.5
				acc_cost_mat[i, j] = min(
					acc_cost_mat[i - 1, j-3] + 2*cost_mat[i, j-2] + cost_mat[i, j-1] + cost_mat[i, j],
					acc_cost_mat[i - 1, j-2] + 2*cost_mat[i, j-1] + cost_mat[i, j],
					acc_cost_mat[i - 1, j - 1] + 2*cost_mat[i, j],
					acc_cost_mat[i - 2, j-1] + 2*cost_mat[i-1, j] + cost_mat[i, j],
					acc_cost_mat[i - 3, j-1] + 2*cost_mat[i-2, j] + cost_mat[i-1, j] + cost_mat[i, j]
				)
		
		# Distancia final normalizada por la longitud del camino
		dtw_dist = acc_cost_mat[-1, -1]/(n_timestamps_1 + n_timestamps_2)
		part_dtw += math.sqrt(dtw_dist)
	
	# Resultado final ponderado con penalización por diferencia de longitud
	res = part_dtw / nLeads + small_c * (n_timestamps_1-n_timestamps_2)**2/min(n_timestamps_1,n_timestamps_2)
	
	return part_dtw, res
	

def dtwParallel_multi(simulatedSignal, referenceSignal,type_dtw=None, local_dissimilarity=None, MTS=None):
	#DTW multivariado espera que las entradas estén en la forma (tiempo, variables), no al revés.
	
	# Elimina la columna de tiempo si existe
	X_o = simulatedSignal[:, 1:]  # Derivaciones I-V6
	Y_o = referenceSignal[:, 1:]
	
	res=dtw_functions.dtw(X_o, Y_o, type_dtw="d", local_dissimilarity=d.euclidean, MTS=True)
	part_dtw=0 #esto no sirve para nada, solo para mantener la lógica que ya teengo montada
	return part_dtw, res


def dtwParallel_uni(simulatedSignal, referenceSignal,local_dissimilarity=None,constrained_path_search=None,get_visualization=None):
	res_part=0
	
	for i in range(1, 13):
		x = simulatedSignal[:, i]
		y = referenceSignal[:, i]
		

		dtw_distance=dtw_functions.dtw(
			x,
			y,
			local_dissimilarity=local_dissimilarity, 
			constrained_path_search=constrained_path_search,        
			get_visualization=constrained_path_search
		)
		res_part+=dtw_distance
	
	part_dtw=0
	res=np.mean(res_part)
	return part_dtw, res



def autoCC_lags(signal1, signal2, reference='signal2',write_corr_images=False, save_path=None):
	"""
	Parameters:
		signal1, signal2: np.ndarray de forma (derivaciones), se espera 12 derivaciones.
			# Cálculo de las derivaciones estándar
			 ECG[:, 0] =I 
			 ECG[:, 1] =II 
			 ECG[:, 2] =III
	
			# Cálculo de las derivaciones aumentadas
			ECG[:, 3] =  aVR 
			ECG[:, 4] = aVL 
			ECG[:, 5] = aVF 
	
			# Cálculo de las derivaciones precordiales
			ECG[:, 6] =  V1
			ECG[:, 7] =  V2
			ECG[:, 8] =  V3
			ECG[:, 9] = V4
			ECG[:, 10] = V5
			ECG[:, 11] = V6
			
		
		reference: 'signal1' o 'signal2'. Especifica cuál es la referencia fija.
		generation: int/str, generación para el título (opcional)
		idx: int/str, identificador para el título (opcional)
		save_path: str, carpeta donde guardar los plots

	Returns:
		aligned_signal1, aligned_signal2: señales alineadas
		cross_corrs: correlaciones por canal entre V1-V6
		mean_cross_corr: media de las correlaciones por canal entre V1-V6
	"""
	
   
	
	
	# Rellenar usando el último valor de la señal (padding "edge")
	if signal1.shape[0] < signal2.shape[0]:
		diff = signal2.shape[0] - signal1.shape[0]
		last_row = signal1[-1:, :]  # toma la última fila como base
		signal1 = np.vstack([signal1, np.tile(last_row, (diff, 1))])
	elif signal2.shape[0] < signal1.shape[0]:
		diff = signal1.shape[0] - signal2.shape[0]
		last_row = signal2[-1:, :]  # toma la última fila como base
		signal2 = np.vstack([signal2, np.tile(last_row, (diff, 1))])


	# Verificar que tienen la misma forma
	assert signal1.shape == signal2.shape, "Las señales deben tener la misma longitud"
	assert reference in ['signal1', 'signal2'], "Referencia debe ser 'signal1' o 'signal2'"

	if reference == 'signal1':
		sig_ref = signal1
		sig_mov = signal2
	else:
		sig_ref = signal2
		sig_mov = signal1
	
	
	num_leads = sig_ref.shape[1]
	aligned_moving = np.zeros_like(sig_mov)
	
	
	
	corr=[]
	lag=[]
	cross_corrs=[]
	
	
	aligned_signal1 = np.zeros_like(sig_ref)
	aligned_signal2 = np.zeros_like(sig_ref)

	for i in range(0, num_leads):
		corr.append(correlate(sig_ref[:, i], sig_mov[:, i], mode='full'))
		lag.append(np.argmax(corr[i]) - (len(sig_ref[:, i]) - 1))
		# Alinear aplicando desfase
		if lag[i] > 0:
			aligned = np.pad(sig_mov[:, i], (lag[i], 0), mode='edge')[:len(sig_mov[:, i])]
			
		else:
			aligned = np.pad(sig_mov[:, i], (0, -lag[i]), mode='edge')[-lag[i]:len(sig_mov[:, i])-lag[i]]
		
		
		
	
		aligned_moving[:, i] = aligned
	
		# Reasignar según cuál era la señal móvil
		if reference == 'signal1':
			aligned_signal1[:, i] = sig_ref[:, i]
			aligned_signal2[:, i] = aligned_moving[:, i]
		else:
			aligned_signal1[:, i] = aligned_moving[:, i]
			aligned_signal2[:, i] = sig_ref[:, i] 
		
		# Correlación por canal
		
		cross_corrs.append(pearsonr(aligned_signal1[:, i], aligned_signal2[:, i])[0])
	
		
	mean_cross_corr=np.mean(cross_corrs)
	mean_lags=np.mean(lag)
	std_lags=np.std(lag)
		
		
	   
	# --- PLOT de señales alineadas ---
	if write_corr_images:
		
		
		deriv_names = ['I','II','III','aVR','aVL','aVF','V1', 'V2', 'V3', 'V4', 'V5', 'V6']
		fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(15, 20))
		axes = axes.flatten()

		for i in range(0, num_leads):  
			#axes[i].plot(aligned_signal1[:, i ], label='Monodomain-simulated ECG', color='red', linestyle='-', linewidth=2.5, alpha=0.9)
			#axes[i].plot(aligned_signal2[:, i ], label='Clinical ECG', color='black', linestyle='--', linewidth=2.5, alpha=0.9)
			#axes[i].set_title(f'{deriv_names[i]} | Corr: {cross_corrs[i]:.2f}',fontsize=22)
			axes[i].plot(aligned_signal1[:, i ], label=f'{deriv_names[i]}', color='red', linestyle='-', linewidth=2.5, alpha=0.9)
			
					 
			axes[i].plot(aligned_signal2[:, i], label=f'CC = {cross_corrs[i]:.2f}', color='black', linestyle='--', linewidth=2.5, alpha=0.9)
			
			
			axes[i].legend(frameon=False,prop={"size":22},loc='upper left')
			# Mostrar el eje Y solo en la primera columna (i = 0, 2, 4)
			if i % 2 == 0:
				axes[i].set_ylabel('Voltage (mV)',fontsize=20)
				axes[i].tick_params(labelsize=20)
				axes[i].set_yticks([-3,-2, -1, 0, 1, 2,3])
			else:
				axes[i].tick_params(labelleft=False)
				
			if i > 9:
			# Eje X para todos
				axes[i].set_xlabel('Time (ms)',fontsize=20)
				axes[i].tick_params(labelsize=20)
			else:
				axes[i].tick_params(labelbottom=False)
				
			axes[i].set_ylim(-3, 3)
	
	
	
		full_title=f"Mean autocorrelation: {mean_cross_corr:.2f} // Mean Lats: {mean_lags:.2f} // Std Lats: {std_lags:.2f}"
			
		   
		plt.suptitle(f' ECG({full_title})', fontsize=26)
		plt.tight_layout(rect=[0, 0, 1, 0.96])
		
		
		# Guardar el plot
		
		
		plt.savefig(save_path)
		
		plt.close()
	
	part_dtw=0 #esto no sirve para nada, solo para mantener el resto de codigo
	
	if std_lags < 1e-6:
		std_lags = 1e-6
	
	res = (1-mean_cross_corr) + std_lags #esto tengo que pensarlo bien pero lo pongo así porque luego se convierte en - para que el ga lo máximice
	
	return part_dtw, res, std_lags
	
	
	
	
	
	
def dtwCamps_stdLags(signal1, signal2,derivaciones, meshVolume, reference='signal2', write_corr_images=False, save_path=None):
	
	part_dtw, res_dtw = dtw_camps(signal1, signal2, derivaciones, meshVolume)
	
	part_error, error, std_lags = autoCC_lags(signal1[:,1:], signal2[:,1:], reference='signal2',write_corr_images=write_corr_images, save_path=save_path)
		
	res_final= res_dtw + std_lags    #esto tengo que pensarlo bien pero lo pongo así porque luego se convierte en - para que el ga lo máximice
	
	return 0, res_final
	
	
	
   
   

def encontrar_max_cuadrado(y):
    """Encuentra el índice del máximo valor cuadrado de la señal."""
   
    return  np.argmax((y - y[0])**2)



def desplazar_senal_padding_constante(y, delta_idx):
    """
    Desplaza la señal y en el tiempo para alinear máximos.
    Rellena con el valor constante del borde (inicio o final) en lugar de ceros.
    
    Parámetros:
    y -- numpy array, señal 1D
    delta_idx -- entero, número de muestras a desplazar (+ hacia la derecha)
    
    Retorna:
    y_desplazada -- señal desplazada con padding constante, misma longitud que y
    """
    N = len(y)
    y_desplazada = np.empty_like(y)

    if delta_idx > 0:
        # Desplazar a la derecha: rellenar inicio con valor inicial
        y_desplazada[:delta_idx] = y[0]
        y_desplazada[delta_idx:] = y[:N - delta_idx]
    elif delta_idx < 0:
        # Desplazar a la izquierda: rellenar final con valor final
        y_desplazada[:N + delta_idx] = y[-delta_idx:]
        y_desplazada[N + delta_idx:] = y[-1]
    else:
        y_desplazada = y.copy()

    return y_desplazada





def calcular_escala_offset(y_m, y_s):
    """Calcula escala y offset minimizando error L2."""
    T = len(y_m)

    A = np.sum(y_s ** 2)
    B = np.sum(y_s)
    C = np.sum(y_m * y_s)
    D = np.sum(y_m)

    M = np.array([[A, B],
                  [B, T]])
    V = np.array([C, D])

    s, r = np.linalg.solve(M, V)

    return s, r
    







    
 
def evaluar_tuplas_globalmente(y_m, y_s, deltas, escalas, offsets):
    """
    Evalúa cada triplete (delta, escala, offset) aplicado a todas las derivaciones simuladas,
    calcula la suma de normas L2 de residuos y devuelve la tupla que minimiza esa suma.
    
    Parámetros:
    y_m -- matriz (T,L) señal medida
    y_s -- matriz (T,L) señal simulada original (sin alinear ni ajustar)
    deltas -- array (L,) de desplazamientos por derivación
    escalas -- array (L,) de escalas por derivación
    offsets -- array (L,) de offsets por derivación
    
    Retorna:
    mejor_tupla -- (delta, escala, offset) que minimiza la suma de normas L2
    min_error -- valor mínimo de la suma de normas L2
    """
    T, L = y_m.shape
    min_error = np.inf
    mejor_tupla = None

    for i in range(L):
        delta = deltas[i]
        escala = escalas[i]
        offset = offsets[i]

        # Desplazamos todas las señales simuladas con el delta i
        y_s_desplazada = np.zeros_like(y_s)
        for j in range(L):
            y_s_desplazada[:, j] = desplazar_senal_padding_constante(y_s[:, j], delta)
        
        # Aplicamos escala y offset i a todas las derivaciones
        y_s_ajustada = escala * y_s_desplazada + offset

        # Calculamos el sumatorio de la norma L2 de los residuos
        error = 0
        for j in range(L):
            residuo = y_m[:, j] - y_s_ajustada[:, j]
            error += np.linalg.norm(residuo, 2)

        if error < min_error:
            min_error = error
            mejor_tupla = (delta, escala, offset)

    return mejor_tupla, min_error




    
  
def L2_norm(ecg_path,target_signal_path):

    '''
    # Simulamos señales multicanal: (T, L)
    T = 200
    L = 3  # número de derivaciones
    t = np.linspace(0, 1, T)

    # Señal medida: 3 pulsos gaussianos en diferentes posiciones
    y_m = np.zeros((T, L))
    centros = [0.3, 0.5, 0.7]
    for i in range(L):
        y_m[:, i] = np.exp(-((t - centros[i]) ** 2) / (2 * 0.03 ** 2))

    # Señal simulada: desplazada, escalada y con offset distinto para cada derivación
    desplazamientos_muestras = [15, 20, 10]  # diferentes desplazamientos
    escalas_reales = [0.7, 0.8, 0.6]
    offsets_reales = [0.1, 0.05, 0.08]
    y_s = np.zeros_like(y_m)
    for i in range(L):
        y_s[:, i] = np.exp(-((t - centros[i] - desplazamientos_muestras[i] * (t[1]-t[0])) ** 2) / (2 * 0.03 ** 2))
        y_s[:, i] = escalas_reales[i] * y_s[:, i] + offsets_reales[i]
    '''
    
    # Cargar datos
    y_m_raw = np.loadtxt(target_signal_path, delimiter=',')
    y_s_raw = ecg_calcul(ecg_path)

    
    
    # Eliminar primera columna (tiempo)
    y_m = y_m_raw[:, 1:]
    y_s = y_s_raw[:, 1:]

    
    # Igualar longitudes con padding constante (último valor)
    long_max = max(y_m.shape[0], y_s.shape[0])

    if y_m.shape[0] < long_max:
        faltan = long_max - y_m.shape[0]
        pad_values = np.tile(y_m[-1, :], (faltan, 1))
        y_m = np.vstack([y_m, pad_values])

    if y_s.shape[0] < long_max:
        faltan = long_max - y_s.shape[0]
        pad_values = np.tile(y_s[-1, :], (faltan, 1))
        y_s = np.vstack([y_s, pad_values])

    
    L=y_m.shape[1] #numero derivaciones
    
    
    # Calculamos desplazamientos para cada derivación
    deltas = []
    for i in range(L):
        idx_max_m = encontrar_max_cuadrado(y_m[:, i])
        idx_max_s = encontrar_max_cuadrado(y_s[:, i])
        delta = idx_max_m - idx_max_s
        deltas.append(delta)
    deltas = np.array(deltas)

    # Desplazamos cada derivación simulada por separado
    y_s_alineada = np.zeros_like(y_s)
    for i in range(L):
        y_s_alineada[:, i] = desplazar_senal_padding_constante(y_s[:, i], deltas[i])

    # Calculamos escala y offset para cada derivación por separado
    escalas = []
    offsets = []
    y_s_ajustada = np.zeros_like(y_s)
    for i in range(L):
        s, r = calcular_escala_offset(y_m[:, i], y_s_alineada[:, i])
        escalas.append(s)
        offsets.append(r)
        y_s_ajustada[:, i] = s * y_s_alineada[:, i] + r #SANDRA
    escalas = np.array(escalas)
    offsets = np.array(offsets)

    '''
    # --- Gráficos ---
    fig, axs = plt.subplots(L, 4, figsize=(20, 4*L))
    for i in range(L):
        # Señales originales y máximos
        axs[i, 0].plot(y_m[:, i], label='Medida', marker='o')
        axs[i, 0].plot(y_s[:, i], label='Simulada', marker='x')
        idx_max_m = encontrar_max_cuadrado(y_m[:, i])
        idx_max_s = encontrar_max_cuadrado(y_s[:, i])
        axs[i, 0].plot(idx_max_m, y_m[idx_max_m, i], 'ro', label='Máximo y_m')
        axs[i, 0].plot(idx_max_s, y_s[idx_max_s, i], 'rx', label='Máximo y_s')
        axs[i, 0].set_title(f'Derivación {i+1} - Originales con máximos')
        axs[i, 0].legend()
        axs[i, 0].grid()

        # Señales alineadas temporalmente
        axs[i, 1].plot(y_m[:, i], label='Medida', marker='o')
        axs[i, 1].plot(y_s_alineada[:, i], label=f'Simulada alineada (delta={deltas[i]})', marker='x')
        axs[i, 1].set_title(f'Derivación {i+1} - Señales alineadas')
        axs[i, 1].legend()
        axs[i, 1].grid()

        # Señales después de escala y offset
        axs[i, 2].plot(y_m[:, i], label='Medida', marker='o')
        axs[i, 2].plot(y_s_ajustada[:, i], label=f'Simulada ajustada (s={escalas[i]:.2f}, r={offsets[i]:.2f})', marker='x')
        axs[i, 2].set_title(f'Derivación {i+1} - Ajuste escala & offset')
        axs[i, 2].legend()
        axs[i, 2].grid()

        # Residuo (error)
        residuo = y_m[:, i] - y_s_ajustada[:, i]
        axs[i, 3].plot(residuo, label='Residuo')
        axs[i, 3].set_title(f'Derivación {i+1} - Residuo')
        axs[i, 3].legend()
        axs[i, 3].grid()

    plt.tight_layout()
    plt.show()

    '''
    # --- Uso después de haber calculado deltas, escalas, offsets ---

    mejor_tupla, error_min = evaluar_tuplas_globalmente(y_m, y_s, deltas, escalas, offsets)
    print(f"Mejor tupla global (delta, escala, offset): {mejor_tupla}")
    print(f"Error mínimo sumado: {error_min:.4f}")
    
    
    # Ahora, aplicar la tupla ganadora (delta, escala, offset) a todas las derivaciones
    delta_opt, escala_opt, offset_opt = mejor_tupla

    ECG_original_ajustado = np.zeros_like(y_s)
    for i in range(L):
        y_s_desplazada = desplazar_senal_padding_constante(y_s[:, i], delta_opt)
        ECG_original_ajustado[:, i] = escala_opt * y_s_desplazada + offset_opt
        
        
    # Crear vector de tiempo desde 0 hasta número de filas - 1
    tiempo = np.arange(ECG_original_ajustado.shape[0]).reshape(-1, 1)

    # Añadir esta columna como primera columna
    ECG_original_ajustado_con_tiempo = np.hstack((tiempo, ECG_original_ajustado))
    
    return mejor_tupla,error_min, ECG_original_ajustado_con_tiempo   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   

def fitness_func(ga_instance, solution, solution_idx):
	"""
	Función de fitness para el algoritmo genético.
	Evalúa una solución específica modificando los nodos de estimulación,
	ejecutando la simulación y calculando el error respecto a la señal objetivo.
	
	Parámetros:
	----------
	ga_instance : pygad.GA
		Instancia del algoritmo genético
	solution : array
		Solución a evaluar (selección de nodos)
	solution_idx : int
		Índice de la solución en la población
		
	Returns:
	-------
	float
		Valor de fitness (negativo del error, para maximizar)
	"""
	
		
	# Convertir solución a enteros
	

	selected_values = np.array(solution, dtype=int)
	
		   
		
		
		
	modules_path_openCarp = ga_instance.modules_path_openCarp
	geom = ga_instance.geom
	openCarp_simulationFiles = ga_instance.openCarp_simulationFiles
	lead_file = ga_instance.lead_file
	torso_file_um = ga_instance.torso_file_um
	torso_file = ga_instance.torso_file
	config_template = ga_instance.config_template
	
	optim_type = ga_instance.optim_type  
	current_dir = ga_instance.current_dir
	derivation_num = ga_instance.derivation_num
	meshVolume = ga_instance.meshVolume
	optimization_function = ga_instance.optimization_function
	nodes_IDs = ga_instance.nodes_IDs
	normalized_target_signal= ga_instance.normalized_target_signal
	
	sol_per_pop = ga_instance.sol_per_pop
	current_generation = ga_instance.generations_completed
	
	absolute_index = current_generation * sol_per_pop + solution_idx  # Índice absoluto
	
	
	# Crear directorio para esta evaluación
	eval_dir = os.path.join(current_dir, f"gen_{current_generation}_sol_{solution_idx}")
	os.makedirs(eval_dir, exist_ok=True)
	
  
	
	manual_ids = selected_values#",".join(str(x) for x in selected_values)
	manual_times = np.zeros_like(selected_values, dtype=float)#",".join(["0"] * len(selected_values))
	
	
	print(f'#####################################################################')
	print(f'#######   points: {manual_ids}')
	print(f'#######   times: {manual_times}')
	print(f'#####################################################################')
	
	
	args_dict = {
		"job_name": eval_dir,
		"geom": ga_instance.geom,
		"simulation_files": ga_instance.openCarp_simulationFiles,
		"duration": 120.,
		"model": "MitchellSchaeffer",
		"myocardial_CV": 570,
		"initial_time": 40,
		"sinusal_mode": "manual",
		"manual_ids": manual_ids,
		"manual_times": manual_times,
		"opt_file": "",
		"kdtree_file": "",
		"save_vtk": False,
		"output_vtk_name": "",
		"lead_file": ga_instance.lead_file,
		"torso_file_um": ga_instance.torso_file_um,
		"torso_file": ga_instance.torso_file,
		"config_template": ga_instance.config_template,
	}
	
	
	def parser():
		parser = tools.standard_parser()
		group  = parser.add_argument_group('experiment specific options')
		group.add_argument('--job-name',
							type=str,
							default=args_dict["job_name"],
							help='Nombre del directorio de salida para los resultados')


		group.add_argument('--geom',
							type = str,
							default = args_dict["geom"],
							help = 'Path and name of the ventricular geometry')
		
		
		group.add_argument('--simulation-files',
							type = str,
							default = args_dict["simulation_files"],
							help = 'Path and name of the simulation mesh files (without extension)') 
		group.add_argument('--duration',
							type = float,
							default = args_dict["duration"],
							help = 'Duration of simulation in [ms] (default: 300.)')
		group.add_argument('--model',
					   type=str,
					   choices=['OHara', 'MitchellSchaeffer'],
					   default='MitchellSchaeffer',
					   help='Electrophysiological model to use (default: MitchellSchaeffer)')
		group.add_argument('--myocardial-CV',
					   type=float,
					   default=args_dict["myocardial_CV"],
					   help='Velocidad de conducción para el miocardio')  
		group.add_argument('--initial-time',
					   type=float,
					   default=args_dict["initial_time"],
					   help='Tiempo del primer estímulo')  
					   
		group.add_argument('--sinusal-mode',
					   type=str,
					   choices=['manual', 'optimizado'],
					   default=args_dict["sinusal_mode"],
					   help='Origen de la estimulación sinusal (manual u optimizado)')
		
		
		group.add_argument('--manual-ids',
					   type=str,
					   default=manual_ids,
					   help='Lista de IDs para estimulación manual, separados por comas (ej: "123,456,789")')
		group.add_argument('--manual-times',
					   type=str,
					   default=manual_times,
					   help='Lista de tiempos correspondientes, separados por comas (ej: "10,20,30")')
		
		group.add_argument('--opt-file',
					   type=str,
					   default='',
					   help='Ruta al archivo .npz con resultados de optimización sinusal')
		group.add_argument('--kdtree-file',
					   type=str,
					   default='',
					   help='Ruta al archivo .pkl del KDTree (solo en modo optimizado)')
		
		group.add_argument('--save-vtk',
					   type=str,
					   default=args_dict["save_vtk"],
					   help='Guardar los puntos de estimulación y los tiempos como un .vtk (True/False)')            
		group.add_argument('--output-vtk-name',
						type=str,
						default="",
						help='Nombre del archivo de salida .vtu con estimulación sinusal')
		
		group.add_argument('--lead-file',
							type = str,
							default = args_dict["lead_file"],
							help = 'Path and name of labeled .vtk lead file [V1,V2,V3,V4,V5,V6,LA,RA,LL]') 
		group.add_argument('--torso-file-um',
							type = str,
							default = args_dict["torso_file_um"],
							help = 'Path and name of the .vtk torso mesh file (um)') 
		
		group.add_argument('--torso-file',
							type = str,
							default = args_dict["torso_file"],
							help = 'Path and name of the .vtk torso mesh file for config file')                     
		
		group.add_argument('--config-template',
							type=str,
							default=args_dict["config_template"],
							help='Ruta al archivo config.json base para ECG')

		return parser
	
	
	
	def jobID(args):
		if args.job_name:
			return args.job_name
		else:
			now = datetime.now()
			timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
			return f'{timestamp}_basic_{args.duration}_{args.model}'
			
	
	
	
		
	@tools.carpexample(parser, jobID)    
	def simulacion(args, job):
		
		filenameorignal=job.ID          
		
		# Geometry reading
		vol_mesh = smart_reader(args.geom)
		vol_mesh = addGlobalIds(vol_mesh)
		
		#save GlobalIds complete mesh
		#vtkWrite(vol_mesh,'./meshes/vol_mesh_globaIds.vtu')
		
		
		
		#####################################################
		## SINUSAL
		#####################################################
		
		if args.sinusal_mode == "manual":
			
		 
			###################
			## MANUAL MODE
			###################
					
			if args.manual_ids is None or args.manual_times is None:
				raise ValueError("En modo manual debes proporcionar 'manual-ids' y 'manual-times'")

			try:
				ids_sinusal_manual = args.manual_ids#[int(x) for x in args["manual_ids"].split(',')]
				tiempos_sinusal_manual = args.manual_times#[float(x) for x in args["manual_times"].split(',')]
			except Exception as e:
				raise ValueError(f"Error al parsear '--manual-ids' o '--manual-times': {e}")

			if len(ids_sinusal_manual) != len(tiempos_sinusal_manual):
				raise ValueError("El número de IDs debe coincidir con el número de tiempos")

		
			global_ids_selected = np.array(ids_sinusal_manual)
			original_stimTimes_sinusal = np.array(tiempos_sinusal_manual)
			
			original_stimTimes_sinusal2 = [x - min(original_stimTimes_sinusal)+ args.initial_time for x in original_stimTimes_sinusal]  #start de sinusal stimulation at 0ms
			

		else:
		
		
			#########################
			## FROM OPTIMIZATION MODE
			#########################
			
			#load optimization result and precomputed kdtree
			opt_path = args.opt_file
			
			if not os.path.exists(args.opt_file):
					raise FileNotFoundError(f"Archivo de optimización no encontrado: {args['opt_file']}")


			best_params_reshape = np.load(opt_path)["best_solution_6d.npy"] # ab tm rt tv ts time   
			
			
			if not os.path.exists(args.kdtree_file):
				# Crear arbol de ucoords
				ids = numpy_support.vtk_to_numpy(vol_mesh.GetPointData().GetArray('GlobalIds'))
				ucoords = np.column_stack(get_cobivecoaux(vol_mesh, ids))
				tree = cKDTree(ucoords)
					   
				# Guardar KDTree
				with open(args.kdtree_file, 'wb') as f:
					pickle.dump(tree, f)
		
				print(f"KDTree guardado exitosamente en {args.kdtree_file}")
					
			else:                
				with open(args.kdtree_file, 'rb') as f:
					tree = pickle.load(f)

		 
			_, global_ids_selected = tree.query(best_params_reshape[:, :5], k=1, workers=-1) #find the globalIds from uvc
			
			
		   
			original_stimTimes_sinusal=best_params_reshape[:,-1] #take optimized stimulation time
			original_stimTimes_sinusal2 = [x - min(original_stimTimes_sinusal) for x in original_stimTimes_sinusal] #start sinusal stimulation at 0ms
			
			
		
		
		
		######### SINUSAL STIMULATION
	 
		num_stim_sinusal, stims_list_sinusal = stimBlock(global_ids_selected, original_stimTimes_sinusal2, job.ID, num_stim=0, filename="node_sinusal")
		
		
		#save stimulation times for sinusal
		stim_txt_path = os.path.join(job.ID, "stimulated_ids_and_times_sinusal.txt")
		with open(stim_txt_path, "w") as f:
			f.write("GlobalID\tTime[ms]\tOriginalTime[ms]\n")
			for gid, t, t_or in zip(global_ids_selected, original_stimTimes_sinusal2,original_stimTimes_sinusal):
				f.write(f"{gid}\t{t:.4f}\t{t_or:.4f}\n")
		 
		
		
		
		
		#save as .vtk file
		if str(args.save_vtk).lower() == "true":
				#save stimulus input for points and times
				vtk_output_path = os.path.join(job.ID, args.output_vtk_name)

				#save GlobalIds complete mesh and endo to check
				#surf_mesh = extract_surface(vol_mesh)
				#endo = threshold(surf_mesh, 3, 4, 'interp_class')
				#vtkWrite(vol_mesh,'./vol_mesh_globaIds.vtu')         
				#vtkWrite(endo,'./endo_globaIds.vtu')
			
				if args.sinusal_mode == "manual":
					create_vtk_from_nodes(vol_mesh=vol_mesh, global_ids=global_ids_selected, uvc_coords=None, stim_times=original_stimTimes_sinusal2, output_filename=vtk_output_path)
				else:
					create_vtk_from_nodes(vol_mesh=vol_mesh, global_ids=global_ids_selected, uvc_coords=best_params_reshape[:, :5], stim_times=original_stimTimes_sinusal2, output_filename=vtk_output_path)
			 
			 
			
		#########################
		## EIKONAL SIMULATION
		#########################

		num_stim_total=num_stim_sinusal
		stims_list_total = stims_list_sinusal #globalIds list for stimulus points
		
		simulation_eik(job, args, num_stim_total, stims_list_total)
		
		
		cmd = [settings.execs.igbextract,
				   '-o', 'asciiTm',
				   '-O', '{}/vm.dat'.format(job.ID),
				   os.path.join(job.ID, 'vm.igb')]
		job.bash(cmd)



		############################
		## LEADFIELD ECG COMPUTATION
		############################
		leads = smart_reader(args.lead_file)
		torso_mesh = smart_reader(args.torso_file_um)

		lead_ids = get_closest_global_ids_sorted_by_label(leads, torso_mesh) 
		
		config_template_path = args.config_template

		unique_config_path = os.path.join(filenameorignal, 'config.json')
		shutil.copy(config_template_path, unique_config_path)
		
		
		modify_json_entry( file_path=unique_config_path, key="VTK_FILE", new_value=args.torso_file )
		modify_json_entry( file_path=unique_config_path, key="MEASUREMENT_LEADS", new_value=lead_ids )
		modify_json_entry(file_path=unique_config_path, key="VM_FILE", new_value=os.path.join(job.ID, "vm.dat"))
		modify_json_entry(file_path=unique_config_path, key="ECG_FILE", new_value=os.path.join(job.ID, "ecg_output.dat"))



		C_matrix_data, heart_node_indices, num_total_nodes, _ = compute_leadfield(unique_config_path)
		compute_ecg(C_matrix_data, heart_node_indices, num_total_nodes, unique_config_path)
		
		
		
		print(f"Output folder: {job.ID}")
	  
		return os.path.join(job.ID, "ecg_output.dat")
	
	
	try:
		# Ejecutar la simulación
		argv = []
		ecg_path = simulacion(argv)
		
		# Verificar si el archivo ECG existe
		if not os.path.exists(ecg_path):
			raise FileNotFoundError(f"No se encontró el archivo ECG: {ecg_path}")


		ECG_normalized = ecg_calcul_normalized(ecg_path)
		ECG_original = ecg_calcul(ecg_path)
		leads_data= np.loadtxt(ecg_path)
		

		if optimization_function=="dtw_camps":
			# Calcular error usando DTW (Dynamic Time Warping)
			part_error, error = dtw_camps(ECG_normalized, normalized_target_signal, derivation_num,meshVolume)
		
		#sandra revisa esto para ver si esa bien la forma en la que se devuelven los rrores y se guardan en el pkl
		
		if optimization_function=="dtwParallel_multi":
			part_error, error = dtwParallel_multi(ECG_normalized, normalized_target_signal, type_dtw="d", local_dissimilarity=d.euclidean, MTS=True)
		
		if optimization_function=="dtwParallel_uni":
			part_error, error = dtwParallel_uni(ECG_normalized, normalized_target_signal, local_dissimilarity=d.euclidean, constrained_path_search="itakura", get_visualization=False)
		
		if optimization_function=="autoCC_lags":
			part_error, error, std_lags = autoCC_lags(ECG_normalized[:,1:], normalized_target_signal[:,1:], reference='signal2',write_corr_images=True, save_path=os.path.join('/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/PRUEBAS/nodos/seed_42_realPSM4',f'gen_{current_generation}_sol_{solution_idx}.png'))
		
		if optimization_function=="autoCC_lags":
			part_error, error = dtwCamps_stdLags(ECG_normalized, normalized_target_signal, derivation_num, meshVolume,reference='signal2', write_corr_images=False, save_path=None)
		
		
		
		
		# Convertir arrays a listas para serialización
		ecg_norm_list = ECG_normalized.tolist() 
		ecg_orig_list = ECG_original.tolist() 
		
	   
		
		# Bloqueamos para evitar condiciones de carrera
		with ga_instance.optimization_lock:

			# Guardar en el diccionario (usa current_generation y solution_idx como claves anidadas)
			if optim_type == "nodes":
			   
				ga_instance.simulation_results[current_generation][solution_idx]= {
					"solution_nodes": list(selected_values),
					"error": error,
					"part_error": part_error,
					"simulated_signal_normalized": ecg_norm_list,
					"simulated_signal_original": ecg_orig_list,
					"leads_data":leads_data
				}   

			if optim_type=="sequence":

			   
				ga_instance.simulation_results[current_generation][solution_idx]={
					"nodes_IDs":list(nodes_IDs),
					"solution_nodes": list(selected_values),
					"error": error,
					"part_error": part_error,
					"simulated_signal_normalized": ecg_norm_list,  # Matriz de la señal
					"simulated_signal_original": ecg_orig_list,  # Matriz de la señal
					"leads_data":leads_data
				}
				
			
			
			# Guardar en la posición preasignada
			ga_instance.optimization_state['all_simulated_signals_normalized'][absolute_index] = ecg_norm_list
			ga_instance.optimization_state['all_simulated_signals_original'][absolute_index] = ecg_orig_list
			ga_instance.optimization_state['all_errors'][absolute_index] = error
			
		
		
			# Actualizar el mejor resultado si corresponde
			 
			if error < ga_instance.optimization_state['best_error']:
				
				ga_instance.optimization_state['best_error'] = error
				ga_instance.optimization_state['ecg_best'] = ecg_norm_list
				ga_instance.optimization_state['best_absolute_index']=absolute_index
				print(f"Nuevo mejor error: {error}")
				sys.stdout.flush()  # Asegura que se imprima inmediatamente
			
			#guardar datos del estado de la simulación
			save_manager_state(ga_instance.optimization_state,ga_instance.simulation_results,"optimization_state.pkl","simulation_results.pkl")
			
			
			# Registrar resultados en archivo de log
			log_file = "registro_resultados.txt"
			file_exists = os.path.isfile(log_file)

			with open(log_file, "a") as f:
				if not file_exists:
					f.write("generation,solution_index,selected_values,error,part_error\n")
				f.write(f"{current_generation},{solution_idx},{selected_values.tolist()},{error},{part_error}\n")
				
				
		   
			
		# Generar gráficas comparativas
		# 1. Gráficas (usamos copias locales de los datos)
		best_error = ga_instance.optimization_state['best_error']
		best_signal = list(ga_instance.optimization_state['ecg_best']) if ga_instance.optimization_state['ecg_best'] else None
		# Convertir Manager.list a lista normal y filtrar valores None
		all_signals = [x for x in list(ga_instance.optimization_state['all_simulated_signals_normalized']) if x is not None]
		
		if best_signal is not None and len(all_signals) > 0:
			graficar_best(all_signals, best_signal, normalized_target_signal, best_error)
		else:
			print("Advertencia: No hay suficientes datos para graficar")
			sys.stdout.flush()
			
		
		# Limpiar archivos temporales
		try:
			shutil.rmtree(eval_dir)  # Elimina recursivamente la carpeta y su contenido
			print(f"✓ Directorio {eval_dir} eliminado")
			sys.stdout.flush()
		except Exception as e:
			print(f"Error al eliminar {eval_dir}: {str(e)}")
			
			
			
		#mutación adaptativa para que la reduzca a partir de la mitad    
		if ga_instance.generations_completed == ga_instance.num_generations // 2:
			ga_instance.mutation_probability = 0.08
			print(f"\nReduciendo mutación a 8% en generación {ga_instance.generations_completed}")
			sys.stdout.flush()    
		return -error
	
	except Exception as e:
		print(f"Error en la simulación: {str(e)}")
		print(f"[ERROR] Error al evaluar solución idx={solution_idx}, generación={current_generation}: {e}")
		traceback.print_exc()

		# Un valor de error muy alto
		fake_error = 999999

		# Bloqueamos para evitar condiciones de carrera
		with ga_instance.optimization_lock:

			# Guardar en el diccionario (usa current_generation y solution_idx como claves anidadas)
			if optim_type == "nodes":
			   
				ga_instance.simulation_results[current_generation][solution_idx]= {
					"solution_nodes": list(selected_values),
					"error": error,
					"part_error": part_error,
					"simulated_signal_normalized": ecg_norm_list,
					"simulated_signal_original": ecg_orig_list,
					"leads_data":leads_data
				}   

			if optim_type=="sequence":

			   
				ga_instance.simulation_results[current_generation][solution_idx]={
					"nodes_IDs":list(nodes_IDs),
					"solution_nodes": list(selected_values),
					"error": error,
					"part_error": part_error,
					"simulated_signal_normalized": ecg_norm_list,  # Matriz de la señal
					"simulated_signal_original": ecg_orig_list,  # Matriz de la señal
					"leads_data":leads_data
				}
				
			
			# Guardar en la posición preasignada
			ga_instance.optimization_state['all_simulated_signals_normalized'][absolute_index] = ecg_norm_list
			ga_instance.optimization_state['all_simulated_signals_original'][absolute_index] = ecg_orig_list
			ga_instance.optimization_state['all_errors'][absolute_index] = error
	
			#guardar datos del estado de la simulación
			save_manager_state(ga_instance.optimization_state,ga_instance.simulation_results,"optimization_state.pkl","simulation_results.pkl")
			
			# También puedes registrar en un log de errores si quieres
			with open("errores_simulacion.log", "a") as log_file:
				log_file.write(f"Error en gen={current_generation}, sol={solution_idx}:\n")
				log_file.write(traceback.format_exc())
				log_file.write("\n" + "="*80 + "\n")
		
	
		return -fake_error
		
		
		
	
def custom_mutation(offspring, ga_instance):
	"""
	Función de mutación personalizada para el algoritmo genético.
	Modifica 2 o 3 genes de cada descendiente, asegurando que no haya repeticiones.
	- Reproducible: Misma semilla → mismos resultados.
	- Única por individuo: Cada descendiente tiene mutaciones distintas.
	"""
	nodes_IDs = ga_instance.nodes_IDs
	optim_type = ga_instance.optim_type
	num_genes = ga_instance.num_genes

	if optim_type == "sequence":
		set_values = ga_instance.set_values

	base_seed = ga_instance.random_seed
	current_gen = ga_instance.generations_completed
	rng = np.random.RandomState(base_seed)

	if optim_type == "nodes":
		values = nodes_IDs
	elif optim_type == "sequence":
		values = set_values
	else:
		raise ValueError("optim_type debe ser 'nodes' o 'sequence'")

	values_set = set(values)

	for i in range(offspring.shape[0]):
		individual_seed = base_seed + current_gen * 1000 + i
		rng_indv = np.random.RandomState(individual_seed)

		num_mutations = rng_indv.randint(2, 4)
		mutation_indices = rng_indv.choice(num_genes, num_mutations, replace=False)

		# Copia del individuo actualizada con las mutaciones para evitar duplicados en tiempo real
		individual_copy = offspring[i].copy()

		for idx in mutation_indices:
			new_value = rng_indv.choice(values)

			# Validar que sea un valor permitido
			while new_value not in values_set:
				print(f"[ADVERTENCIA] Se generó un valor inválido: {new_value}. Seleccionando otro.")
				sys.stdout.flush()
				new_value = rng_indv.choice(values)

			# Evitar duplicados en el individuo actualizado
			while new_value in individual_copy:
				new_value = rng_indv.choice(values)
				while new_value not in values_set:
					print(f"[ADVERTENCIA] Se generó un valor inválido al evitar duplicados: {new_value}. Seleccionando otro.")
					sys.stdout.flush()
					new_value = rng_indv.choice(values)

			# Asignar mutación y actualizar la copia
			offspring[i, idx] = new_value
			individual_copy[idx] = new_value  # importante para evitar duplicados posteriores

	return offspring




	
def setup_initial_population_times(params, nodes_IDs):
	"""
	Configura la población inicial y los valores posibles para mutación.
	
	Returns:
	- Si optim_type == "nodes": Retorna solo la población (np.ndarray).
	- Si optim_type == "sequence": Retorna (población, set_values) (tuple).
	"""
	np.random.seed(params['random_seed'])
	
	'''
	if params['optim_type'] == "nodes":
		population = np.array([
			np.random.choice(nodes_IDs, size=params['num_genes'], replace=False)
			for _ in range(params['sol_per_pop'])
		])
		return population  # Solo población (shape: sol_per_pop x num_genes)
	'''
	if params['optim_type'] == "sequence":
		set_values = np.arange(1, params['last_seq'] + 1)  # Ej: 1..21
		population = np.array([
			np.random.choice(set_values, size=params['num_genes'], replace=True)
			for _ in range(params['sol_per_pop'])
		])
		return population, set_values  # Tupla con población y valores posibles
					  


def setup_initial_population_precomputed_points(params):
	total_population=np.load(params['population_path'],allow_pickle=True)
	
	
	np.random.seed(params['random_seed'])
   
	indices = np.random.choice(len(total_population),params['sol_per_pop'], replace=False)
	initial_population = total_population[indices]

	return initial_population.tolist()





def configure_genetic_algorithm(params, initial_population):
	"""Configura y retorna una instancia del algoritmo genético."""
	manager = Manager()
	# Calcular el tamaño máximo previsible
	max_size = params['num_generations'] * params['sol_per_pop'] + params['sol_per_pop']
	
	
	# Crear estructuras de datos compartidas
	shared_state = manager.dict()
	shared_state2 = manager.dict()
	
	
	# Inicializar simulation_results por adelantado
	for gen in range(params['num_generations']):
		gen_dict = manager.dict()  # Diccionario para esta generación
		for idx in range(params['sol_per_pop']):  # Corregido: params['sol_per_pop']
			gen_dict[idx] = None  # Inicializar cada solución como None
		shared_state2[gen] = gen_dict
		
		
	# Inicializar listas con tamaño fijo y valores por defecto
	shared_state['all_simulated_signals_normalized'] = manager.list([None] * max_size)
	shared_state['all_simulated_signals_original'] = manager.list([None] * max_size)
	shared_state['all_errors'] = manager.list([float('inf')] * max_size)  # Usar 'inf' como valor por defecto para errores
	
	shared_state['best_error'] = float('inf')
	shared_state['ecg_best'] = None
	shared_state['best_absolute_index'] = -1
	
	
	
	# Crear instancia de GA
	ga_instance = pygad.GA(
		num_generations=params['num_generations'],
		num_parents_mating=params['num_parents_mating'],
		sol_per_pop=params['sol_per_pop'],
		num_genes=params['num_genes'],
		initial_population=initial_population,
		parent_selection_type=params['parent_selection_type'],
		keep_elitism=params['keep_elitism'],
		crossover_type=params['crossover_type'],
		mutation_type=custom_mutation,
		mutation_probability=params['mutation_probability'],
		stop_criteria = params['stop_criteria'],
		random_seed=params['random_seed'],
		fitness_func=fitness_func,
		parallel_processing=['process', params['num_cpu']],
		allow_duplicate_genes=False,
		gene_space= params['nodes_IDs'],
		suppress_warnings=True
	)
	
	# Asignar estructuras compartidas
	ga_instance.__dict__['optimization_lock'] = manager.Lock()
	ga_instance.__dict__['optimization_state'] = shared_state
	ga_instance.__dict__['simulation_results'] = shared_state2 
	
	# Asignar otros parámetros necesarios
	for key, value in params.items():
		if not hasattr(ga_instance, key):
			ga_instance.__dict__[key] = value
	
	return ga_instance