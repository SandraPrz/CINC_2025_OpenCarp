# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 19:23:18 2025

@author: Sandra
"""

import numpy as np
from scipy.signal import correlate
from scipy.stats import pearsonr
import json
import base64
from io import BytesIO
import time
import csv
import pickle
import sys
from multiprocessing import Manager
import matplotlib.pyplot as plt
import time
import numpy as np
import copy
import os
from multiprocessing.managers import DictProxy

import vtk
from vtk.util.numpy_support import vtk_to_numpy



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

def cargar_puntos_vtk(vtk_file):
    """Carga un archivo .vtk y extrae los puntos, soportando UnstructuredGrid y PolyData."""
    reader = vtk.vtkUnstructuredGridReader()  # Cambiamos a UnstructuredGridReader
   # reader=vtk.vtkPolyDataReader()
    
    reader.SetFileName(vtk_file)
    reader.Update()

    unstructured_grid = reader.GetOutput()
    points_vtk = unstructured_grid.GetPoints()

    if points_vtk is None:
        raise ValueError("El archivo .vtk no contiene puntos válidos.")

    return vtk_to_numpy(points_vtk.GetData())



def cargar_nodos(nodos_file):
    """Carga un archivo de nodos, filtrando solo los que tienen un 4 en la segunda columna (purkinje)."""
    nodos = []
    with open(nodos_file, 'r') as f:
        lines = f.readlines()[2:]  # Omitimos las primeras dos líneas

        for line in lines:
            line = line.split("!")[0].strip()  # Eliminar comentarios después de '!'
            valores = line.split()
            if len(valores) < 5:
                continue  # Omitir líneas sin datos suficientes
            
            nodo_id, tipo, x, y, z = int(valores[0]), int(valores[1]), float(valores[2]), float(valores[3]), float(valores[4])
            
            if tipo == 4:  # Solo nos interesan los nodos donde el segundo valor es 4
                nodos.append((nodo_id, np.array([x, y, z])))

    return nodos



def encontrar_nodo_mas_cercano(punto, nodos):
    """Encuentra el nodo más cercano a un punto en el espacio."""
    nodo_mas_cercano = min(nodos, key=lambda nodo: np.linalg.norm(nodo[1] - punto))
    return nodo_mas_cercano[0]  # Retorna solo el ID del nodo

def procesar_archivos(vtk_file, nodos_file,write=None):
    """Carga los puntos, encuentra los nodos más cercanos y guarda los resultados."""
   
    puntos = cargar_puntos_vtk(vtk_file)
    nodos = cargar_nodos(nodos_file)
    
    elvira_node_IDs = []
    for i, punto in enumerate(puntos):
        nodo_id = encontrar_nodo_mas_cercano(punto, nodos)
        elvira_node_IDs.append(nodo_id)  
    
    # Si `write` es una ruta válida, escribir los resultados en el archivo indicado
    if write:
        with open(write, "w") as file:
            for nodo_id in elvira_node_IDs:
                file.write(f"{nodo_id}\n")

    return elvira_node_IDs









def align_and_cross_correlate(signal1, signal2, reference_lead=0, reference='signal2', 
                              generation=None, idx=None,write_corr_images=None, save_path='correlaciones'):
    """
    Alinea temporalmente dos señales de ECG de forma multicanal y calcula la correlación cruzada.

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
            
        reference_lead: señal que se tiene en cuenta para calcular el desplzamiento
        reference: 'signal1' o 'signal2'. Especifica cuál es la referencia fija.
        generation: int/str, generación para el título (opcional)
        idx: int/str, identificador para el título (opcional)
        save_path: str, carpeta donde guardar los plots

    Returns:
        aligned_signal1, aligned_signal2: señales alineadas
        cross_corrs: correlaciones por canal entre V1-V6
        mean_cross_corr: media de las correlaciones por canal entre V1-V6
    """
    
    assert signal1.shape == signal2.shape, "Las señales tienen que tener la misma longitud"
    assert reference in ['signal1', 'signal2'], "Referencia debe ser 'signal1' o 'signal2'"

    signal1_total=signal1
    signal2_total=signal2
    
    signal1=signal1[:,7:]
    signal2=signal2[:,7:]
    
       
    
    sig_ref = signal1 if reference == 'signal1' else signal2
    sig_mov = signal2 if reference == 'signal1' else signal1
    
    sig_ref_total = signal1_total if reference == 'signal1' else signal2_total
    sig_mov_total = signal2_total if reference == 'signal1' else signal1_total
    
    

    num_leads = sig_ref.shape[1]
    num_leads_total = sig_ref_total.shape[1]
    
    
    aligned_moving = np.zeros_like(sig_mov)
    aligned_moving_total = np.zeros_like(sig_mov_total)
    

    corr = correlate(sig_ref[:, reference_lead], sig_mov[:, reference_lead], mode='full')
    lag = np.argmax(corr) - (len(sig_ref) - 1)
    
    for i in range(0, num_leads):
        # Alinear aplicando desfase
        if lag > 0:
            aligned = np.pad(sig_mov[:, i], (lag, 0), mode='constant')[:len(sig_mov)]
            
        else:
            aligned = np.pad(sig_mov[:, i], (0, -lag), mode='constant')[-lag:len(sig_mov)-lag]
        aligned_moving[:, i] = aligned
        
        
    for i in range(0, num_leads_total):
        # Alinear aplicando desfase
        if lag > 0:
            aligned_total = np.pad(sig_mov_total[:, i], (lag, 0), mode='constant')[:len(sig_mov_total)]
            
        else:
            aligned_total = np.pad(sig_mov_total[:, i], (0, -lag), mode='constant')[-lag:len(sig_mov_total)-lag]
        aligned_moving_total[:, i] = aligned_total

    # Reasignar según cuál era la señal móvil
    aligned_signal1 = signal1 if reference == 'signal1' else aligned_moving
    aligned_signal2 = aligned_moving if reference == 'signal1' else signal2
    
    aligned_signal1_total = signal1_total if reference == 'signal1' else aligned_moving_total
    aligned_signal2_total = aligned_moving_total if reference == 'signal1' else signal2_total

    # Correlación por canal
    cross_corrs = [
        pearsonr(aligned_signal1[:, i], aligned_signal2[:, i])[0]
        for i in range(0, num_leads)
    ]
    mean_cross_corr = np.mean(cross_corrs)

    # --- PLOT de señales alineadas ---
    if write_corr_images:
        # Crear la carpeta si no existe
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        deriv_names = ['I','II','III','aVR','aVL','aVF','V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(15, 20))
        axes = axes.flatten()

        for i in range(0, num_leads_total-1):  # Solo V1-V6 (índices del 6 al 11 en las señales, pero en el plot son 0-5)
            #axes[i].plot(aligned_signal1[:, i ], label='Monodomain-simulated ECG', color='red', linestyle='-', linewidth=2.5, alpha=0.9)
            #axes[i].plot(aligned_signal2[:, i ], label='Clinical ECG', color='black', linestyle='--', linewidth=2.5, alpha=0.9)
            #axes[i].set_title(f'{deriv_names[i]} | Corr: {cross_corrs[i]:.2f}',fontsize=22)
            axes[i].plot(aligned_signal1_total[:, i+1 ], label=f'{deriv_names[i]}', color='red', linestyle='-', linewidth=2.5, alpha=0.9)
            
            if i>=6:
                           
                axes[i].plot(aligned_signal2_total[:, i+1], label=f'CC = {cross_corrs[i-6]:.2f}', color='black', linestyle='--', linewidth=2.5, alpha=0.9)
            else:
                axes[i].plot(aligned_signal2_total[:, i+1 ], color='black', linestyle='--', linewidth=2.5, alpha=0.9)
            
            
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
                
    
            
        # Mostrar la leyenda solo en el último gráfico
       # axes[-1].legend(fontsize='large')



        
        
        # Crear el título según los parámetros disponibles
       
        
        full_title=f"Mean correlation: {mean_cross_corr:.2f}"
        
       
        plt.suptitle(f' ECG({full_title})', fontsize=26)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Guardar el plot
        filename_parts = []
        if generation is not None:
            filename_parts.append(f"gen_{generation}")
        if idx is not None:
            filename_parts.append(f"idx_{idx}")
        filename_parts.append(f"corr_{mean_cross_corr:.2f}.svg")
        
        filename = "_".join(filename_parts)
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath)
        
        plt.close()
        
    return aligned_signal1, aligned_signal2, cross_corrs, mean_cross_corr


def load_manager_state_local(target_ecg_path,file_path,AHA_dict=None,output_csv_path=None,write_csv=None,best_nodes_path=None,write_corr_images=None):
    
    target_signal = np.loadtxt(target_ecg_path, delimiter=',')
    normalized_target_signal = normalizar_señales(target_signal)
    rows=[]
    

    """Carga todos los datos desde un único archivo."""
    try:
        with open(file_path, 'rb') as f:
            optimization_state= pickle.load(f)
            
        optimization_state['target_normalized_signal']=normalized_target_signal
        
        optimization_state['all_simulated_signals_normalized'] = [
            np.array(signal_list) for signal_list in optimization_state['all_simulated_signals_normalized']
            ] 
        
        optimization_state['all_simulated_signals_original'] = [
            np.array(signal_list) for signal_list in optimization_state['all_simulated_signals_original']
            ]
        
        optimization_state['ecg_best'] = np.array(optimization_state['ecg_best'])
        
        for gen in optimization_state['simulation_results'].keys():
            for idx in optimization_state['simulation_results'][gen].keys():
                optimization_state['simulation_results'][gen][idx]['simulated_signal_normalized']=np.array(optimization_state['simulation_results'][gen][idx]['simulated_signal_normalized'])
                optimization_state['simulation_results'][gen][idx]['simulated_signal_original']=np.array(optimization_state['simulation_results'][gen][idx]['simulated_signal_original'])
                optimization_state['simulation_results'][gen][idx]['solution_nodes'] = np.array(optimization_state['simulation_results'][gen][idx]['solution_nodes'])
                
                if output_csv_path!=None:
                    save_path=os.path.dirname(output_csv_path)
                
    
                    aligned_signal1, aligned_signal2, cross_corrs, mean_cross_corr = align_and_cross_correlate(
                        signal1=optimization_state['simulation_results'][gen][idx]['simulated_signal_normalized'][0:target_signal.shape[0],:],
                        signal2=normalized_target_signal[:,:],
                        reference_lead=0,
                        reference='signal2',
                        generation=gen, idx=idx,
                        write_corr_images=True,  # como keyword argument
                        save_path=save_path
                    )
                else:
                    aligned_signal1, aligned_signal2, cross_corrs, mean_cross_corr = align_and_cross_correlate(
                        signal1=optimization_state['simulation_results'][gen][idx]['simulated_signal_normalized'][0:target_signal.shape[0],:],
                        signal2=normalized_target_signal[:,:],
                        reference_lead=0,
                        reference='signal2',
                        
                    )
                    
                optimization_state['simulation_results'][gen][idx]['cross_corr']=np.array(cross_corrs)
                optimization_state['simulation_results'][gen][idx]['mean_cross_corr']=np.array(mean_cross_corr)
                
                
                if write_csv:
                    max_nodes = len(optimization_state['simulation_results'][gen][idx]['solution_nodes'])
                    
                    solution_nodes_list = list(optimization_state['simulation_results'][gen][idx]['solution_nodes'])
                    solution_node_dict = {f'solution_node_{i+1}': solution_nodes_list[i] if i < len(solution_nodes_list) else '' for i in range(max_nodes)}

                    row1 = {
                        'generation': gen,
                        'individual': idx,
                        
                    }
                    
                    row1.update(solution_node_dict)
                    
                    row2 = {
                        
                        'error': optimization_state['simulation_results'][gen][idx]['error'],
                        'mean_cross_corr': optimization_state['simulation_results'][gen][idx]['mean_cross_corr']
                    }
                    
                    row1.update(row2)
                            
                    if AHA_dict:
                        
                        rowAHA = {
                            '1': 0,
                            '2': 0,
                            '3': 0,
                            '4': 0,
                            '5': 0,
                            '6': 0,
                            '7': 0,
                            '8': 0,
                            '9': 0,
                            '10': 0,
                            '11': 0,
                            '12': 0,
                            '13': 0,
                            '14': 0,
                            '15': 0,
                            '16': 0,
                            '17': 0,
                            '18': 0,
                            '19': 0,
                            '20': 0,
                            '21': 0,
                            '22': 0,
                            '23': 0,
                            '24': 0,
                            '25': 0,
                            '26': 0,
                            '27': 0,
                            '28': 0,
                            '29': 0,
                            '30': 0,
                            '31': 0,
                            '32': 0,
                            '33': 0,
                            '34': 0
                            }
                        
                        if optim_type=='nodes':
                            for idx_sol in optimization_state['simulation_results'][gen][idx]['solution_nodes']:
                                key = AHA_dict[f'{idx_sol}']
                                if key in rowAHA:
                                    
                                    rowAHA[key] = 1
                                    
                            rows.append({**row1, **rowAHA})
                            
                        if optim_type=='sequence':  
                            for idx_sol in optimization_state['simulation_results'][gen][idx]['nodes_IDs']:
                                key = AHA_dict[f'{idx_sol}']
                                if key in rowAHA:
                                    
                                    rowAHA[key] = 1
                                    
                            rows.append({**row1, **rowAHA})
                            
                    else:
                        rows.append(row1)

        if write_csv:
            # Guardar CSV
            fieldnames = rows[0].keys()
            with open(output_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
                    
             
                
        return optimization_state
            
    except Exception as e:
        print(f"Error al cargar los datos: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    



def from_ID_to_vtk_result(target_ecg_path,optim_type,optimization_state_path, nodos_file, output_vtk_file):
    """
    Dado un conjunto de IDs, obtiene las coordenadas de los nodos correspondientes 
    y genera un archivo VTK con esas coordenadas como Unstructured Grid.
    
    ids_input: Lista de IDs de nodos a extraer.
    nodos_file: Archivo de nodos de entrada.
    output_vtk_file: Ruta del archivo VTK de salida.
    """
    optimization_state=load_manager_state_local(target_ecg_path,optimization_state_path,output_csv_path=None,write_csv=None,write_corr_images=None)
    
    
    
    
    # Cargar nodos desde el archivo
    nodos = cargar_nodos(nodos_file)
    # Crear un diccionario para acceso rápido a las coordenadas de los nodos por ID
    nodos_dict = {nodo_id: coordenadas for nodo_id, coordenadas in nodos}
    
    
    # Crear un objeto UnstructuredGrid para guardar las coordenadas filtradas
    unstructured_grid = vtk.vtkUnstructuredGrid()
    
    # Crear un objeto vtkPoints para agregar los puntos al UnstructuredGrid
    points_vtk = vtk.vtkPoints()
    
    # Arrays para almacenar los atributos
   
    gen_array = vtk.vtkIntArray()
    gen_array.SetName("gen")
    
    idx_array = vtk.vtkIntArray()
    idx_array.SetName("idx")
    
    error_array = vtk.vtkFloatArray()
    error_array.SetName("error")
    
    cross_mean_array = vtk.vtkFloatArray()
    cross_mean_array.SetName("mean_cross_corr")
    
    
    if optim_type=="sequence":
        activation_times_array = vtk.vtkFloatArray()
        activation_times_array.SetName("activation_times")
    elif optim_type=="nodes":
        nodeID_array = vtk.vtkIntArray()
        nodeID_array.SetName("NODES_ID")
        
        
    simulation_results=optimization_state['simulation_results']

    
    for gen in optimization_state['simulation_results'].keys(): 
        
        for idx in optimization_state['simulation_results'][gen].keys():
            
            error = simulation_results[gen][idx]['error']
            cross_mean= simulation_results[gen][idx]['mean_cross_corr']
            
            if optim_type=="sequence":
                ids_input = simulation_results[gen][idx]['nodes_IDs']  # Cargar listado de puntos desde un archivo de texto
                activation_times_input = simulation_results[gen][idx]['solution_nodes']   
            elif optim_type=="nodes":
                ids_input = simulation_results[gen][idx]['solution_nodes']  # Cargar listado de puntos desde un archivo de texto
        

            for nodo_id in ids_input:
                if nodo_id in nodos_dict:
                    # Agregar el punto
                    punto = nodos_dict[nodo_id]
                    vtk_index = points_vtk.InsertNextPoint(punto)
                    
                    # Asignar atributos correspondientes
                    if optim_type=="nodes":
                        nodeID_array.InsertNextValue(int(nodo_id))
                    idx_array.InsertNextValue(idx)
                    gen_array.InsertNextValue(gen)
                    error_array.InsertNextValue(error)
                    cross_mean_array.InsertNextValue(cross_mean)
                    
                    if optim_type=="sequence":
                        # Encontrar el índice de nodo_id en ids_input usando np.where
                        node_index = np.where(ids_input == nodo_id)[0][0]  # Devuelve el primer índice donde coincida
                        activation_times_array.InsertNextValue(float(activation_times_input[node_index]))
                        
                        
                    # Añadir celda de tipo VTK_VERTEX
                    unstructured_grid.InsertNextCell(vtk.VTK_VERTEX, 1, [vtk_index])
            
    # Asignar los puntos y atributos al UnstructuredGrid
    unstructured_grid.SetPoints(points_vtk)
    
    if optim_type=="nodes":
        unstructured_grid.GetPointData().AddArray(nodeID_array)
    unstructured_grid.GetPointData().AddArray(idx_array)
    unstructured_grid.GetPointData().AddArray(gen_array)
    unstructured_grid.GetPointData().AddArray(error_array)   
    unstructured_grid.GetPointData().AddArray(cross_mean_array) 
    
    if optim_type=="sequence":
        unstructured_grid.GetPointData().AddArray(activation_times_array) 
    
    
    # Guardar en archivo VTK
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(output_vtk_file)
    writer.SetInputData(unstructured_grid)
    writer.Write()
        
    print(f"Archivo VTK generado correctamente en: {output_vtk_file}")
    


############################################################################################################################
#####################################################################################################
##############################################################################
##############################################################

seeds=[42]

AHA_PSM4 = {
    '4171774': '1',
    '4171773': '1',
    '4180668': '2',
    '4180666': '2',
    '4180669': '2',
    '4180671': '2',
    '4180667':'2',
    '4194032': '3',
    '4194031': '3',
    '4194029': '3',
    '4194028': '3',
    '4188591': '4',
    '4188590': '4',
    '4188588': '4',
    '4171179': '5',
    '4171178': '5',
    '4171181':'5',
    '4151226': '6',
    '4119377': '7',
    '4119378':'7',
    '4119376': '7',
    '4155892': '8',
    '4155891': '8',
    '4155888': '8',
    '4155890': '8',
    '4201307': '9',
    '4201308': '9',
    '4202392': '10',
    '4202390': '10',
    '4202389':'10',
    '4130256': '11',
    '4130255': '11',
    '4149987': '12',
    '4149984': '12',
    '4149983': '12',
    '4149990': '12',
    '4149986':'12',
    '4117050':'13',
    '4117054': '13',
    '4161817': '14',
    '4161816': '14',
    '4126747': '15',
    '4126746': '15',
    '4116905': '16',
    '4133676': '17',
    '4133674': '17',
    '4133673': '17',
    '4133672': '17',
    '4133675': '17',
    '4268292': '18',
    '4255327': '25',
    '4238203': '20',
    '4238202': '20',
    '4259271': '21',
    '4259270': '21',
    '4262598': '22',
    '4262597': '22',
    '4264637': '23',
    '4256035': '24',
    '4236740': '25',
    '4239030': '26',
    '4231686': '27',
    '4231685': '27',
    '4240412': '28',
    '4233591': '29',
    '4212931': '30',
    '4222615': '31',
    '4211981': '32',
    '4211980': '32',
    '4212638': '33',
    '4212637': '33',
    '4202973': '34'
}


    
for seed in seeds:
    
    ruta_directorio = f'E:/jorge_presentacion/{seed}/results/'

    # Crear el directorio si no existe
    if not os.path.exists(ruta_directorio):
        os.makedirs(ruta_directorio)
        print(f'📁 Directorio creado: {ruta_directorio}')
    else:
        print(f'✅ El directorio ya existe: {ruta_directorio}')
        
        
        
    # Construcción dinámica de paths
    target_ecg_path = 'E:/jorge_presentacion/ID4_ECG_filtrado.csv'
    file_path = f'E:/jorge_presentacion/{seed}/optimization_state.pkl'
    output_csv_path = f'E:/jorge_presentacion/{seed}//results/result_{seed}.csv'
    output_vtk_file = f'E:/jorge_presentacion/{seed}/results/result_{seed}.vtk'
    nodos_file = f'E:/jorge_presentacion/NODES.dat'
    
    write_csv=True
    optim_type="sequence"
    
    
    load_manager_state_local(target_ecg_path,file_path,None,output_csv_path,write_csv,write_corr_images=True)
    from_ID_to_vtk_result(target_ecg_path,optim_type,file_path, nodos_file, output_vtk_file)

