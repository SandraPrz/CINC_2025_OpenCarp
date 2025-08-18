import numpy as np
from modules.simulation_utils import modify_stimulation_nodes, simulacion
from modules.signal_processing import ecg_calcul_normalized, ecg_calcul
from modules.ga_functions import dtw_camps
from skopt import gp_minimize
from modules.results_handler import save_manager_state
from modules.visualization import graficar_best
import pickle
import numba
import math
import shutil
import multiprocessing
from multiprocessing import Manager
import os
import sys
from skopt.callbacks import DeltaYStopper
from functools import partial


from SALib.sample import saltelli


from carputils import tools
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from dtwParallel import dtw_functions
from scipy.spatial import distance as d
from SALib.sample import sobol
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


    
def save_optimization_results(result, params):
    results_dict = {
        'best_parameters': result.x,
        'best_score': result.fun,
        'all_parameters': result.x_iters,
        'all_scores': result.func_vals,
        'optimization_params': params
    }
    
    with open('bayesian_optimization_results.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
        

def configure_algorithm(params):
    """Configura y retorna una instancia del algoritmo genético."""
    manager = Manager()
    # Calcular el tamaño máximo previsible
    max_size = params['n_calls']
    
    
    # Crear estructuras de datos compartidas
    shared_state = manager.dict()
    shared_state2 = manager.dict()
    
    # Inicializar simulation_results por adelantado
    gen=0
    gen_dict = manager.dict()  # Diccionario para esta generación
    for idx in range(max_size):  
        gen_dict[idx] = None  # Inicializar cada solución como None
    shared_state2[gen] = gen_dict
        
        
        
    # Inicializar listas con tamaño fijo y valores por defecto
    shared_state['all_simulated_signals_normalized'] = manager.list([None] * max_size)
    shared_state['all_simulated_signals_original'] = manager.list([None] * max_size)
    shared_state['all_errors'] = manager.list([float('inf')] * max_size)  # Usar 'inf' como valor por defecto para errores
    
    shared_state['best_error'] = float('inf')
    shared_state['ecg_best'] = None
    shared_state['best_absolute_index'] = -1
    shared_state['simulation_results'] = None
    shared_state['execution_counter'] = manager.Value('i', 0)  # 'i' indica un entero

    
    # Asignar estructuras compartidas
    optimization_lock = manager.Lock()
    optimization_state = shared_state
    simulation_results = shared_state2 
    
    

    #########################################33333

        
        
    geom_path=params['geom']


    vol_mesh = smart_reader(geom_path)
    vol_mesh = addGlobalIds(vol_mesh)
    surf_mesh = extract_surface(vol_mesh)
    endo_rv = threshold(surf_mesh, 0.9, 1, 'tm')
    endo_rv = threshold(endo_rv, 0.999, 1.01, 'tv')  # rango para atrapar valores ~1
    endo_lv = threshold(surf_mesh, 0.9, 1, 'tm')
    endo_lv = threshold(endo_lv, 0.0, 0.001, 'tv')  # rango para atrapar valores ~0

    #sacar los ids de los endocardios y sus uvcs

    ids_endo_rv = numpy_support.vtk_to_numpy(endo_rv.GetPointData().GetArray('GlobalIds'))
    ids_endo_lv = numpy_support.vtk_to_numpy(endo_lv.GetPointData().GetArray('GlobalIds'))

    ucoords_endo_rv = np.column_stack(get_cobivecoaux(vol_mesh, ids_endo_rv))
    ucoords_endo_lv = np.column_stack(get_cobivecoaux(vol_mesh, ids_endo_lv))


    #sacar las uvcs de los nodos de las disintas zonas

    


    # Diccionario con límites aproximados de coordenadas UVC para cada región
    # Cada límite es un tuple (min, max) para (longitudinal, circunferencial, radial)

    limits_uvc = {
        'ant_sept_lv': {
            'ab': [0, 0.73],   # zona media-apical entre 30% y 50%
            'rt': [0.47, 0.78],    # zona septal anterior en rotacional (~0°-36°)
            'tv': [0.999, 1.01],
            'tm': [0.9, 1.0],
            'ts': [1.999, 2.01]
        },
        'ant_lv': {
            'ab': [0.1, 0.7],   
            'rt': [0.25, 0.7],   # anterior-lateral (~36°-90°)
            'tv': [0.999, 1.01],
            'tm': [0.9, 1.0],
            'ts': [1.999, 2.01]
        },
        'post_lv': {
            'ab': [0.05, 0.7],   
            'rt': [0., 0.4],  # lateral-posterior (~90°-162°)
            'tv': [0.999, 1.01],
            'tm': [0.9, 1.0],
            'ts': [1.999, 2.01]
        },
        'sept_lv': {
            'ab': [0., 0.73],   
            'rt': [0.7, 1.0],   # septal izquierdo, final rotacional (~342°-360°)
            'tv': [0.999, 1.01],
            'tm': [0.9, 1.0],
            'ts': [1.999, 2.01]
        },
        'sept_rv': {
            'ab': [0., 0.73],   
            'rt': [0.7, 1.0],    # septal derecho (~324°-360°)
            'tv': [0.0, 0.001],
            'tm': [0.9, 1.0],
            'ts': [1.999, 2.01]
        },
        'mod_rv': {
            'ab': [0., 0.73],   
            'rt': [0.1, 0.6],    # moderador del RV (~252°-324°)
            'tv': [0.0, 0.001],
            'tm': [0.9, 1.0],
            'ts': [1.999, 2.01]
        },
        't_mod': {
            'time': [0, 25]    #preexcitacion banda moderadora
        }
    }


    '''
    vtkWrite(endo_lv,os.path.join(params['current_dir'],'endo_lv.vtu')) 
    vtkWrite(endo_rv,os.path.join(params['current_dir'],'endo_rv.vtu')) 

    #ant_sept
    ant_sept = threshold(endo_lv, limits_uvc['ant_sept_lv']['ab'][0], limits_uvc['ant_sept_lv']['ab'][1], 'ab')
    ant_sept = threshold(ant_sept, limits_uvc['ant_sept_lv']['rt'][0], limits_uvc['ant_sept_lv']['rt'][1], 'rt')
    ant_sept = threshold(ant_sept, limits_uvc['ant_sept_lv']['ts'][0], limits_uvc['ant_sept_lv']['ts'][1], 'ts')

    vtkWrite(ant_sept,os.path.join(params['current_dir'],'ant_sept.vtu'))

    #ant
    ant = threshold(endo_lv, limits_uvc['ant_lv']['ab'][0], limits_uvc['ant_lv']['ab'][1], 'ab')
    ant = threshold(ant, limits_uvc['ant_lv']['rt'][0], limits_uvc['ant_lv']['rt'][1], 'rt')
    ant = threshold(ant, limits_uvc['ant_lv']['ts'][0], limits_uvc['ant_lv']['ts'][1], 'ts')

    vtkWrite(ant,os.path.join(params['current_dir'],'ant.vtu'))

    #post
    post = threshold(endo_lv, limits_uvc['post_lv']['ab'][0], limits_uvc['post_lv']['ab'][1], 'ab')
    post = threshold(post, limits_uvc['post_lv']['rt'][0], limits_uvc['post_lv']['rt'][1], 'rt')
    post = threshold(post, limits_uvc['post_lv']['ts'][0], limits_uvc['post_lv']['ts'][1], 'ts')

    vtkWrite(post,os.path.join(params['current_dir'],'post.vtu'))

    #sept_lv
    sept_lv = threshold(endo_lv, limits_uvc['sept_lv']['ab'][0], limits_uvc['sept_lv']['ab'][1], 'ab')
    sept_lv = threshold(sept_lv, limits_uvc['sept_lv']['rt'][0], limits_uvc['sept_lv']['rt'][1], 'rt')
    sept_lv = threshold(sept_lv, limits_uvc['sept_lv']['ts'][0], limits_uvc['sept_lv']['ts'][1], 'ts')

    vtkWrite(sept_lv,os.path.join(params['current_dir'],'sept_lv.vtu'))

    #sept_rv
    sept_rv = threshold(endo_rv, limits_uvc['sept_rv']['ab'][0], limits_uvc['sept_rv']['ab'][1], 'ab')
    sept_rv = threshold(sept_rv, limits_uvc['sept_rv']['rt'][0], limits_uvc['sept_rv']['rt'][1], 'rt')
    sept_rv = threshold(sept_rv, limits_uvc['sept_rv']['ts'][0], limits_uvc['sept_rv']['ts'][1], 'ts')

    vtkWrite(sept_rv,os.path.join(params['current_dir'],'sept_rv.vtu'))

    #mod_rv
    mod_rv = threshold(endo_rv, limits_uvc['mod_rv']['ab'][0], limits_uvc['mod_rv']['ab'][1], 'ab')
    mod_rv = threshold(mod_rv, limits_uvc['mod_rv']['rt'][0], limits_uvc['mod_rv']['rt'][1], 'rt')
    mod_rv = threshold(mod_rv, limits_uvc['mod_rv']['ts'][0], limits_uvc['mod_rv']['ts'][1], 'ts')

    vtkWrite(mod_rv,os.path.join(params['current_dir'],'mod_rv.vtu'))

    '''

    num_vars= params['variables_sampler']
    names_list=['ant_sept_lv_ab','ant_lv_ab','post_lv_ab','sept_lv_ab','sept_rv_ab','mod_rv_ab','ant_sept_lv_rt','ant_lv_rt','post_lv_rt','sept_lv_rt','sept_rv_rt','mod_rv_rt','t_mod']
    bounds_list = []

    for name in names_list:
        if name == 't_mod':
            bounds_list.append(limits_uvc['t_mod']['time'])
            continue
        # separar parte de la región y tipo de coordenada (_ab, _rt)
        region, coord = name.rsplit('_', 1)
        
        # extraer límites del diccionario limits_uvc
        bounds = limits_uvc[region][coord]
        bounds_list.append(bounds)

    #print(bounds_list)

    n = params['muestras_iniciales']

    samples=Saltelli_sampler(num_vars, names_list, bounds_list, n, seed=params['random_seed'])
        

    points_uvc = [{} for _ in range(samples.shape[0])]

    #samples=abx6  rtx6  timex1
    for i in range(samples.shape[0]):
        points_uvc[i] = {
            'ant_sept_lv': {
                'ab': samples[i, 0],   # longitudinal
                'rt': samples[i, 6],   # rotacional
                'tv': 0,
                'tm': 1,
                'ts': 2
            },
            'ant_lv': {
                'ab': samples[i, 1],
                'rt': samples[i, 7],
                'tv': 0,
                'tm': 1,
                'ts': 2
            },
            'post_lv': {
                'ab': samples[i, 2],
                'rt': samples[i, 8],
                'tv': 0,
                'tm': 1,
                'ts': 2
            },
            'sept_lv': {
                'ab': samples[i, 3],
                'rt': samples[i, 9],
                'tv': 0,
                'tm': 1,
                'ts': 2
            },
            'sept_rv': {
                'ab': samples[i, 4],
                'rt': samples[i, 10],
                'tv': 1,
                'tm': 1,
                'ts': 2
            },
            'mod_rv': {
                'ab': samples[i, 5],
                'rt': samples[i, 11],
                'tv': 1,
                'tm': 1,
                'ts': 2
            },
            't_mod': {
                'time': samples[i, 12]
            }
        }


    best_params_reshape = np.zeros((6, 6, samples.shape[0])) # ab tm rt tv ts time

    for i,_ in enumerate(samples):
        best_params_reshape[0,:,i]=(points_uvc[i]['ant_sept_lv']['ab'],points_uvc[i]['ant_sept_lv']['tm'],points_uvc[i]['ant_sept_lv']['rt'],points_uvc[i]['ant_sept_lv']['tv'],points_uvc[i]['ant_sept_lv']['ts'],points_uvc[i]['t_mod']['time'])
        best_params_reshape[1,:,i]=(points_uvc[i]['ant_lv']['ab'],points_uvc[i]['ant_lv']['tm'],points_uvc[i]['ant_lv']['rt'],points_uvc[i]['ant_lv']['tv'],points_uvc[i]['ant_lv']['ts'],points_uvc[i]['t_mod']['time'])
        best_params_reshape[2,:,i]=(points_uvc[i]['post_lv']['ab'],points_uvc[i]['post_lv']['tm'],points_uvc[i]['post_lv']['rt'],points_uvc[i]['post_lv']['tv'],points_uvc[i]['post_lv']['ts'],points_uvc[i]['t_mod']['time'])
        best_params_reshape[3,:,i]=(points_uvc[i]['sept_lv']['ab'],points_uvc[i]['sept_lv']['tm'],points_uvc[i]['sept_lv']['rt'],points_uvc[i]['sept_lv']['tv'],points_uvc[i]['sept_lv']['ts'],points_uvc[i]['t_mod']['time'])
        best_params_reshape[4,:,i]=(points_uvc[i]['sept_rv']['ab'],points_uvc[i]['sept_rv']['tm'],points_uvc[i]['sept_rv']['rt'],points_uvc[i]['sept_rv']['tv'],points_uvc[i]['sept_rv']['ts'],points_uvc[i]['t_mod']['time'])
        best_params_reshape[5,:,i]=(points_uvc[i]['mod_rv']['ab'],points_uvc[i]['mod_rv']['tm'],points_uvc[i]['mod_rv']['rt'],points_uvc[i]['mod_rv']['tv'],points_uvc[i]['mod_rv']['ts'],0)


    np.save(
    os.path.join(params['current_dir'], "best_params_reshape.npy"),
    best_params_reshape
)


    kdtree_path=params['kd_tree_file']

    with open(kdtree_path, 'rb') as f:
        tree = pickle.load(f)


    vol_mesh = smart_reader(geom_path)
    vol_mesh = addGlobalIds(vol_mesh)


    ids = numpy_support.vtk_to_numpy(vol_mesh.GetPointData().GetArray('GlobalIds'))



    # Nombres de las regiones para los 7 puntos por individuo
    region_names = ["ant_sept_lv", "ant_lv", "post_lv", "sept_lv", "sept_rv", "mod_rv"]

    all_points = vtk.vtkPoints()
    all_cells = vtk.vtkCellArray()

    # Arrays para etiquetas
    individuo_array = vtk.vtkIntArray()
    individuo_array.SetName("Individuo")

    region_array = vtk.vtkStringArray()
    region_array.SetName("Region")

    global_ids_output_array = vtk.vtkIntArray()
    global_ids_output_array.SetName("GlobalIds")

    for i in range(best_params_reshape.shape[2]):  # loop sobre muestras

        _, idx = tree.query(best_params_reshape[:, :5, i], k=1, workers=-1)
        global_ids_selected = idx
        
        #print(global_ids_selected)
        #print(best_params_reshape[:, :5, i])

        selected_ids_np = np.array(global_ids_selected, dtype=np.int64)
        vtk_id_array = numpy_support.numpy_to_vtk(selected_ids_np, deep=True, array_type=vtk.VTK_ID_TYPE)
        vtk_id_array.SetName("GlobalIds")

        selection_node = vtk.vtkSelectionNode()
        selection_node.SetFieldType(vtk.vtkSelectionNode.POINT)
        selection_node.SetContentType(vtk.vtkSelectionNode.INDICES)
        selection_node.SetSelectionList(vtk_id_array)

        selection = vtk.vtkSelection()
        selection.AddNode(selection_node)

        extract_selection = vtk.vtkExtractSelection()
        extract_selection.SetInputData(0, vol_mesh)
        extract_selection.SetInputData(1, selection)
    
        extract_selection.Update()

        selected_points = vtk.vtkUnstructuredGrid.SafeDownCast(extract_selection.GetOutput())

        global_ids_array = selected_points.GetPointData().GetArray('GlobalIds')

        
        


        if not global_ids_array:
            raise RuntimeError("No se encontró el array 'GlobalIds' en selected_points")

        for p in range(selected_points.GetNumberOfPoints()):
            coord = selected_points.GetPoint(p)
            global_id = global_ids_array.GetValue(p)

            # Buscar todos los índices donde aparece este global_id
            indices = np.where(selected_ids_np == global_id)[0]

            if len(indices) == 0:
                # No encontrado, agregar con región unknown (opcional)
                pid = all_points.InsertNextPoint(coord)
                all_cells.InsertNextCell(1)
                all_cells.InsertCellPoint(pid)
                individuo_array.InsertNextValue(i)
                region_array.InsertNextValue("unknown")
                global_ids_output_array.InsertNextValue(global_id)
            else:
                # Para cada índice/región correspondiente al global_id, añadir un punto
                for region_idx in indices:
                    pid = all_points.InsertNextPoint(coord)
                    all_cells.InsertNextCell(1)
                    all_cells.InsertCellPoint(pid)
                    individuo_array.InsertNextValue(i)
                    region_array.InsertNextValue(region_names[region_idx])
                    global_ids_output_array.InsertNextValue(global_id)




    # Crear malla final
    output_grid = vtk.vtkUnstructuredGrid()
    output_grid.SetPoints(all_points)
    output_grid.SetCells(vtk.VTK_VERTEX, all_cells)
    output_grid.GetPointData().AddArray(individuo_array)
    output_grid.GetPointData().AddArray(region_array)
    output_grid.GetPointData().AddArray(global_ids_output_array)

    # Guardar archivo
    output_path = os.path.join(params['current_dir'],'sampled_points.vtu')

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(output_grid)
    writer.Write()



    print("Archivo guardado con etiquetas 'Individuo' y 'Region'")

    
    return optimization_lock,optimization_state,simulation_results,best_params_reshape





def Saltelli_sampler(num_vars, names_list, bounds_list, n, seed=None):
    problem = {
        'num_vars': num_vars,
        'names': names_list,
        'bounds': bounds_list
    }

    # Semilla global de numpy para reproducibilidad
    if seed is not None:
        np.random.seed(seed)

    # Generar muestras de Saltelli
    param_values = saltelli.sample(problem, N=n, calc_second_order=False)

    print(f"Total muestras generadas: {param_values.shape[0]}")
    return param_values





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





























def run_gillette_method_block_parallel(params, normalized_target_signal, optimization_lock, 
                                     optimization_state, simulation_results, best_params_reshape):
    total_samples = best_params_reshape.shape[2]
    block_size = params.get('num_cpu', 5)  # Usa 5 como default si no está definido
    
    with ProcessPoolExecutor(max_workers=block_size) as executor:
        # Divide en bloques de 'block_size'
        for start_idx in range(0, total_samples, block_size):
            end_idx = min(start_idx + block_size, total_samples)
            print(f"Procesando muestras {start_idx} a {end_idx-1}")
            
            # Ejecuta en paralelo para el bloque actual
            futures = []
            for i in range(start_idx, end_idx):
                futures.append(executor.submit(
                    run_gilette_method,
                    sample_idx=i,
                    params=params,
                    normalized_target_signal=normalized_target_signal,
                    optimization_lock=optimization_lock,
                    optimization_state=optimization_state,
                    simulation_results=simulation_results,
                    best_params_reshape=best_params_reshape
                ))
            
            # Solo espera a que terminen (las actualizaciones ya se hicieron dentro de cada worker)
            for future in futures:
                future.result()  # Esto propagará cualquier excepción
            
            




def run_gilette_method(sample_idx,params,normalized_target_signal, optimization_lock, optimization_state, simulation_results,best_params_reshape):
  
    ######################################################################################################
   
    
    
    times=  best_params_reshape[:, -1, sample_idx]
    
    kdtree_path=params['kd_tree_file']

    with open(kdtree_path, 'rb') as f:
        tree = pickle.load(f)

    _, idx = tree.query(best_params_reshape[:, :5, sample_idx], k=1, workers=-1)
    global_ids_selected = idx

    nodes_IDs=global_ids_selected




    with optimization_lock:
        absolute_index = optimization_state['execution_counter'].value
        optimization_state['execution_counter'].value += 1

    
    # Crear directorio para esta evaluación
    eval_dir = os.path.join(params['current_dir'], f"idx_{absolute_index}_pid_{os.getpid()}")
    os.makedirs(eval_dir, exist_ok=True)
    

    print(f'#####################################################################')
    print(f'#######   points: {nodes_IDs}')
    print(f'#######   times: {times}')
    print(f'#####################################################################')
    

    # Ejecutar simulación

    args_dict = {
        "job_name": eval_dir,
        "geom": params['geom'],
        "simulation_files": params['openCarp_simulationFiles'],
        "duration": params['OpenCarp_sim_duration'],
        "model": "MitchellSchaeffer",
        "myocardial_CV": params['OpenCarp_sim_myo_vel'],
        "initial_time": params['OpenCarp_sim_initial_time'],
        "manual_ids": nodes_IDs,
        "manual_times": times,
        "save_vtk": False,
        "output_vtk_name": "",
        "lead_file": params['lead_file'],
        "torso_file_um": params['torso_file_um'],
        "torso_file": params['torso_file'],
        "config_template": params['config_template'],
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
                        
        
        group.add_argument('--manual-ids',
                        type=str,
                        default=args_dict["manual_ids"],
                        help='Lista de IDs para estimulación manual, separados por comas (ej: "123,456,789")')
        group.add_argument('--manual-times',
                        type=str,
                        default=args_dict["manual_times"],
                        help='Lista de tiempos correspondientes, separados por comas (ej: "10,20,30")')
        
        
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
            
                if args.sinusal_mode == "only_times":
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

        unique_config_path = os.path.join(filenameorignal, 'config_refneg.json')
        shutil.copy(config_template_path, unique_config_path)
        
        
        modify_json_entry( file_path=unique_config_path, key="VTK_FILE", new_value=args.torso_file )
        modify_json_entry( file_path=unique_config_path, key="MEASUREMENT_LEADS", new_value=lead_ids )
        modify_json_entry(file_path=unique_config_path, key="VM_FILE", new_value=os.path.join(job.ID, "vm.dat"))
        modify_json_entry(file_path=unique_config_path, key="ECG_FILE", new_value=os.path.join(job.ID, "ecg_output.dat"))



        C_matrix_data, heart_node_indices, num_total_nodes, _ = compute_leadfield(unique_config_path)
        compute_ecg(C_matrix_data, heart_node_indices, num_total_nodes, unique_config_path)
        
        
        
        print(f"Output folder: {job.ID}")
        
        return global_ids_selected,original_stimTimes_sinusal2,os.path.join(job.ID, "ecg_output.dat")


    try:
        # Ejecutar la simulación
        argv = []
        stims_list_total,original_stimTimes_sinusal2,ecg_path = simulacion(argv)


        # Verificar si el archivo ECG existe
        if not os.path.exists(ecg_path):
            raise FileNotFoundError(f"No se encontró el archivo ECG: {ecg_path}")

        target_signal = np.loadtxt(params['target_ecg_path'], delimiter=',')
        

        # Calcular error
        optimization_function=params['optimization_function']
        derivation_num=params['derivation_num']
        meshVolume=params['meshVolume']

        if optimization_function=="dtw_camps":
            ECG_normalized = ecg_calcul_normalized(ecg_path)
            ECG_original = ecg_calcul(ecg_path)
            leads_data= np.loadtxt(ecg_path)
            # Calcular error usando DTW (Dynamic Time Warping)
            part_error, error = dtw_camps(ECG_normalized, normalized_target_signal, derivation_num,meshVolume)
        
             # Convertir arrays a listas para serialización
            ecg_norm_list = ECG_normalized.tolist() 
            ecg_orig_list = ECG_original.tolist()
        
        if optimization_function=="L2_norm":
            ECG_normalized = []
            
            leads_data= np.loadtxt(ecg_path)
            #part error es realmente la mejor tupla (delta, escala, offset)
            part_error, error , ECG_original  = L2_norm(ecg_path,params['target_ecg_path'])
            
             # Convertir arrays a listas para serialización
            ecg_norm_list = [] 
            ecg_orig_list = ECG_original.tolist()



        



        # Bloqueamos para evitar condiciones de carrera
        with optimization_lock:


            if params['optim_type']=="gilette_method":

            
                    simulation_results[0][absolute_index]={
                        "nodes_IDs": stims_list_total,
                        "solution_nodes": original_stimTimes_sinusal2,
                        "error": error,
                        "part_error": part_error,
                        "simulated_signal_normalized": ecg_norm_list,  # Matriz de la señal
                        "simulated_signal_original": ecg_orig_list,  # Matriz de la señal
                        "leads_data":leads_data
                    }
                    
            # Guardar en la posición preasignada
            optimization_state['all_simulated_signals_normalized'][absolute_index] = ecg_norm_list
            optimization_state['all_simulated_signals_original'][absolute_index] = ecg_orig_list
            optimization_state['all_errors'][absolute_index] = error
        


            # Actualizar el mejor resultado si corresponde
            
            if error < optimization_state['best_error']:
            
                    optimization_state['best_error'] = error
                    optimization_state['ecg_best'] = ecg_orig_list
                    optimization_state['best_absolute_index']=absolute_index
                    print(f"Nuevo mejor error: {error}")
                    sys.stdout.flush()  # Asegura que se imprima inmediatamente
        
            #guardar datos del estado de la simulación
            save_manager_state(optimization_state,simulation_results,"optimization_state.pkl","simulation_results.pkl")
        
        
            # Registrar resultados en archivo de log
            log_file = "registro_resultados.txt"
            file_exists = os.path.isfile(log_file)

            with open(log_file, "a") as f:
                if not file_exists:
                    
                    f.write("absolute_index,nodos_estimulados,solution_times,error,part_error\n")

                stim_times_str = ",".join(f"{float(x):.4f}" for x in original_stimTimes_sinusal2)
                f.write(f"{absolute_index},{stims_list_total},{stim_times_str},{error},{part_error}\n")
            
            
        
        
        # Generar gráficas comparativas
        # 1. Gráficas (usamos copias locales de los datos)
        best_error = optimization_state['best_error']
        best_signal = list(optimization_state['ecg_best']) if optimization_state['ecg_best'] else None
        # Convertir Manager.list a lista normal y filtrar valores None
        all_signals = [x for x in list(optimization_state['all_simulated_signals_original']) if x is not None]

        if best_signal is not None and len(all_signals) > 0:
            graficar_best(all_signals, best_signal, target_signal, best_error)
        else:
            print("Advertencia: No hay suficientes datos para graficar")


        # Limpiar archivos temporales
        try:
            shutil.rmtree(eval_dir)  # Elimina recursivamente la carpeta y su contenido
            print(f"✓ Directorio {eval_dir} eliminado")
            sys.stdout.flush()
        except Exception as e:
            print(f"Error al eliminar {eval_dir}: {str(e)}")
        

        return error


    except Exception as e:
        print(f"Error en la simulacion: {str(e)}")
        print(f"[ERROR] Error al evaluar iteracion={absolute_index}, solucion idx={stims_list_total}, stim_times_str = {','.join(f'{float(x):.4f}' for x in original_stimTimes_sinusal2)}: {e}")
        traceback.print_exc()

        # Un valor de error muy alto
        fake_error = 999999


        # Bloqueamos para evitar condiciones de carrera
        with optimization_lock:


            if params['optim_type']=="gilette_method":

            
                    simulation_results[0][absolute_index]={
                        "nodes_IDs":stims_list_total,
                        "solution_nodes": original_stimTimes_sinusal2,
                        "error": error,
                        "part_error": part_error,
                        "simulated_signal_normalized": ecg_norm_list,  # Matriz de la señal
                        "simulated_signal_original": ecg_orig_list,  # Matriz de la señal
                        "leads_data":leads_data
                    }
                    
            # Guardar en la posición preasignada
            optimization_state['all_simulated_signals_normalized'][absolute_index] = ecg_norm_list
            optimization_state['all_simulated_signals_original'][absolute_index] = ecg_orig_list
            optimization_state['all_errors'][absolute_index] = error


            #guardar datos del estado de la simulación
            save_manager_state(optimization_state,simulation_results,"optimization_state.pkl","simulation_results.pkl")
        
            # También puedes registrar en un log de errores si quieres
            with open("errores_simulacion.log", "a") as log_file:
                stim_times_str = ",".join(f"{float(x):.4f}" for x in original_stimTimes_sinusal2)
                log_file.write(f"Error en iteracion={absolute_index}, nodes={stims_list_total}, times={stim_times_str}:\n")
                log_file.write(traceback.format_exc())
                log_file.write("\n" + "="*80 + "\n")

        return fake_error




