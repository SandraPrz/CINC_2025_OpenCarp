import json
import base64
from io import BytesIO
import time
from modules.visualization import graficar_best
import pickle
import sys
from multiprocessing import Manager
import matplotlib.pyplot as plt
import time
import numpy as np
import copy

from multiprocessing.managers import DictProxy

######## FUNCIONES DE GUARDADO DE RESULTADOS

'''
def save_optimization_state(state, filename="optimization_state.pkl"):
    """
    Guarda el estado completo de optimización usando pickle.
    Maneja cualquier objeto Python serializable.

    Parámetros:
    -----------
    state : dict
        Diccionario con el estado de optimización
    filename : str
        Nombre del archivo de salida (.pkl recomendado)
    """
    with open(filename, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Estado guardado en {filename} (formato binario pickle)")
    sys.stdout.flush()
'''





def save_manager_state(general_state_dict, simulation_results_dict, 
                     file_path1="optimization_state.pkl", 
                     file_path2="simulation_results.pkl",write=True):
    """Guarda todos los datos del Manager de manera segura."""
    try:
        # Convertir datos generales
        data_to_save = {
            'all_simulated_signals_normalized': list(general_state_dict.get('all_simulated_signals_normalized', [])),
            'all_simulated_signals_original': list(general_state_dict.get('all_simulated_signals_original', [])),
            'all_errors': list(general_state_dict.get('all_errors', [])),
            'best_error': float(general_state_dict.get('best_error', float('inf'))),
            'ecg_best': list(general_state_dict['ecg_best']) if general_state_dict.get('ecg_best') is not None else None,
            'best_absolute_index': int(general_state_dict.get('best_absolute_index', -1)),
            'simulation_results': {}  # Inicializado como dict vacío
        }

        # Convertir simulation_results
        for gen in simulation_results_dict.keys():
            if simulation_results_dict[gen] is None:
                continue
                
            data_to_save['simulation_results'][gen] = {}
            
            for sol in simulation_results_dict[gen].keys():
                solution_data = simulation_results_dict[gen][sol]
                if solution_data is None:
                    continue
                    
                data_to_save['simulation_results'][gen][sol] = {
                    'solution_nodes': list(solution_data.get('solution_nodes', [])),
                    'error': float(solution_data.get('error', float('inf'))),
                    'part_error': list(solution_data.get('part_error', (0.0, 0.0, 0.0))),
                    'simulated_signal_normalized': list(solution_data.get('simulated_signal_normalized', [])),
                    'simulated_signal_original': list(solution_data.get('simulated_signal_original', [])),
                    'leads_data': np.array(solution_data.get('leads_data', []))
                }
                # Añadir nodes_IDs si existe
                if 'nodes_IDs' in solution_data:
                    data_to_save['simulation_results'][gen][sol]['nodes_IDs'] = list(solution_data['nodes_IDs'])
        
        if write==True:
            # Guardar ambos archivos
            with open(file_path1, 'wb') as f1:
                pickle.dump(data_to_save, f1)
                
            with open(file_path2, 'wb') as f2:
                pickle.dump(data_to_save['simulation_results'], f2)
                
            print(f"Datos guardados correctamente en {file_path1} y {file_path2}")
        
        return data_to_save   
        
    except Exception as e:
        print(f"Error al guardar: {str(e)}")
        import traceback
        traceback.print_exc()


def load_manager_state_local(file_path="optimization_state.pkl"):
    """Carga todos los datos desde un único archivo."""
    try:
        with open(file_path, 'rb') as f:
            optimization_state= pickle.load(f)
            
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
                optimization_state['simulation_results'][gen][idx]['leads_data']=np.array(optimization_state['simulation_results'][gen][idx]['leads_data'])
                optimization_state['simulation_results'][gen][idx]['solution_nodes'] = np.array(optimization_state['simulation_results'][gen][idx]['solution_nodes'])

        return optimization_state
            
    except Exception as e:
        print(f"Error al cargar los datos: {str(e)}")
        import traceback
        traceback.print_exc()
        return None



# Guardar la instancia GA después del entrenamiento
def save_ga_instance(ga_instance, filename="ga_instance.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(ga_instance, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Instancia GA guardada en {filename}")
    sys.stdout.flush()

# Cargar la instancia posteriormente
def load_ga_instance(filename="ga_instance.pkl"):
    with open(filename, 'rb') as f:
        return pickle.load(f)








def guardar_resultados_completos(params, opt_state, target, 
                               ga_instance, filename="resultados_completos.pkl"):
    """
    Guarda todos los resultados en un único archivo .pkl
    
    Args:
        params: Parámetros de entrada
        opt_state: Estado de optimización (puede ser Manager.dict())
        target: Señal objetivo
        ga_instance: Instancia del algoritmo genético
        filename: Nombre del archivo de salida (.pkl)
    """
    # Generar gráfica en memoria
    buf = BytesIO()
    graficar_best(opt_state['all_simulated_signals_normalized'], opt_state['best_ecg'], target, opt_state['best_error'], final='yes')
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    buf.seek(0)
    
    optimization_state=save_manager_state(optimization_state,simulation_results,write=False)
    
    # Crear estructura de datos completa
    resultados = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0"
        },
        "params": params,
        "ecg_target": target,
        "ga_instance": ga_instance,
        "plot_bytes": buf.getvalue()
    }
    
    # Guardar a archivo .pkl
    with open(filename, 'wb') as f:
        pickle.dump(resultados, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✓ Todos los resultados guardados en {filename} (formato pickle)")
    sys.stdout.flush()



