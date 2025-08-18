import numpy as np
from modules.simulation_utils import modify_stimulation_nodes, simulacion
from modules.signal_processing import ecg_calcul_normalized, ecg_calcul, normalizar_señales
from modules.ga_functions import dtw_camps, L2_norm
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

import optuna
from carputils import tools
import traceback


from dtwParallel import dtw_functions
from scipy.spatial import distance as d
from optuna.integration import BoTorchSampler
from optuna.samplers import TPESampler
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
        

def configure_bayesian_algorithm(params):
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
    
    
    
    return optimization_lock,optimization_state,simulation_results









def optuna_objective(trial, nodes, params, target_signal, optimization_lock, optimization_state, simulation_results):
    
    normalized_target_signal = normalizar_señales(target_signal)
    
    if params['sinusal_mode']=='only_times':
        # Sugerir un tiempo para cada nodo (igual que antes)
        times = [trial.suggest_int(f"time_{i}", params['initial_time'], params['last_seq']+1) for i in range(len(nodes))]
    
        
         
        
        # Ejecutar la evaluación original
        error = objective_function_only_times(
            times,
            nodes,
            params,
            target_signal,
            optimization_lock,
            optimization_state,
            simulation_results
        )



    if params['sinusal_mode']=='points_times':
        #cargar ficheros de malla

        vol_mesh = smart_reader(params['geom'])
        vol_mesh = addGlobalIds(vol_mesh)
        surf_mesh = extract_surface(vol_mesh)
        endo_rv = threshold(surf_mesh, 0.9, 1, 'tm')
        endo_lv = threshold(surf_mesh, 0.9, 1, 'tm')

        #sacar los ids de los endocardios y sus uvcs

        ids_endo_rv = numpy_support.vtk_to_numpy(endo_rv.GetPointData().GetArray('GlobalIds'))
        ids_endo_lv = numpy_support.vtk_to_numpy(endo_lv.GetPointData().GetArray('GlobalIds'))

        ucoords_endo_rv = np.column_stack(get_cobivecoaux(vol_mesh, ids_endo_rv))
        ucoords_endo_lv = np.column_stack(get_cobivecoaux(vol_mesh, ids_endo_lv))

        #sacar las uvcs de los nodos iniciales de estimulacion

        
        initial_nodes_ids=nodes 
        ucoords_initial_nodes = np.zeros((len(nodes), 5))

        for i,id in enumerate(initial_nodes_ids):
            if id in ids_endo_rv:
                ucoords_initial_nodes[i,:]=np.column_stack(get_cobivecoaux(vol_mesh, [id]))
                
            
                

 
        #np.save("ucoords_initial_nodes.npy", ucoords_initial_nodes)
        
        ucoords_initial_nodes_der = ucoords_initial_nodes[ucoords_initial_nodes[:,3] == 1]
        ucoords_initial_nodes_izq = ucoords_initial_nodes[ucoords_initial_nodes[:,3] == 0]




        #sacar min y max ab tm rt tv ts 

        mins_rv = np.min(ucoords_initial_nodes_der, axis=0)
        maxs_rv = np.max(ucoords_initial_nodes_der, axis=0)

        mins_lv = np.min(ucoords_initial_nodes_izq, axis=0)
        maxs_lv = np.max(ucoords_initial_nodes_izq, axis=0)


        instancia = np.zeros((len(nodes), 6))

        #ab: 
        valor_base_ab=ucoords_initial_nodes[:,0]
        
        for i in range(len(nodes)):
            if ucoords_initial_nodes[i,3] == 0: #izquierdo
                base_ab = valor_base_ab[i]
                min_delta = mins_lv[0] - base_ab  
                max_delta = maxs_lv[0] - base_ab  

                delta_ab = trial.suggest_float(f"delta_ab_{i}", min_delta, max_delta)
                instancia[i,0]=base_ab+delta_ab

            else:	
                base_ab = valor_base_ab[i]
                min_delta = mins_rv[0] - base_ab  
                max_delta = maxs_rv[0] - base_ab  

                delta_ab = trial.suggest_float(f"delta_ab_{i}", min_delta, max_delta)
                instancia[i,0]=base_ab+delta_ab


        #tm: 
        valor_base_tm=ucoords_initial_nodes[:,1]
        
        for i in range(len(nodes)):
            if ucoords_initial_nodes[i,3] == 0: #izquierdo
                base_tm = valor_base_tm[i]
                min_delta = mins_lv[1] - base_tm  
                max_delta = maxs_lv[1] - base_tm  

                delta_tm = trial.suggest_float(f"delta_tm_{i}", min_delta, max_delta)
                instancia[i,1]=base_tm+delta_tm

            else:	
                base_tm = valor_base_tm[i]
                min_delta = mins_rv[1] - base_tm  
                max_delta = maxs_rv[1] - base_tm  

                delta_tm = trial.suggest_float(f"delta_tm_{i}", min_delta, max_delta)
                instancia[i,1]=base_tm+delta_tm

        
        #rt: 
        valor_base_rt=ucoords_initial_nodes[:,2]
        for i in range(len(nodes)):
            if ucoords_initial_nodes[i,3] == 0: #izquierdo
                base_rt = valor_base_rt[i]
                min_delta = mins_lv[2] - base_rt  
                max_delta = maxs_lv[2] - base_rt 

                delta_rt = trial.suggest_float(f"delta_rt_{i}", min_delta, max_delta)
                instancia[i,2]=base_rt+delta_rt

            else:	
                base_rt = valor_base_rt[i]
                min_delta = mins_rv[1] - base_rt  
                max_delta = maxs_rv[1] - base_rt  

                delta_rt = trial.suggest_float(f"delta_rt_{i}", min_delta, max_delta)
                instancia[i,2]=base_rt+delta_rt

        #tv ts time: 
        times = [trial.suggest_int(f"time_{i}", params['initial_time'], params['last_seq']+1) for i in range(len(nodes))]
    
        
    

        for i in range(len(nodes)):
                                
            instancia[i,3]=ucoords_initial_nodes[i,3]
            instancia[i,4]=2 #excluimos los puentes
            instancia[i,5]=times[i]


       
        # Ejecutar la evaluación original
        error = objective_function_only_times(
            times,
            nodes,
            params,
            target_signal,
            optimization_lock,
            optimization_state,
            simulation_results
        )




    if params['sinusal_mode']=='only_points':

        #cargar ficheros de malla

        vol_mesh = smart_reader(params['geom'])
        vol_mesh = addGlobalIds(vol_mesh)
        surf_mesh = extract_surface(vol_mesh)
        endo_rv = threshold(surf_mesh, 0.9, 1, 'tm')
        endo_lv = threshold(surf_mesh, 0.9, 1, 'tm')

        #sacar los ids de los endocardios y sus uvcs

        ids_endo_rv = numpy_support.vtk_to_numpy(endo_rv.GetPointData().GetArray('GlobalIds'))
        ids_endo_lv = numpy_support.vtk_to_numpy(endo_lv.GetPointData().GetArray('GlobalIds'))

        ucoords_endo_rv = np.column_stack(get_cobivecoaux(vol_mesh, ids_endo_rv))
        ucoords_endo_lv = np.column_stack(get_cobivecoaux(vol_mesh, ids_endo_lv))

        #sacar las uvcs de los nodos iniciales de estimulacion

        
        initial_nodes_ids=nodes 
        ucoords_initial_nodes = np.zeros((len(nodes), 5))

        for i,id in enumerate(initial_nodes_ids):
            if id in ids_endo_rv:
                ucoords_initial_nodes[i,:]=np.column_stack(get_cobivecoaux(vol_mesh, [id]))
                
            else:
                ucoords_initial_nodes[i,:]=np.column_stack(get_cobivecoaux(vol_mesh, [id]))
                

 
        #np.save("ucoords_initial_nodes.npy", ucoords_initial_nodes)
        
        ucoords_initial_nodes_der = ucoords_initial_nodes[ucoords_initial_nodes[:,3] == 1]
        ucoords_initial_nodes_izq = ucoords_initial_nodes[ucoords_initial_nodes[:,3] == 0]




        #sacar min y max ab tm rt tv ts 

        mins_rv = np.min(ucoords_initial_nodes_der, axis=0)
        maxs_rv = np.max(ucoords_initial_nodes_der, axis=0)

        mins_lv = np.min(ucoords_initial_nodes_izq, axis=0)
        maxs_lv = np.max(ucoords_initial_nodes_izq, axis=0)


        instancia = np.zeros((len(nodes), 6))

        #ab: 
        valor_base_ab=ucoords_initial_nodes[:,0]
        
        for i in range(len(nodes)):
            if ucoords_initial_nodes[i,3] == 0: #izquierdo
                base_ab = valor_base_ab[i]
                min_delta = mins_lv[0] - base_ab  
                max_delta = maxs_lv[0] - base_ab  

                delta_ab = trial.suggest_float(f"delta_ab_{i}", min_delta, max_delta)
                instancia[i,0]=base_ab+delta_ab

            else:	
                base_ab = valor_base_ab[i]
                min_delta = mins_rv[0] - base_ab  
                max_delta = maxs_rv[0] - base_ab  

                delta_ab = trial.suggest_float(f"delta_ab_{i}", min_delta, max_delta)
                instancia[i,0]=base_ab+delta_ab


        #tm: 
        valor_base_tm=ucoords_initial_nodes[:,1]
        
        for i in range(len(nodes)):
            if ucoords_initial_nodes[i,3] == 0: #izquierdo
                base_tm = valor_base_tm[i]
                min_delta = mins_lv[1] - base_tm  
                max_delta = maxs_lv[1] - base_tm  
                delta_tm = trial.suggest_float(f"delta_tm_{i}", min_delta, max_delta)
                instancia[i,1]=base_tm+delta_tm

            else:	
                base_tm = valor_base_tm[i]
                min_delta = mins_rv[1] - base_tm  
                max_delta = maxs_rv[1] - base_tm  

                delta_tm = trial.suggest_float(f"delta_tm_{i}", min_delta, max_delta)
                instancia[i,1]=base_tm+delta_tm

        
        #rt: 
        valor_base_rt=ucoords_initial_nodes[:,2]
        for i in range(len(nodes)):
            if ucoords_initial_nodes[i,3] == 0: #izquierdo
                base_rt = valor_base_rt[i]
                min_delta = mins_lv[2] - base_rt  
                max_delta = maxs_lv[2] - base_rt  

                delta_rt = trial.suggest_float(f"delta_rt_{i}", min_delta, max_delta)
                instancia[i,2]=base_rt+delta_rt

            else:	
                base_rt = valor_base_rt[i]
                min_delta = mins_rv[1] - base_rt  
                max_delta = maxs_rv[1] - base_rt  

                delta_rt = trial.suggest_float(f"delta_rt_{i}", min_delta, max_delta)
                instancia[i,2]=base_rt+delta_rt

        #tv ts time: 
        
        times = [0] * len(nodes)

        for i in range(len(nodes)):
                                
            instancia[i,3]=ucoords_initial_nodes[i,3]
            instancia[i,4]=2 #excluimos los puentes
            instancia[i,5]=times[i]

          
        
        # Ejecutar la evaluación original
        error = objective_function_only_times(
            times,
            nodes,
            params,
            target_signal,
            optimization_lock,
            optimization_state,
            simulation_results
        )
    

    
    return error


def run_optuna_optimization(nodes, params, target_signal, optimization_lock, optimization_state, simulation_results):
    # Elegir el sampler
    if params['optuna_sampler'] == "botorch":
        sampler = BoTorchSampler()
   
    if params['optuna_sampler'] == "TPE":
        sampler = TPESampler(n_startup_trials=params['intentos_iniciales'])

    # Crear estudio (sin pruning)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler)#,
        #storage=params.get("storage"),   # Ej. "sqlite:///optuna_study.db"
        #load_if_exists=True              # Retomar si ya existe
    #)

    # Función objetivo con parámetros fijos
    
    objective_func = lambda trial: optuna_objective(
            trial, nodes, params, target_signal, optimization_lock, optimization_state, simulation_results
        )


    # Optimizar
    study.optimize(
        objective_func,
        n_trials=params['n_calls'],
        n_jobs=params.get("n_jobs", params['num_cpu'])
    )

    return study



def objective_function_only_times(times, nodes, params,target_signal,optimization_lock,optimization_state,simulation_results):
    
    normalized_target_signal = normalizar_señales(target_signal)
    
    nodes_ids=nodes    
    # Convertir tiempos a enteros si es necesario
    times = np.array(times, dtype=int)

    

    with optimization_lock:
        absolute_index = optimization_state['execution_counter'].value
        optimization_state['execution_counter'].value += 1

     
    # Crear directorio para esta evaluación
    eval_dir = os.path.join(params['current_dir'], f"idx_{absolute_index}_pid_{os.getpid()}")
    os.makedirs(eval_dir, exist_ok=True)
    
     
    print(f'#####################################################################')
    print(f'#######   points: {nodes_ids}')
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
        "sinusal_mode": params['sinusal_mode'],
        "manual_ids": nodes_ids,
        "manual_times": times,
        "opt_file": eval_dir,
        "kdtree_file": params['kd_tree_file'],
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
                       
        group.add_argument('--sinusal-mode',
                       type=str,
                       choices=['only_times', 'points_times'],
                       default=args_dict["sinusal_mode"],
                       help='Origen de la estimulación sinusal (only_times u points_times)')
        
        
        group.add_argument('--manual-ids',
                       type=str,
                       default=args_dict["manual_ids"],
                       help='Lista de IDs para estimulación manual, separados por comas (ej: "123,456,789")')
        group.add_argument('--manual-times',
                       type=str,
                       default=args_dict["manual_times"],
                       help='Lista de tiempos correspondientes, separados por comas (ej: "10,20,30")')
        
        group.add_argument('--opt-file',
                       type=str,
                       default='',
                       help='Ruta al archivo .npz con resultados de optimización sinusal')
        group.add_argument('--kdtree-file',
                       type=str,
                       default='',
                       help='Ruta al archivo .pkl del KDTree (solo en modo points_times/point)')
        
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
        
        if args.sinusal_mode == "only_times":
            
         
            ###################
            ## ONLY TIMES MODE
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
            

        if args.sinusal_mode == "points_times":
        
        
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
            original_stimTimes_sinusal2 = [x - min(original_stimTimes_sinusal)+ args.initial_time for x in original_stimTimes_sinusal] #start sinusal stimulation at 0ms
            
            
        
        
        
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

        unique_config_path = os.path.join(filenameorignal, 'config.json')
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
        
        if optimization_function=="dtwParallel_multi":
            ECG_normalized = ecg_calcul_normalized(ecg_path)
            ECG_original = ecg_calcul(ecg_path)
            leads_data= np.loadtxt(ecg_path)
            part_error, error = dtwParallel_multi(ECG_normalized, normalized_target_signal, type_dtw="d", local_dissimilarity=d.euclidean, MTS=True)

            # Convertir arrays a listas para serialización
            ecg_norm_list = ECG_normalized.tolist() 
            ecg_orig_list = ECG_original.tolist() 
 
 
        if optimization_function=="dtwParallel_uni":
            ECG_normalized = ecg_calcul_normalized(ecg_path)
            ECG_original = ecg_calcul(ecg_path)
            leads_data= np.loadtxt(ecg_path)
            part_error, error = dtwParallel_uni(ECG_normalized, normalized_target_signal, local_dissimilarity=d.euclidean, constrained_path_search="itakura", get_visualization=False)

            # Convertir arrays a listas para serialización
            ecg_norm_list = ECG_normalized.tolist() 
            ecg_orig_list = ECG_original.tolist() 
 

 
        if optimization_function=="autoCC_lags":
            ECG_normalized = ecg_calcul_normalized(ecg_path)
            ECG_original = ecg_calcul(ecg_path)
            leads_data= np.loadtxt(ecg_path)
            part_error, error, std_lags = autoCC_lags(ECG_normalized[:,1:], normalized_target_signal[:,1:], reference='signal2',write_corr_images=True, save_path=os.path.join('/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/PRUEBAS/nodos/seed_42_realPSM4',f'gen_{current_generation}_sol_{solution_idx}.png'))
 
            # Convertir arrays a listas para serialización
            ecg_norm_list = ECG_normalized.tolist() 
            ecg_orig_list = ECG_original.tolist() 
 

 
        if optimization_function=="autoCC_lags":
            ECG_normalized = ecg_calcul_normalized(ecg_path)
            ECG_original = ecg_calcul(ecg_path)
            leads_data= np.loadtxt(ecg_path)
            part_error, error = dtwCamps_stdLags(ECG_normalized, normalized_target_signal, derivation_num, meshVolume,reference='signal2', write_corr_images=False, save_path=None)
 
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
        
    
    
            # Actualizar el mejor resultado si corresponde
         
            if error < optimization_state['best_error']:
            
                    optimization_state['best_error'] = error
                    
                    if params['optim_type']=="sequence":
                        optimization_state['ecg_best'] = ecg_norm_list
                        
                    if params['optim_type']=="gilette_method":
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
                f.write(f"{absolute_index},{stims_list_total},{times.tolist()},{error},{part_error}\n")
            
            
       
        
        # Generar gráficas comparativas
        # 1. Gráficas (usamos copias locales de los datos)
        best_error = optimization_state['best_error']
        best_signal = list(optimization_state['ecg_best']) if optimization_state['ecg_best'] else None
        # Convertir Manager.list a lista normal y filtrar valores None
        
        if params['optim_type']=="sequence":
            all_signals = [x for x in list(optimization_state['all_simulated_signals_normalized']) if x is not None]
            if best_signal is not None and len(all_signals) > 0:
                graficar_best(all_signals, best_signal, normalized_target_signal, best_error)
            else:
                print("Advertencia: No hay suficientes datos para graficar")
    
        if params['optim_type']=="gilette_method":
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
            print(f"⚠️ Error al eliminar {eval_dir}: {str(e)}")
        
 
        return error
    
    
    except Exception as e:
        print(f"Error en la simulacion: {str(e)}")
        print(f"[ERROR] Error al evaluar solucion idx={nodes_ids}, evaluacion={absolute_index}: {e}")
        traceback.print_exc()

        # Un valor de error muy alto
        fake_error = 999999


        # Bloqueamos para evitar condiciones de carrera
        with optimization_lock:


            

           
            simulation_results[0][absolute_index]={
                "nodes_IDs":list(params['nodes_IDs']),
                "solution_nodes": times,
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
                log_file.write(f"Error en isntancia={absolute_index}, sol={times}:\n")
                log_file.write(traceback.format_exc())
                log_file.write("\n" + "="*80 + "\n")

        return fake_error





'''
def objective_function_points_times(instancia, nodes, params,target_signal,optimization_lock,optimization_state,simulation_results):
    
    normalized_target_signal = normalizar_señales(target_signal)
    

    nodes_ids=nodes

    with optimization_lock:
        absolute_index = optimization_state['execution_counter'].value
        optimization_state['execution_counter'].value += 1

     
    # Crear directorio para esta evaluación
    eval_dir = os.path.join(params['current_dir'], f"idx_{absolute_index}_pid_{os.getpid()}")
    os.makedirs(eval_dir, exist_ok=True)
    
     
    print(f'#####################################################################')
    print(f'#######   points: {nodes_ids}')
    print(f'#######   times: {instancia[:,-1]}')
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
        "sinusal_mode": params['sinusal_mode'],
        "manual_ids": [],
        "manual_times": [],
        "opt_file": instancia,
        "kdtree_file": params['kd_tree_file'],
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
                       
        group.add_argument('--sinusal-mode',
                       type=str,
                       choices=['only_times', 'points_times'],
                       default=args_dict["sinusal_mode"],
                       help='Origen de la estimulación sinusal (manual u points_times)')
        
        
        group.add_argument('--manual-ids',
                       type=str,
                       default=args_dict["manual_ids"],
                       help='Lista de IDs para estimulación manual, separados por comas (ej: "123,456,789")')
        group.add_argument('--manual-times',
                       type=str,
                       default=args_dict["manual_times"],
                       help='Lista de tiempos correspondientes, separados por comas (ej: "10,20,30")')
        
        group.add_argument('--opt-file',
                       type=str,
                       default=instancia,
                       help='Ruta al archivo .npz con resultados de optimización sinusal')
        group.add_argument('--kdtree-file',
                       type=str,
                       default=args_dict["kdtree_file"],
                       help='Ruta al archivo .pkl del KDTree (solo en modo points_times)')
        
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
        
        if args.sinusal_mode == "only_times":
            
         
            ###################
            ## ONLY TIMES MODE
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
            

        if args.sinusal_mode == "points_times":
        
        
            #########################
            ## FROM OPTIMIZATION MODE
            #########################
            
            
            best_params_reshape = args.opt_file # ab tm rt tv ts time   
            
            
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
            original_stimTimes_sinusal2 = [x - min(original_stimTimes_sinusal)+ args.initial_time for x in original_stimTimes_sinusal] #start sinusal stimulation at 0ms
            
            
        
        
        
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
        
        if optimization_function=="dtwParallel_multi":
            ECG_normalized = ecg_calcul_normalized(ecg_path)
            ECG_original = ecg_calcul(ecg_path)
            leads_data= np.loadtxt(ecg_path)
            part_error, error = dtwParallel_multi(ECG_normalized, normalized_target_signal, type_dtw="d", local_dissimilarity=d.euclidean, MTS=True)

            # Convertir arrays a listas para serialización
            ecg_norm_list = ECG_normalized.tolist() 
            ecg_orig_list = ECG_original.tolist() 
 
 
        if optimization_function=="dtwParallel_uni":
            ECG_normalized = ecg_calcul_normalized(ecg_path)
            ECG_original = ecg_calcul(ecg_path)
            leads_data= np.loadtxt(ecg_path)
            part_error, error = dtwParallel_uni(ECG_normalized, normalized_target_signal, local_dissimilarity=d.euclidean, constrained_path_search="itakura", get_visualization=False)

            # Convertir arrays a listas para serialización
            ecg_norm_list = ECG_normalized.tolist() 
            ecg_orig_list = ECG_original.tolist() 
 

 
        if optimization_function=="autoCC_lags":
            ECG_normalized = ecg_calcul_normalized(ecg_path)
            ECG_original = ecg_calcul(ecg_path)
            leads_data= np.loadtxt(ecg_path)
            part_error, error, std_lags = autoCC_lags(ECG_normalized[:,1:], normalized_target_signal[:,1:], reference='signal2',write_corr_images=True, save_path=os.path.join('/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/PRUEBAS/nodos/seed_42_realPSM4',f'gen_{current_generation}_sol_{solution_idx}.png'))
 
            # Convertir arrays a listas para serialización
            ecg_norm_list = ECG_normalized.tolist() 
            ecg_orig_list = ECG_original.tolist() 
 

 
        if optimization_function=="autoCC_lags":
            ECG_normalized = ecg_calcul_normalized(ecg_path)
            ECG_original = ecg_calcul(ecg_path)
            leads_data= np.loadtxt(ecg_path)
            part_error, error = dtwCamps_stdLags(ECG_normalized, normalized_target_signal, derivation_num, meshVolume,reference='signal2', write_corr_images=False, save_path=None)
 
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
                    
                    if params['optim_type']=='sequence':
                        optimization_state['ecg_best'] = ecg_norm_list
                    
                    if params['optim_type']=='gilette_method':
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
        
        if params['optim_type']=='sequence':
            all_signals = [x for x in list(optimization_state['all_simulated_signals_normalized']) if x is not None]
        
            if best_signal is not None and len(all_signals) > 0:
                graficar_best(all_signals, best_signal, normalized_target_signal, best_error)
            else:
                print("Advertencia: No hay suficientes datos para graficar")
        
        if params['optim_type']=='gilette_method':
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


'''

def objective_function_only_points(instancia, nodes, params,target_signal,optimization_lock,optimization_state,simulation_results):
    
    normalized_target_signal = normalizar_señales(target_signal)

    nodes_ids=nodes

    with optimization_lock:
        absolute_index = optimization_state['execution_counter'].value
        optimization_state['execution_counter'].value += 1

     
    # Crear directorio para esta evaluación
    eval_dir = os.path.join(params['current_dir'], f"idx_{absolute_index}_pid_{os.getpid()}")
    os.makedirs(eval_dir, exist_ok=True)
    
     
    print(f'#####################################################################')
    print(f'#######   points: {nodes_ids}')
    print(f'#######   times: {instancia[:,-1]}')
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
        "sinusal_mode": params['sinusal_mode'],
        "manual_ids": [],
        "manual_times": [],
        "opt_file": instancia,
        "kdtree_file": params['kd_tree_file'],
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
                       
        group.add_argument('--sinusal-mode',
                       type=str,
                       choices=['only_times', 'points_times', 'only_points'],
                       default=args_dict["sinusal_mode"],
                       help='Origen de la estimulación sinusal (manual u points_times)')
        
        
        group.add_argument('--manual-ids',
                       type=str,
                       default=args_dict["manual_ids"],
                       help='Lista de IDs para estimulación manual, separados por comas (ej: "123,456,789")')
        group.add_argument('--manual-times',
                       type=str,
                       default=args_dict["manual_times"],
                       help='Lista de tiempos correspondientes, separados por comas (ej: "10,20,30")')
        
        group.add_argument('--opt-file',
                       type=str,
                       default=instancia,
                       help='Ruta al archivo .npz con resultados de optimización sinusal')
        group.add_argument('--kdtree-file',
                       type=str,
                       default=args_dict["kdtree_file"],
                       help='Ruta al archivo .pkl del KDTree (solo en modo points_times)')
        
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
        
        if args.sinusal_mode == "only_times":
            
         
            ###################
            ## ONLY TIMES MODE
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
            

        if args.sinusal_mode == "points_times" or args.sinusal_mode == "only_points":
        
        
            #########################
            ## FROM OPTIMIZATION MODE
            #########################
            
            
            best_params_reshape = args.opt_file # ab tm rt tv ts time   
            
            
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
            original_stimTimes_sinusal2 = [x - min(original_stimTimes_sinusal)+ args.initial_time for x in original_stimTimes_sinusal] #start sinusal stimulation at 0ms
            
            
        
        
        
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


       
        # Calcular error
        optimization_function=params['optimization_function']
        derivation_num=params['derivation_num']
        meshVolume=params['meshVolume']

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
        
        if optimization_function=="dtwParallel_multi":
            ECG_normalized = ecg_calcul_normalized(ecg_path)
            ECG_original = ecg_calcul(ecg_path)
            leads_data= np.loadtxt(ecg_path)
            part_error, error = dtwParallel_multi(ECG_normalized, normalized_target_signal, type_dtw="d", local_dissimilarity=d.euclidean, MTS=True)

            # Convertir arrays a listas para serialización
            ecg_norm_list = ECG_normalized.tolist() 
            ecg_orig_list = ECG_original.tolist() 
 
 
        if optimization_function=="dtwParallel_uni":
            ECG_normalized = ecg_calcul_normalized(ecg_path)
            ECG_original = ecg_calcul(ecg_path)
            leads_data= np.loadtxt(ecg_path)
            part_error, error = dtwParallel_uni(ECG_normalized, normalized_target_signal, local_dissimilarity=d.euclidean, constrained_path_search="itakura", get_visualization=False)

            # Convertir arrays a listas para serialización
            ecg_norm_list = ECG_normalized.tolist() 
            ecg_orig_list = ECG_original.tolist() 
 

 
        if optimization_function=="autoCC_lags":
            ECG_normalized = ecg_calcul_normalized(ecg_path)
            ECG_original = ecg_calcul(ecg_path)
            leads_data= np.loadtxt(ecg_path)
            part_error, error, std_lags = autoCC_lags(ECG_normalized[:,1:], normalized_target_signal[:,1:], reference='signal2',write_corr_images=True, save_path=os.path.join('/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/PRUEBAS/nodos/seed_42_realPSM4',f'gen_{current_generation}_sol_{solution_idx}.png'))
 
            # Convertir arrays a listas para serialización
            ecg_norm_list = ECG_normalized.tolist() 
            ecg_orig_list = ECG_original.tolist() 
 

 
        if optimization_function=="autoCC_lags":
            ECG_normalized = ecg_calcul_normalized(ecg_path)
            ECG_original = ecg_calcul(ecg_path)
            leads_data= np.loadtxt(ecg_path)
            part_error, error = dtwCamps_stdLags(ECG_normalized, normalized_target_signal, derivation_num, meshVolume,reference='signal2', write_corr_images=False, save_path=None)
 
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
                    
                    if params['optim_type']=="sequence":
                        optimization_state['ecg_best'] = ecg_norm_list
                        
                    if params['optim_type']=="gilette_method":
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
        if params['optim_type']=="sequence":
            all_signals = [x for x in list(optimization_state['all_simulated_signals_normalized']) if x is not None]
            if best_signal is not None and len(all_signals) > 0:
                graficar_best(all_signals, best_signal, normalized_target_signal, best_error)
            else:
                print("Advertencia: No hay suficientes datos para graficar")
        if params['optim_type']=="gilette_method":
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


            if params['optim_type']=="sequence":

           
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