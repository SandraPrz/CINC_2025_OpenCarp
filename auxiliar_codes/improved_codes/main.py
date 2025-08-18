# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:51:59 2025
@author: Sandra

Script principal para la optimización de estimulación cardíaca.
"""
import os
import sys
import json
import numpy as np

from modules.signal_processing import normalizar_señales, ecg_calcul, ecg_calcul_normalized
from modules.simulation_utils import modify_stimulation_nodes
from modules.ga_functions import fitness_func, custom_mutation, setup_initial_population_times,setup_initial_population_precomputed_points , configure_genetic_algorithm
from modules.visualization import graficar_best
from modules.results_handler import guardar_resultados_completos,save_ga_instance
#from total_set_creator_vtk import from_ID_to_vtk_result
from datetime import (date, datetime)
import vtk
import pickle
import shutil
from scipy.io import savemat
from scipy.spatial import cKDTree
import json
from vtk.util import numpy_support


from carputils import settings
from carputils import tools
from carputils import mesh
from carputils.carpio import txt

from modules.stimOpts import stimBlock
from modules.simulation import simulation_eik
from modules.functions import (create_vtk_from_nodes, smart_reader, extract_surface, threshold, 
                        get_cobivecoaux, vtkWrite, addGlobalIds, farthest_point_sampling_global_ids,get_closest_global_ids, get_closest_global_ids_sorted_by_label,modify_json_entry,
                        )
from modules.leadfield import (compute_leadfield, compute_ecg)


def load_parameters(params_file):
    """Carga los parámetros desde el archivo JSON."""
    with open(params_file, 'r') as f:
        return json.load(f)


def initialize_nodes(params):
    """Carga y procesa los nodos según el tipo de archivo."""
    sys.path.append(params['modules_path'])
    #import total_set_creator_vtk
    
    set=np.loadtxt(params['nodes_IDs_path'], dtype=int)
    print(f"El set de valores posibles para nodos de estimulación se ha leido con un total de {len(set)} valores únicos")
    print(set)
    sys.stdout.flush()
    return set


def log_initial_population(population):
    """Registra la población inicial en un archivo de log."""
    with open("registro_resultados.txt", "a") as f:
        if not os.path.isfile("registro_resultados.txt"):
            f.write("initial population\n")
        f.write(f"{list(population)}\n")



def main():
    # Cargar parámetros
    if len(sys.argv) < 2:
        print("Error: Debes proporcionar el archivo de parámetros como argumento")
        sys.exit(1)
        
    params = load_parameters(sys.argv[1])
    print(f"Parámetros cargados: {params}")
    
    # Inicializar estado de optimización
    total_evaluaciones = params['num_generations'] * params['sol_per_pop']
    
    
    
      
    # Cargar nodos
    nodes_IDs = initialize_nodes(params)
    params['nodes_IDs']=nodes_IDs
    
    # Cargar y normalizar señal objetivo
    target_signal = np.loadtxt(params['target_ecg_path'], delimiter=',')
    normalized_target_signal = normalizar_señales(target_signal)
    
    # Configurar población inicial
    if params['optim_type'] == "nodes":
        initial_population = setup_initial_population_precomputed_points(params)
    
    if params['optim_type'] == "sequence":
        initial_population, set_values = setup_initial_population_times(params, nodes_IDs)
        
        
    print(f"Población inicial: {list(initial_population)}\n")
    sys.stdout.flush()
    
    # Registrar población inicial
    log_initial_population(initial_population)
    
    # Configurar e inicializar el algoritmo genético
    ga_instance = configure_genetic_algorithm(params, initial_population)
    
    ga_instance.nodes_IDs=params['nodes_IDs']
    ga_instance.normalized_target_signal=normalized_target_signal
    
    
    ga_instance.modules_path_openCarp=params['modules_path_openCarp']
    ga_instance.geom=params['geom']
    ga_instance.openCarp_simulationFiles=params['openCarp_simulationFiles']
    ga_instance.lead_file=params['lead_file']
    ga_instance.torso_file=params['torso_file']
    ga_instance.torso_file_um=params['torso_file_um']
    ga_instance.config_template=params['config_template']
    
    

    if ga_instance.optim_type=="sequence":
        if params['num_genes'] != len(nodes_IDs):
            print(f"El número de genes {params['num_genes']} indicado no coincide con el número de nodos de estimulación {len(nodes_IDs)} fijados en el archivo {params['nodes_IDs_path']}... fuerzo el cambio\n")
            sys.stdout.flush()
            params['num_genes'] = len(nodes_IDs)
    
    ga_instance.num_genes=params['num_genes']
    
    if ga_instance.optim_type=="sequence":
        ga_instance.set_values= set_values
    
    
    # Ejecutar algoritmo genético
    
    ga_instance.run()
    
    #guardar le modelo al finalizar
    save_ga_instance(ga_instance)
    
    # Obtener y mostrar resultados
    solution, solution_fitness, _ = ga_instance.best_solution()
    print(f"Mejor selección de valores: {solution}")
    print(f"Error mínimo encontrado: {solution_fitness}")
    sys.stdout.flush()
    
    
    
    # Guardar solución
    with open("best_solution.txt", 'w') as f:
        for num in solution:
            f.write(f"{num}\n")
    
    
    
    ###simulacion final
    eval_dir = os.path.join(params['current_dir'], f"best_solution")
    os.makedirs(eval_dir, exist_ok=True)
    
    manual_times = np.zeros_like(solution, dtype=float)
    
    args_dict = {
        "job_name": eval_dir,
        "geom": ga_instance.geom,
        "simulation_files": ga_instance.openCarp_simulationFiles,
        "duration": 120.,
        "model": "MitchellSchaeffer",
        "myocardial_CV": 570,
        "initial_time": 40,
        "sinusal_mode": "manual",
        "manual_ids": solution,
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
    
    except Exception as e:
            print(f"Error en la simulación: {str(e)}")
            os._exit(1) 
 
    
    # Procesar y visualizar resultados finales
    
    ECG_best_normalized = ecg_calcul_normalized(ecg_path)
    ECG_best = ecg_calcul(ecg_path)
    
    
    
    # Convertir Manager.list a lista normal y filtrar valores None
    all_signals = [x for x in list(ga_instance.optimization_state['all_simulated_signals_normalized']) if x is not None]
    
    
    graficar_best(all_signals, ECG_best_normalized, normalized_target_signal,solution_fitness, final='yes')
    
    
    # Guardar todos los resultados
    guardar_resultados_completos(
        params=params,
        opt_state=ga_instance.optimization_state,
        target=normalized_target_signal,
        ga_instance=ga_instance
    )
    
    
        
    
if __name__ == "__main__":
    main()