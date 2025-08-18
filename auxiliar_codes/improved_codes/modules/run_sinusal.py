#!/usr/bin/env python

import os
import numpy as np
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

from stimOpts import stimBlock
from simulation import simulation_eik
from functions import (create_vtk_from_nodes, smart_reader, extract_surface, threshold, 
                        get_cobivecoaux, vtkWrite, addGlobalIds, farthest_point_sampling_global_ids,get_closest_global_ids, get_closest_global_ids_sorted_by_label,modify_json_entry,
                        )
from leadfield import (compute_leadfield, compute_ecg)

     
        
def parser():
    parser = tools.standard_parser()
    group  = parser.add_argument_group('experiment specific options')
    group.add_argument('--job-name',
                        type=str,
                        default="sinusal_optimizacion_jorge3",
                        help='Nombre del directorio de salida para los resultados')


    group.add_argument('--geom',
                        type = str,
                        default = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/mesh/PSM1_1000_tetra_fibers_uvc.vtu",
                        help = 'Path and name of the ventricular geometry')
    
    
    group.add_argument('--simulation-files',
                        type = str,
                        default = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/mesh/PSM1_1000_tetra_fibers_retag",
                        help = 'Path and name of the simulation mesh files (without extension)') 
    group.add_argument('--duration',
                        type = float,
                        default = 120.,
                        help = 'Duration of simulation in [ms] (default: 300.)')
    group.add_argument('--model',
                   type=str,
                   choices=['OHara', 'MitchellSchaeffer'],
                   default='MitchellSchaeffer',
                   help='Electrophysiological model to use (default: MitchellSchaeffer)')
    group.add_argument('--myocardial-CV',
                   type=float,
                   default=570,
                   help='Velocidad de conducción para el miocardio')  

                   
    group.add_argument('--sinusal-mode',
                   type=str,
                   choices=['manual', 'optimizado'],
                   default='manual',
                   help='Origen de la estimulación sinusal (manual u optimizado)')
    
    
    group.add_argument('--manual-ids',
                   type=str,
                   default=" 8312, 34513, 18299, 48043, 14543, 49157",
                   help='Lista de IDs para estimulación manual, separados por comas (ej: "123,456,789")')
    group.add_argument('--manual-times',
                   type=str,
                   default="6.3477,6.3477,6.3477,6.3477,6.3477,0.0000",
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
                   default=False,
                   help='Guardar los puntos de estimulación y los tiempos como un .vtk (True/False)')            
    group.add_argument('--output-vtk-name',
                    type=str,
                    default="",
                    help='Nombre del archivo de salida .vtu con estimulación sinusal')
    
    group.add_argument('--lead-file',
                        type = str,
                        default = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/mesh/PSM1_electrodes_um_labeled.vtk",
                        help = 'Path and name of labeled .vtk lead file [V1,V2,V3,V4,V5,V6,LA,RA,LL]') 
    group.add_argument('--torso-file',
                        type = str,
                        default = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/mesh/PSM1_tetra_a_um.vtk",
                        help = 'Path and name of the .vtk torso mesh file') 
    
    group.add_argument('--config-template',
                        type=str,
                        default='/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/mesh/config.json',
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
def run(args, job):
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
            raise ValueError("En modo manual debes proporcionar '--manual-ids' y '--manual-times'")

        try:
            ids_sinusal_manual = [int(x) for x in args.manual_ids.split(',')]
            tiempos_sinusal_manual = [float(x) for x in args.manual_times.split(',')]
        except Exception as e:
            raise ValueError(f"Error al parsear '--manual-ids' o '--manual-times': {e}")

        if len(ids_sinusal_manual) != len(tiempos_sinusal_manual):
            raise ValueError("El número de IDs debe coincidir con el número de tiempos")

    
        global_ids_selected = np.array(ids_sinusal_manual)
        original_stimTimes_sinusal = np.array(tiempos_sinusal_manual)
        
        original_stimTimes_sinusal2 = [x - min(original_stimTimes_sinusal) for x in original_stimTimes_sinusal] #start de sinusal stimulation at 0ms
        

    else:
    
    
        #########################
        ## FROM OPTIMIZATION MODE
        #########################
        
        #load optimization result and precomputed kdtree
        opt_path = args.opt_file
        
        if not os.path.exists(args.opt_file):
                raise FileNotFoundError(f"Archivo de optimización no encontrado: {args.opt_file}")

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
    torso_mesh = smart_reader(args.torso_file)

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



if __name__ == "__main__":
    run()





#Stimulation points definition by globalIds and times
        #ELVIRA file (pk nodes IDs)--> optimización GA 19 nodos2 + secuencia: (/gpfs/projects/upv100/WORK-SANDRA/pruebas_autoSim/real_ecg_ID6/PRUEBAS_FINAL/sequence_bayesian/seed_19_newpop/model)
        #3903805 13
        #3908967 14
        #3782938 16
        #3918392 4
        #3813204 21
        #3894094 8
        #3810770 20
        
#ids_sinusal_manual  =[29022,1718,37641,15468,20371,8790,8854] #values from globalIds label from endocardial threshold PRV LV SLV RV PLV PLV SLV [29022,1718,37641,15468,20371,8790,8854]
#tiempos_sinusal_manual =[13,14,16,4,21,8,20]
 