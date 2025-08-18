import os
import subprocess
import sys
import time
import shutil
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

from modules.stimOpts import stimBlock
from modules.simulation import simulation_eik
from modules.functions import (create_vtk_from_nodes, smart_reader, extract_surface, threshold, 
                        get_cobivecoaux, vtkWrite, addGlobalIds, farthest_point_sampling_global_ids,get_closest_global_ids, get_closest_global_ids_sorted_by_label,modify_json_entry,
                        )
from modules.leadfield import (compute_leadfield, compute_ecg)


   
######## GESTIÓN DE ARCCHIVOS DE SIMULACIÓN

def modify_stimulation_nodes(file_path, new_values, optim_type, nodes_IDs):
    """
    Modifica el archivo de estímulo con nuevos valores de nodos o secuencias temporales.
    
    Parámetros:
    file_path : str - Ruta al archivo de estímulo
    new_values : array - Nuevos valores (nodos o secuencias)
    optim_type : str - Tipo de optimización ('nodes' o 'sequence')
    """
    if optim_type == "nodes":
        # Leer el archivo actual
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Encontrar la línea donde se indica el número total de nodos
        for i, line in enumerate(lines):
            if line.strip().endswith("! Total number of nodes to be stimulated"):
                node_count_index = i
                break
        else:
            raise ValueError("No se encontró la sección de nodos a estimular en el archivo.")
        
        # Modificar la cantidad de nodos
        lines[node_count_index] = f"{len(new_values)} 0 \t ! Total number of nodes to be stimulated\n"
        
        # Crear líneas para los nuevos nodos (cada nodo con tiempo 1)
        new_node_lines = [f"{node} 1\n" for node in new_values]
        
        # Insertar los nuevos nodos reemplazando los antiguos
        lines = lines[:node_count_index + 1] + new_node_lines
        
        # Sobrescribir el archivo con los nuevos datos
        with open(file_path, 'w') as f:
            f.writelines(lines)
        
        print(f'Archivo de stimulus modificado con {new_values}')
        sys.stdout.flush()  # Asegura que se imprima inmediatamente
        
    elif optim_type == "sequence":
        # Leer el archivo actual
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Encontrar la línea donde se indica el número total de nodos
        for i, line in enumerate(lines):
            if line.strip().endswith("! Total number of nodes to be stimulated"):
                node_count_index = i
                break
        else:
            raise ValueError("No se encontró la sección de nodos a estimular en el archivo.")
        
        # Modificar la cantidad de nodos
        lines[node_count_index] = f"{len(nodes_IDs)} 0 \t ! Total number of nodes to be stimulated\n"
        
        # Crear líneas para los nuevos nodos con sus tiempos de secuencia
        new_node_lines = [f"{node} {time_seq}\n" for node, time_seq in zip(nodes_IDs, new_values)]
        
        # Insertar los nuevos nodos reemplazando los antiguos
        lines = lines[:node_count_index + 1] + new_node_lines
        
        # Sobrescribir el archivo con los nuevos datos
        with open(file_path, 'w') as f:
            f.writelines(lines)
        
        print(f'Archivo de stimulus modificado con {new_values}')
        sys.stdout.flush()  # Asegura que se imprima inmediatamente
 
 


def simulacion(args, job):
    
    filenameorignal=job.ID          
    
    # Geometry reading
    vol_mesh = smart_reader(args["geom"])
    vol_mesh = addGlobalIds(vol_mesh)
    
    #save GlobalIds complete mesh
    #vtkWrite(vol_mesh,'./meshes/vol_mesh_globaIds.vtu')
    
    
    
    #####################################################
    ## SINUSAL
    #####################################################
    
    if args["sinusal_mode"] == "manual":
        
     
        ###################
        ## MANUAL MODE
        ###################
                
        if args["manual_ids"] is None or args["manual_times"] is None:
            raise ValueError("En modo manual debes proporcionar 'manual-ids' y 'manual-times'")

        try:
            ids_sinusal_manual = args["manual_ids"]#[int(x) for x in args["manual_ids"].split(',')]
            tiempos_sinusal_manual = args["manual_times"]#[float(x) for x in args["manual_times"].split(',')]
        except Exception as e:
            raise ValueError(f"Error al parsear '--manual-ids' o '--manual-times': {e}")

        if len(ids_sinusal_manual) != len(tiempos_sinusal_manual):
            raise ValueError("El número de IDs debe coincidir con el número de tiempos")

    
        global_ids_selected = np.array(ids_sinusal_manual)
        original_stimTimes_sinusal = np.array(tiempos_sinusal_manual)
        
        original_stimTimes_sinusal2 = [x - min(original_stimTimes_sinusal)+ args["initial_time"] for x in original_stimTimes_sinusal]  #start de sinusal stimulation at 0ms
        

    else:
    
    
        #########################
        ## FROM OPTIMIZATION MODE
        #########################
        
        #load optimization result and precomputed kdtree
        opt_path = args["opt_file"]
        
        if not os.path.exists(args["opt_file"]):
                raise FileNotFoundError(f"Archivo de optimización no encontrado: {args['opt_file']}")


        best_params_reshape = np.load(opt_path)["best_solution_6d.npy"] # ab tm rt tv ts time   
        
        
        if not os.path.exists(args["kdtree_file"]):
            # Crear arbol de ucoords
            ids = numpy_support.vtk_to_numpy(vol_mesh.GetPointData().GetArray('GlobalIds'))
            ucoords = np.column_stack(get_cobivecoaux(vol_mesh, ids))
            tree = cKDTree(ucoords)
                   
            # Guardar KDTree
            with open(args["kdtree_file"], 'wb') as f:
                pickle.dump(tree, f)
    
            print(f"KDTree guardado exitosamente en {args['kdtree_file']}")
                
        else:                
            with open(args["kdtree_file"], 'rb') as f:
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
    if str(args["save_vtk"]).lower() == "true":
            #save stimulus input for points and times
            vtk_output_path = os.path.join(job.ID, args["output_vtk_name"])

            #save GlobalIds complete mesh and endo to check
            #surf_mesh = extract_surface(vol_mesh)
            #endo = threshold(surf_mesh, 3, 4, 'interp_class')
            #vtkWrite(vol_mesh,'./vol_mesh_globaIds.vtu')         
            #vtkWrite(endo,'./endo_globaIds.vtu')
        
            if args["sinusal_mode"] == "manual":
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
    leads = smart_reader(args["lead_file"])
    torso_mesh = smart_reader(args["torso_file_um"])

    lead_ids = get_closest_global_ids_sorted_by_label(leads, torso_mesh) 
    
    config_template_path = args["config_template"]

    unique_config_path = os.path.join(filenameorignal, 'config.json')
    shutil.copy(config_template_path, unique_config_path)
    
    
    modify_json_entry( file_path=unique_config_path, key="VTK_FILE", new_value=args["torso_file"] )
    modify_json_entry( file_path=unique_config_path, key="MEASUREMENT_LEADS", new_value=lead_ids )
    modify_json_entry(file_path=unique_config_path, key="VM_FILE", new_value=os.path.join(job.ID, "vm.dat"))
    modify_json_entry(file_path=unique_config_path, key="ECG_FILE", new_value=os.path.join(job.ID, "ecg_output.dat"))



    C_matrix_data, heart_node_indices, num_total_nodes, _ = compute_leadfield(unique_config_path)
    compute_ecg(C_matrix_data, heart_node_indices, num_total_nodes, unique_config_path)
    
    
    
    print(f"Output folder: {job.ID}")
       
       
       
       
    
def simulacion_final(selected_values,output_path):
    """
    Lanza la simulación final con los mejores parámetros encontrados.
    
    Parámetros:
    selected_values : array - Valores seleccionados para la simulación final
    """
    try:
        # Ejecutar sbatch y obtener el Job ID
        resultado = subprocess.run(["sbatch", "run_sim_final.sh"], capture_output=True, text=True, check=True)
        output = resultado.stdout.strip()

        # Extraer el Job ID de la salida de sbatch
        job_id = None
        if "Submitted batch job" in output:
            job_id = output.split()[-1]

        if not job_id:
            print("No se pudo obtener el Job ID.")
            sys.stdout.flush()  # Asegura que se imprima inmediatamente
            return None

        print(f"Job enviado con éxito: {job_id}. Esperando a que termine...")
        sys.stdout.flush()  # Asegura que se imprima inmediatamente

        # Esperar hasta que el job termine
        while True:
            check_result = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
            if job_id not in check_result.stdout:
                break  # Sale del bucle cuando el job desaparece de la cola

            time.sleep(30)  # Esperar antes de volver a comprobar

        print(f"Job {job_id} ha terminado.")
        sys.stdout.flush()  # Asegura que se imprima inmediatamente
        time.sleep(30)  # Espera adicional para asegurar que los archivos estén disponibles
        
        
        
        # Intentar leer el archivo eccg_aiso.dat
        
        if not os.path.exists(os.path.join(output_path,"post_S2","ens", "ecg_aiso.dat")):
            print(f"El archivo ecg_aiso.dat no se encuentra. Limpiando directorio y reintentando...")
            sys.stdout.flush()  # Asegura que se imprima inmediatamente

            # Limpiar el directorio y volver a ejecutar el bloque dentro del try
            limpiar_directorio()
            simulacion_final(selected_values,output_path)  # Llamar nuevamente al mismo proceso para intentar otra vez
            return  # Terminar la ejecución actual para evitar múltiples ejecuciones del proceso en paralelo.

        print(f"El archivo ecg_aiso.dat ha sido encontrado. Procesando...")
        sys.stdout.flush()  # Asegura que se imprima inmediatamente
        # Aquí continúa el código que debe procesar el archivo cuando esté disponible


    except subprocess.CalledProcessError as e:
        print("Error al ejecutar sbatch:", e.stderr)
        sys.stdout.flush()  # Asegura que se imprima inmediatamente        





def limpiar_directorio():
    """
    Elimina archivos temporales del directorio actual, excepto los importantes
    como archivos de modelo, scripts, resultados, etc.
    No elimina ningún directorio.
    """
    directorio_actual = os.getcwd()
    script_actual = os.path.basename(__file__)  # Nombre del script actual

    for item in os.listdir(directorio_actual):
        item_path = os.path.join(directorio_actual, item)

        # No borrar si es el script actual, archivos importantes o carpetas específicas
        if (
            item == "model" or 
            item == "post_S2" or 
            item == script_actual or 
            item.endswith((".json", ".sh", ".txt", ".err", ".out", ".png", ".vtk", ".csv", ".pkl", ".py"))
        ):
            continue

        # Solo eliminar archivos, nunca directorios
        if os.path.isfile(item_path):
            os.remove(item_path)

            




