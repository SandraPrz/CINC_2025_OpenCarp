import os
import subprocess
import json
import sys
import time
import random
import numpy as np

class Parameters:
    def __init__(self):
        
        
        ############################ PARÁMETROS DE ENTRADA ############################
        # Generar una semilla aleatoria (usando time.time() + PID del proceso para mayor aleatoriedad)
        # Generar semilla dentro del rango permitido (0 a 2**32 - 1)
        self.SEED = 19
        random.seed(self.SEED)  # Semilla para 'random'
        np.random.seed(self.SEED)  # Semilla para 'numpy.random'
        self.random_seed = self.SEED
        print(f"SEED generada: {self.SEED}")  # ¡IMPORTANTE! Guarda este valor para reproducibilidad
        sys.stdout.flush()
        self.python_path = "/home/nodo38/sandra/virtual_env/GA_OpenCarp/bin/python3"
        
        
        
        
        # Obtener el directorio del script actual
        self.current_dir = os.path.dirname(os.path.abspath(__file__)) #gilette
        self.main_dir = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/improved_codes/"

        
        # Rutas de archivos
        self.geom = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/mesh/PSM1_1000_tetra_fibers_uvc.vtu" #gilette
        self.openCarp_simulationFiles = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/mesh/PSM1_1000_tetra_fibers_retag" #gilette
        self.lead_file = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/mesh/PSM1_electrodes_um_labeled.vtk" #gilette
        self.torso_file = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/mesh/PSM1_tetra_a.vtk" #gilette
        self.torso_file_um = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/mesh/PSM1_tetra_a_um.vtk" #gilette
        self.config_template = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/mesh/config_refneg.json" #gilette
        self.target_ecg_path = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/mesh/ID4_ECG_filtrado.csv" #gilette
        self.modules_path = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/improved_codes/modules/"
        self.modules_path_openCarp = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/improved_codes/modules"
        self.kd_tree_file="/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/mesh/kdtree.pkl" #gilette


        # Datos de la malla
        self.meshVolume = 186.8 #gilette
        


        # Tipo de optimizacion
        self.optim_type = "gilette_method" #nodes/sequence/gilette_method #gilette
       
        self.derivation_num = "total"  #gilette
        self.optimization_function = "L2_norm" #gilette
        self.num_cpu=5   #gilette  
        self.muestras_iniciales=128 #tiene que ser potencia de 2
        self.variables_sampler=13
        self.n_calls = self.muestras_iniciales * (self.variables_sampler + 2) 	#gilette N total =N×(D+2) siendo N las muestras iniciales y D el numero de parametros a optimizar(13)
        
        
        # caracteristicas simulacion
        self.OpenCarp_sim_duration=120. #gilette
        self.OpenCarp_sim_myo_vel=570 #gilette 
        self.OpenCarp_sim_initial_time=0 #gilette
        
        
        
        
        
        
        #Guardamos los parámetros en un archivo JSON
        self.save_parameters()

        
        
    def save_parameters(self):
        params = {
            'target_ecg_path': self.target_ecg_path,
            'modules_path': self.modules_path,
            'modules_path_openCarp': self.modules_path_openCarp,
            'meshVolume': self.meshVolume,
            'optim_type': self.optim_type,
            'derivation_num': self.derivation_num,
            'optimization_function': self.optimization_function,
            'n_calls': self.n_calls,
            'current_dir': self.current_dir,
            'geom' : self.geom,
            'openCarp_simulationFiles' : self.openCarp_simulationFiles,
            'lead_file' : self.lead_file,
            'torso_file_um' : self.torso_file_um,
            'torso_file' : self.torso_file,
            'config_template' : self.config_template,
            'num_cpu': self.num_cpu,
            'python_path': self.python_path,
            'random_seed': self.random_seed,
            'kd_tree_file': self.kd_tree_file,
            'OpenCarp_sim_duration':self.OpenCarp_sim_duration,
            'OpenCarp_sim_myo_vel':self.OpenCarp_sim_myo_vel,
            'OpenCarp_sim_initial_time':self.OpenCarp_sim_initial_time,
            'muestras_iniciales':self.muestras_iniciales,
            'variables_sampler':self.variables_sampler
        }

        # Guardar como archivo JSON
        self.params_file = os.path.join(self.current_dir, 'parameters.json')
        with open(self.params_file, 'w') as f:
            json.dump(params, f, indent=4)

        print(f"Parámetros guardados en {self.params_file}")  # Cambiado a self.params_file
        sys.stdout.flush()

if __name__ == "__main__":
    # Crear una instancia de Parameters para guardar los parámetros
    params = Parameters()

    # Ruta al script principal
    main_script = os.path.join(params.main_dir, "mainGilette_method.py")

    # Verificar que el script existe
    if not os.path.exists(main_script):
        print(f"Error: No se encontró el script principal en {main_script}")
        sys.stdout.flush()
        print("Asegurate de que la estructura de directorios es correcta:")
        sys.stdout.flush()
        print(f"Debe existir: {params.main_dir}/mainGilette_method.py")
        sys.stdout.flush()
        sys.exit(1)

    # Ejecutar el script principal
    print(f"\nEjecutando optimizacion con parametros guardados...")
    sys.stdout.flush()
    subprocess.run([params.python_path, main_script, params.params_file])
