import os
import subprocess
import json
import sys
import time
import random
import numpy as np

class Parameters:
    def __init__(self):
        
        # Generar una semilla aleatoria (usando time.time() + PID del proceso para mayor aleatoriedad)
        # Generar semilla dentro del rango permitido (0 a 2**32 - 1)
        self.SEED = 19
        random.seed(self.SEED)  # Semilla para 'random'
        np.random.seed(self.SEED)  # Semilla para 'numpy.random'
        
        print(f"SEED generada: {self.SEED}")  # ¡IMPORTANTE! Guarda este valor para reproducibilidad
        sys.stdout.flush()

        
        
        # Obtener el directorio del script actual
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.main_dir = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/improved_codes/"

        ############################ PARÁMETROS DE ENTRADA ############################
        # Rutas de archivos
        self.geom = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/mesh/prueba2.vtu"
        self.openCarp_simulationFiles = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/mesh/PSM1_1000_tetra_fibers_retag"
        self.lead_file = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/mesh/PSM1_electrodes_um_labeled.vtk"
        self.torso_file = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/mesh/PSM1_tetra_a.vtk"
        self.torso_file_um = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/mesh/PSM1_tetra_a_um.vtk"
        self.config_template = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/mesh/config_refneg.json"
        self.target_ecg_path = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/mesh/ID6_ECG_filtrado.csv"
        self.modules_path = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/improved_codes/modules/"
        self.modules_path_openCarp = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/improved_codes/modules"
        self.population_path="/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/results_population/results_population/poblacion_final.npy"
        
	# Datos de la malla
        self.meshVolume = 186.8 #open... tengo dudas de si hay que cambiarle las unidades
        self.last_seq = 21

        # Nodos de estimulacion
        self.nodes_IDs_path = "/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/CINC_2025_OpenCarp/auxiliar_codes/results_population/results_population/unique_aha_values.txt" #open

        # Tipo de optimizacion
        self.optim_type = "nodes"
        self.derivation_num = "total"
        self.optimization_function = "dtw_camps"  #"dtw_camps"
        self.num_cpu = 5                                                                                                                                                                                               
        
        # Parametros del algoritmo genetico
        self.num_generations = 24 # Total: 20x12=240 eval
        self.num_parents_mating = 5
        self.sol_per_pop = 23 # Multiplo de evaluaciones_por_bloque
        self.num_genes = 7 #si la optimizacion es tipo "nodes" este valor será el número de nodos de estimulación que queremos aplicar, si es tipo "sequence" este valor será el número de nodos que introducimos en nodes_IDs
        self.parent_selection_type = "tournament" # Mayor presion selectiva
        
        self.keep_elitism = 3 # Elitismo (15% poblacion)
        self.crossover_type = "uniform"  # Mejor mezcla genetica
        self.mutation_probability = 0.20 # Exploracion inicial alta
        self.random_seed = self.SEED
        self.python_path = "/home/nodo38/sandra/virtual_env/GA_OpenCarp/bin/python3"
        self.stop_criteria = "saturate_5" 
        
        
    
    
    
    
        # Guardamos los parámetros en un archivo JSON
        self.save_parameters()

    def save_parameters(self):
        params = {
            'target_ecg_path': self.target_ecg_path,
            'modules_path': self.modules_path,
            'modules_path_openCarp': self.modules_path_openCarp,
            'population_path': self.population_path,
            'meshVolume': self.meshVolume,
            'last_seq': self.last_seq,
            'nodes_IDs_path': self.nodes_IDs_path,
            'optim_type': self.optim_type,
            'derivation_num': self.derivation_num,
            'optimization_function': self.optimization_function,
            'num_generations': self.num_generations,
            'num_parents_mating': self.num_parents_mating,
            'sol_per_pop': self.sol_per_pop,
            'num_genes': self.num_genes,
            'parent_selection_type': self.parent_selection_type,
            'keep_elitism': self.keep_elitism,
            'crossover_type': self.crossover_type,
            'mutation_probability': self.mutation_probability,
            'random_seed': self.random_seed,
            'current_dir': self.current_dir,
            'geom' : self.geom,
            'openCarp_simulationFiles' : self.openCarp_simulationFiles,
            'lead_file' : self.lead_file,
            'torso_file_um' : self.torso_file_um,
            'torso_file' : self.torso_file,
            'config_template' : self.config_template,
            'num_cpu' :  self.num_cpu,
            'stop_criteria' : self.stop_criteria
        }

        # Guardar como archivo JSON
        self.params_file = os.path.join(self.current_dir, 'parameters.json')
        with open(self.params_file, 'w') as f:
            json.dump(params, f, indent=4)

        print(f"Parametros guardados en {self.params_file}")  # Cambiado a self.params_file
        sys.stdout.flush()

if __name__ == "__main__":
    # Crear una instancia de Parameters para guardar los parametros
    params = Parameters()

    # Ruta al script principal
    main_script = os.path.join(params.main_dir, "main.py")

    # Verificar que el script existe
    if not os.path.exists(main_script):
        print(f"Error: No se encontro el script principal en {main_script}")
        sys.stdout.flush()
        print("Asegurate de que la estructura de directorios es correcta:")
        sys.stdout.flush()
        print(f"Debe existir: {params.main_dir}/main.py")
        sys.stdout.flush()
        sys.exit(1)

    # Ejecutar el script principal
    print(f"\nEjecutando optimizacion con parametros guardados...")
    sys.stdout.flush()
    subprocess.run([params.python_path, main_script, params.params_file])
