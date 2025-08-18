# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:51:59 2025

@author: Sandra
"""
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from modules.results_handler import load_manager_state_local


##################################################################################################################
#########################################################################################
###############################################################
#####################################


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
    # Ordenar los nodos por su distancia al punto (en caso de empate, se selecciona el primero por orden de aparición)
    nodos_ordenados = sorted(nodos, key=lambda nodo: np.linalg.norm(nodo[1] - punto))
    
    # Retorna el ID del nodo más cercano (el primero en la lista ordenada)
    return nodos_ordenados[0][0]


def procesar_archivos(vtk_file, nodos_file,write=None):
    """Carga los puntos, encuentra los nodos más cercanos y guarda los resultados."""
    total_elvira_node_IDs=[]
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



    


def from_ID_to_vtk(ids_input_path, nodos_file, output_vtk_file):
    """
    Dado un conjunto de IDs, obtiene las coordenadas de los nodos correspondientes 
    y genera un archivo VTK con esas coordenadas como Unstructured Grid.
    
    ids_input: Lista de IDs de nodos a extraer.
    nodos_file: Archivo de nodos de entrada.
    output_vtk_file: Ruta del archivo VTK de salida.
    """
    # Cargar nodos desde el archivo
    nodos = cargar_nodos(nodos_file)
    
    ids_input = np.loadtxt(ids_input_path, dtype=int)  # Cargar listado de puntos desde un archivo de texto

    # Crear un diccionario para acceso rápido a las coordenadas de los nodos por ID
    nodos_dict = {nodo_id: coordenadas for nodo_id, coordenadas in nodos}
    
    # Filtrar las coordenadas para los IDs de entrada
    puntos_filtrados = []
    stimulus_ids = []  # Para almacenar los IDs de estímulo correspondientes
    for nodo_id in ids_input:
        if nodo_id in nodos_dict:
            puntos_filtrados.append(nodos_dict[nodo_id])
            stimulus_ids.append(nodo_id)  # El ID del nodo también será el STIMULUS_ID
    
    puntos_filtrados = np.array(puntos_filtrados)
    
    # Crear un objeto UnstructuredGrid para guardar las coordenadas filtradas
    unstructured_grid = vtk.vtkUnstructuredGrid()
    
    # Crear un objeto vtkPoints para agregar los puntos al UnstructuredGrid
    points_vtk = vtk.vtkPoints()
    
    # Añadir los puntos filtrados
    for punto in puntos_filtrados:
        points_vtk.InsertNextPoint(punto)
    
    # Asignar los puntos al UnstructuredGrid
    unstructured_grid.SetPoints(points_vtk)
    
    # Añadir celdas de tipo punto (cada punto será representado como una celda de tipo VTK_VERTEX)
    for i in range(len(puntos_filtrados)):
        unstructured_grid.InsertNextCell(vtk.VTK_VERTEX, 1, [i])  # Cada celda es un punto
    
    # Crear un vtkIntArray para el atributo STIMULUS_ID
    stimulus_array = vtk.vtkIntArray()
    stimulus_array.SetName("STIMULUS_ID")
    
    # Añadir el STIMULUS_ID a cada punto
    for stim_id in stimulus_ids:
        stimulus_array.InsertNextValue(stim_id)
    
    # Asignar el atributo STIMULUS_ID al UnstructuredGrid
    unstructured_grid.GetPointData().AddArray(stimulus_array)
    
    # Crear un escritor VTK para guardar los datos en un archivo .vtk
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(output_vtk_file)
    writer.SetInputData(unstructured_grid)
    writer.Write()
    
    print(f"Archivo VTK generado correctamente en: {output_vtk_file}")



def from_points_to_pk(ids_input_path_sinpk, nodos_file_sinpk, vtk_file_sinpk, nodos_file_pk,write_pk_nodes):
    """
    Esta función sirve para una vez tengo los puntos de activación que se han optimizado, si quiero sacar un purkinje a partir de esta simulacion
    que no cubra todo el endocardio, saber cuáles son los nuevos nodos que tengo que estimular.
    
    ids_input_path_sinpk=ruta al elvira_nodes_IDs.txt que se le pasa al GA para que cree la población inicial
    nodos_file_sinpk=archivo de NODES.dat de la geometría usada durante la optimización del GA
    
    vtk_file_sinpk=ruta que guarda los puntos correspondientes a los IDs que se buscan en la malla con purkinje completo original
    nodos_file_pk=nuevo fichero NODES.dat que sacamos montando el purkinje con el ventrículo y los PMJs que usamos cuando teniamos el purkinje completo
    write_pk_nodes=ruta a un nuevo fichero que nos diga cuales son los IDs de los nuevos nodos a estimular
    
    """
    
    from_ID_to_vtk(ids_input_path_sinpk, nodos_file_sinpk, vtk_file_sinpk) #pasamos del resultado del GA a un .vtk con los puntos de estimulación en el espacio
    procesar_archivos(vtk_file_sinpk, nodos_file_pk,write_pk_nodes) #buscamos el ID qe cada uno de esos puntos tiene en el archivo de NODES de la malla con el purinje parcial y escribimos un fichero con los nuevos nodos a estimular.

    

def from_ID_to_vtk_result(genetation_num, optim_type,optimization_state_path, nodos_file, output_vtk_file):
    """
    Dado un conjunto de IDs, obtiene las coordenadas de los nodos correspondientes 
    y genera un archivo VTK con esas coordenadas como Unstructured Grid.
    
    ids_input: Lista de IDs de nodos a extraer.
    nodos_file: Archivo de nodos de entrada.
    output_vtk_file: Ruta del archivo VTK de salida.
    """
    optimization_state=load_manager_state_local(optimization_state_path)
    
    
    
    
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
    
    if optim_type=="sequence":
        activation_times_array = vtk.vtkFloatArray()
        activation_times_array.SetName("activation_times")
    elif optim_type=="nodes":
        nodeID_array = vtk.vtkIntArray()
        nodeID_array.SetName("NODES_ID")
        
        
        
    for gen in optimization_state['simulation_results'].keys(): 
        
        for idx in optimization_state['simulation_results'][gen].keys():
            
            error = simulation_results[gen][idx]['error']
            
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
                    nodeID_array.InsertNextValue(int(nodo_id))
                    idx_array.InsertNextValue(idx)
                    gen_array.InsertNextValue(gen)
                    error_array.InsertNextValue(error)
                    
                    if optim_type=="sequence":
                        # Encontrar el índice de nodo_id en ids_input usando np.where
                        node_index = np.where(ids_input == nodo_id)[0][0]  # Devuelve el primer índice donde coincida
                        activation_times_array.InsertNextValue(float(activation_times_input[node_index]))
                        
                        
                    # Añadir celda de tipo VTK_VERTEX
                    unstructured_grid.InsertNextCell(vtk.VTK_VERTEX, 1, [vtk_index])
            
    # Asignar los puntos y atributos al UnstructuredGrid
    unstructured_grid.SetPoints(points_vtk)
    unstructured_grid.GetPointData().AddArray(nodeID_array)
    unstructured_grid.GetPointData().AddArray(idx_array)
    unstructured_grid.GetPointData().AddArray(gen_array)
    unstructured_grid.GetPointData().AddArray(error_array)     
    if optim_type=="sequence":
        unstructured_grid.GetPointData().AddArray(activation_times_array) 
    
    
    # Guardar en archivo VTK
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(output_vtk_file)
    writer.SetInputData(unstructured_grid)
    writer.Write()
        
    print(f"Archivo VTK generado correctamente en: {output_vtk_file}")
    

