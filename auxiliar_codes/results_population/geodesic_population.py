import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import os
from multiprocessing import Pool, cpu_count
import itertools
import sys
import time
from pathlib import Path
import pyvista as pv


output_dir= './results_population'

# Configurar semilla para reproducibilidad
SEMILLA = 42  # Puedes usar cualquier número entero
np.random.seed(SEMILLA)



# ----------------------------
# 1. Cargar malla y nodos candidatos
# ----------------------------

def addGlobalIds(input_vtk):
    """
    Adds GlobalIds to points of a vtkPolyData using vtkIdFilter.
    
    Parameters:
        polydata (vtk.vtkPolyData): The input mesh.
    
    Returns:
        vtk.vtkPolyData: New mesh with point GlobalIds.
    """
    id_filter = vtk.vtkIdFilter()
    id_filter.SetInputData(input_vtk)
    id_filter.PointIdsOn()
    id_filter.CellIdsOff()
    id_filter.SetPointIdsArrayName("GlobalIds")  # <-- Correct method
    id_filter.Update()
    
    return id_filter.GetOutput()



def extract_surface(mesh):
    """! Extract surface of a mesh
     @param mesh The vtk mesh object
     @return surface The vtk surface mesh
    """
    surf_filter = vtk.vtkDataSetSurfaceFilter()
    surf_filter.SetInputData(mesh)
    surf_filter.Update()

    geomfilt = vtk.vtkGeometryFilter()
    geomfilt.SetInputData(surf_filter.GetOutput())
    geomfilt.Update()
    pdata = geomfilt.GetOutput()

    return pdata


def threshold(mesh, low, high, point_data):
    """! Threshold a mesh.

    This function creates a new vtk mesh object that contais only 
    the points whithin the range between de low and high threshold.

    The vtk library is needed.

    @param mesh The mesh to threshold. It must be a vtk object with point data.
    @param low Lower threshold. The new mesh will only contain.
                points with the attribute to study higher than the low threshold.
    @param high Higher threshold. The new mesh will only contain points with the 
                attribute to study lower than the low threshold.
    @param point_data Attribute to study in every point of the mesh to decide if 
                the new mesh will contain that point.

    @return  The new vtk mesh object.
    """
    thresh = vtk.vtkThreshold()
    thresh.SetInputData(mesh)
    thresh.SetLowerThreshold(low)
    thresh.SetUpperThreshold(high)
    thresh.SetInputArrayToProcess(
        0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", point_data)
    thresh.Update()

    return thresh.GetOutput()


def vtkWrite(data, filename):
    """
    Writes a VTK or VTU file from a vtkDataSet object.

    Parameters:
        data (vtk.vtkDataSet): The VTK data object to be written.
        filename (str): The output filename (should end in .vtk or .vtu).
    """
    if filename.endswith(".vtu"):
        writer = vtk.vtkXMLUnstructuredGridWriter()
    elif filename.endswith(".vtk"):
        writer = vtk.vtkUnstructuredGridWriter()
    else:
        raise ValueError("Unsupported file extension. Use .vtk or .vtu")

    writer.SetFileName(filename)

    # VTK writers often need casting
    if isinstance(data, vtk.vtkUnstructuredGrid):
        writer.SetInputData(data)
    else:
        raise TypeError("Only vtkUnstructuredGrid is supported in this function.")

    if writer.Write() == 1:
        print(f"File written successfully to {filename}")
    else:
        print("Failed to write the file.")

        
def smart_reader(path):

    """! Reads a mesh file in .vtk, .obj, .stl, .ply, .vtp or .vtu format.

    This function reads a vtk file for the structures.

    The vtk and pathlib libraries are needed.

    @param filepath The path to the file.

    @return  The mesh object.
    """
    valid_suffixes = ['.obj', '.stl', '.ply', '.vtk', '.vtp', '.vtu']
    path = Path(path)
    if path.suffix:
        ext = path.suffix.lower()
    if path.suffix not in valid_suffixes:
        print(f'No reader for this file suffix: {ext}')
        return None
    else:
        if ext == ".ply":
            reader = vtk.vtkPLYReader()
            reader.SetFileName(path)
            reader.Update()
            poly_data = reader.GetOutput()
        elif ext == ".vtp":
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(path)
            reader.Update()
            poly_data = reader.GetOutput()
        elif ext == ".obj":
            reader = vtk.vtkOBJReader()
            reader.SetFileName(path)
            reader.Update()
            poly_data = reader.GetOutput()
        elif ext == ".stl":
            reader = vtk.vtkSTLReader()
            reader.SetFileName(path)
            reader.Update()
            poly_data = reader.GetOutput()
        elif ext == ".vtk":
            data_checker = vtk.vtkDataSetReader()
            data_checker.SetFileName(path)
            data_checker.Update()
            if data_checker.IsFilePolyData():
                reader = vtk.vtkPolyDataReader()
            elif data_checker.IsFileUnstructuredGrid():
                reader = vtk.vtkUnstructuredGridReader()
            else:
                return "No polydata or unstructured grid"
            reader.SetFileName(path)
            reader.Update()
            poly_data = reader.GetOutput()
        elif ext == ".vtu":
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(path)
            reader.Update()
            poly_data = reader.GetOutput()

        return poly_data


def get_aha_centroids_and_global_ids(endo):
    
    # Verificaciones
    has_AHA = endo.GetPointData().HasArray("parcellation")
    if not has_AHA:
        raise ValueError("La malla no contiene 'parcellation'. Asegúrate de haber etiquetado los AHA")

    
    has_global_ids = endo.GetPointData().HasArray("GlobalIds")
    if not has_global_ids:
        raise ValueError("La malla no contiene 'GlobalIds'. Asegúrate de haber llamado a `addGlobalIds`.")


    aha_array = endo.GetPointData().GetArray("parcellation")
    aha_labels = np.unique([aha_array.GetValue(i) for i in range(aha_array.GetNumberOfTuples())])
    points = endo.GetPoints()
    global_ids = endo.GetPointData().GetArray("GlobalIds")  # o "GlobalIds" según cómo se haya guardado


    vi_results=[]
    vi_points=[]
    vi_ids=[]
    vi_aha=[]
    
    vd_results=[]
    vd_points=[]
    vd_ids=[]
    vd_aha=[]
    
    for label in aha_labels:
        # Índices de todos los puntos con esta etiqueta
        idx = [i for i in range(aha_array.GetNumberOfTuples()) if aha_array.GetValue(i) == label]
        if not idx:
            continue
    
        # Obtener coordenadas de los puntos de este AHA
        coords = np.array([points.GetPoint(i) for i in idx])
        
        # Calcular centroide
        centroid = np.mean(coords, axis=0)
        
        # Calcular distancias al centroide
        distances = np.linalg.norm(coords - centroid, axis=1)
    
        # Encontrar el índice del punto más cercano al centroide
        closest_local_idx = np.argmin(distances)
        closest_global_idx = idx[closest_local_idx]
        
        
        # Obtener GlobalId y punto más cercano
        global_id_centroid_endo = global_ids.GetValue(closest_global_idx)

        centroid_endo_point = endo.GetPoint(closest_global_idx) 
        
   

        result = {
            'AHA_segment': int(label),
            'centroid_points': centroid_endo_point,
            'closest_node_id': int(global_id_centroid_endo),  
        }


        if 0 < label <= 17:
            vi_results.append(result)
            vi_points.append(centroid_endo_point)
            vi_ids.append(global_id_centroid_endo)
            vi_aha.append(label)
        elif 17 < label:
            vd_results.append(result)
            vd_points.append(centroid_endo_point)
            vd_ids.append(global_id_centroid_endo)
            vd_aha.append(label)


    vi_poly = pv.PolyData(np.array(vi_points))
    vi_poly.point_data['GlobalIds'] = np.array(vi_ids)
    vi_poly.point_data['AHA_segment'] = np.array(vi_aha)
    filename = os.path.join(output_dir, "root_candidates_vi.vtp")
    vi_poly.save(filename)

    vd_poly = pv.PolyData(np.array(vd_points))
    vd_poly.point_data['GlobalIds'] = np.array(vd_ids)
    vd_poly.point_data['AHA_segment'] = np.array(vd_aha)
    filename = os.path.join(output_dir, "root_candidates_vd.vtp")
    vd_poly.save(filename)

    return vi_results, vd_results


'''
def cargar_puntos_vtk(vtk_file):
    """Carga archivos .vtk como PolyData y devuelve puntos como numpy array."""
    # Leer exclusivamente como PolyData
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file)
    reader.Update()
    
    polydata = reader.GetOutput()
    if polydata is None:
        raise ValueError("No se pudo leer el archivo como PolyData.")
    
    points = polydata.GetPoints()
    if points is None:
        raise ValueError("El archivo .vtk no contiene puntos válidos.")
    
    return vtk_to_numpy(points.GetData())



def cargar_malla_completa(vtk_file):
    """Carga la malla como vtkPolyData (solo lectura directa)."""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file)
    reader.Update()
    
    polydata = reader.GetOutput()
    if polydata is None:
        raise ValueError("No se pudo leer el archivo como PolyData.")
    
    if polydata.GetNumberOfPoints() == 0:
        raise ValueError("La malla cargada no contiene puntos.")
    
    return polydata
 
'''  



# ----------------------------
# 2. Cálculo de distancias geodésicas
# ----------------------------

def build_global_id_map(mesh):
    #esta función lo que pretende es mapear los globalIdsque tiene la malla del endocardio a los nuevos ids de esta submalla para que la función dijkstra funcione
    global_ids = mesh.GetPointData().GetArray("GlobalIds")
    if global_ids is None:
        raise RuntimeError("La malla no tiene GlobalIds.")

    id_map = {}
    for i in range(mesh.GetNumberOfPoints()):
        gid = int(global_ids.GetValue(i))
        id_map[gid] = i
    return id_map
    
def calcular_distancia_geodesica(args):
    """Función para multiprocessing con PolyData."""
    malla_poly, closest_points, i, j = args
    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(malla_poly)
    dijkstra.SetStartVertex(closest_points[i])
    dijkstra.SetEndVertex(closest_points[j])
    dijkstra.Update()
    
    if dijkstra.GetOutput().GetNumberOfPoints() == 0:
        print(f"Warning: No se pudo calcular distancia entre {i}- {closest_points[i]} y {j} - {closest_points[j]}")
        return (i, j, np.inf)
    
    return (i, j, dijkstra.GetOutput().GetLength())

def calcular_matriz_geodesica_paralelo(malla_poly, nodos, n_procesos=None):
    """Versión optimizada para PolyData."""
    #convetir a polydata
    convertidor = vtk.vtkDataSetSurfaceFilter()
    convertidor.SetInputData(malla_poly)
    convertidor.Update()
    malla_poly = convertidor.GetOutput()
    
    
    n = len(nodos)
    dist_matrix = np.zeros((n, n))
    
    id_map = build_global_id_map(malla_poly)
    local_indices = [id_map[gid] for gid in nodos]

    
    pares = [(malla_poly, local_indices, i, j) for i, j in itertools.combinations(range(n), 2)]
    total_pares = len(pares)
    
    if n_procesos is None:
        n_procesos = min(cpu_count(), 112)
    
    print(f"Calculando {total_pares} pares de distancias...")
    start_time = time.time()
    
    resultados = []
    with Pool(processes=n_procesos) as pool:
        for i, result in enumerate(pool.imap(calcular_distancia_geodesica, pares)):
            resultados.append(result)
            if i % 100 == 0 or i == total_pares - 1:
                elapsed = time.time() - start_time
                remaining = (elapsed/(i+1)) * (total_pares-(i+1)) if i > 0 else 0
                print(f"Progreso: {i+1}/{total_pares} | "
                      f"Tiempo: {elapsed:.1f}s | "
                      f"Estimado: {remaining:.1f}s restantes", end='/r')
    
    for i, j, dist in resultados:
        dist_matrix[i, j] = dist_matrix[j, i] = dist
    
    print("/nCálculo completado.")
    return dist_matrix



# ----------------------------
# 3. Generación de población
# ----------------------------

def generar_poblacion(nodos, dist_geodesica, n_individuos=3000, n_puntos=7, min_dist=30.0):
    """Genera población con verificación de distancias."""
    n = len(nodos)
    poblacion = []
    intentos = 0
    max_intentos = n_individuos * 1000
    
    print(f"/nGenerando {n_individuos} individuos...")
    
    while len(poblacion) < n_individuos and intentos < max_intentos:
        candidato = np.random.choice(n, size=n_puntos, replace=False)
        valido = True
        for i, j in itertools.combinations(candidato, 2):
            if dist_geodesica[i, j] < min_dist:
                valido = False
                break
        if valido:
            poblacion.append(candidato)
        
        intentos += 1
        if intentos % 100 == 0:
            print(f"Intentos: {intentos} | Válidos: {len(poblacion)}", end='/r')
    
    print(f"/nPoblación generada: {len(poblacion)} individuos válidos")
    return np.array(poblacion)



def combinar_poblaciones(poblacion_der, poblacion_izq, ptos_der, ptos_izq):
    min_len = min(len(poblacion_der), len(poblacion_izq))
    poblacion_combinada = []
    
    for i in range(min_len):
        # Combinar 2 puntos derechos + 5 puntos izquierdos
        combinacion = np.concatenate([poblacion_der[i][:ptos_der], poblacion_izq[i][:ptos_izq]])
        poblacion_combinada.append(combinacion)
    
    return np.array(poblacion_combinada)

# ----------------------------
# 4. Guardado de individuos (versión corregida)
# ----------------------------



def guardar_poblacion_simple(nodos_der,nodos_izq, poblacion, nodos, output_path, n_puntos_der=3, n_puntos_izq=4):
    """
    Guarda toda la población como un solo archivo VTP usando PyVista de forma simple.
    """
    puntos_totales = []
    punto_ids = []
    individuo_ids = []
    tipo_punto = []
    etiquetas_global = []
    

    for individuo_id, individuo in enumerate(poblacion):
        indices = list(individuo)
        
        for i, idx in enumerate(indices):
            
                        
            if i<n_puntos_der:
                punto_global_id, coord_der = nodos[nodos_der[idx]]
                puntos_totales.append(coord_der)
                punto_ids.append(idx)
                individuo_ids.append(individuo_id)
                tipo_punto.append(0)
                etiquetas_global.append(punto_global_id)

            else:
                punto_global_id, coord_izq = nodos[nodos_izq[idx]]
                puntos_totales.append(coord_izq)
                punto_ids.append(idx)
                individuo_ids.append(individuo_id)
                tipo_punto.append(1)
                etiquetas_global.append(punto_global_id)
            
            
            
            
            
    # Convertir a array numpy
    puntos_array = np.array(puntos_totales)

    # Crear PyVista PolyData
    poly = pv.PolyData(puntos_array)
    poly['PuntoID'] = np.array(punto_ids)
    poly['IndividuoID'] = np.array(individuo_ids)
    poly['TipoPunto'] = np.array(tipo_punto)
    poly['ID_malla_completa'] = np.array(etiquetas_global)

    # Guardar a disco
    filename = os.path.join(output_dir, "poblacion_simple.vtp")
    poly.save(filename)
    print(f"Población guardada en: {filename}")






'''
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



def procesar_archivos(vtk_file, nodos,write=None):
    """Carga los puntos, encuentra los nodos más cercanos y guarda los resultados."""
    total_elvira_node_IDs=[]
    puntos = cargar_puntos_vtk(vtk_file)
    
    
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





def procesar_individuo(args):
    """Función para procesar un individuo en paralelo."""
    individuo_id, individuo, nodos_der, nodos_izq, nodos_file, output_path, n_puntos_der,n_puntos_izq = args
    
    try:
        # Obtener puntos del individuo actual
        puntos_der = nodos_der[individuo[:n_puntos_der]]
        puntos_izq = nodos_izq[individuo[n_puntos_der:]]
        
        # Crear VTK temporal en memoria (sin guardar a disco)
        temp_points = vtk.vtkPoints()
        temp_vertices = vtk.vtkCellArray()
        temp_ugrid = vtk.vtkUnstructuredGrid()
        
        # Añadir puntos
        for punto in np.vstack([puntos_der, puntos_izq]):
            temp_points.InsertNextPoint(punto)
            temp_vertices.InsertNextCell(1)
            temp_vertices.InsertCellPoint(0)  # Solo un punto por celda
        
        temp_ugrid.SetPoints(temp_points)
        temp_ugrid.SetCells(vtk.VTK_VERTEX, temp_vertices)
        
        # Guardar temporalmente a disco (necesario para procesar_archivos)
        temp_vtk = os.path.join(output_path, f"temp_individuo_{individuo_id}.vtk")
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(temp_vtk)
        writer.SetInputData(temp_ugrid)
        writer.Write()
        
        # Procesar el archivo
        resultados = procesar_archivos(temp_vtk, nodos_file)
        
        # Eliminar archivo temporal
        os.remove(temp_vtk)
        
        return (individuo_id, resultados, puntos_der, puntos_izq)
    
    except Exception as e:
        print(f"Error procesando individuo {individuo_id}: {str(e)}")
        return (individuo_id, None, None, None)

'''

def guardar_poblacion_completa_paralelo(nodos_der, nodos_izq, poblacion, nodos_file, output_path,n_puntos_der=3,n_puntos_izq=4, n_procesos=None):
    """Versión paralelizada del guardado de población."""
    try:
        if n_procesos is None:
            n_procesos = min(cpu_count(), 112)
        
        # Crear estructura VTK principal
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        ugrid = vtk.vtkUnstructuredGrid()
        
        # Arrays para metadatos
        punto_ids = vtk.vtkIntArray()
        punto_ids.SetName("PuntoID")
        
        individuo_ids = vtk.vtkIntArray()
        individuo_ids.SetName("IndividuoID")
        
        tipo_punto = vtk.vtkIntArray()
        tipo_punto.SetName("TipoPunto")
        
        etiquetas_array = vtk.vtkIntArray()
        etiquetas_array.SetName("ID_malla_completa")
        
        # Preparar argumentos para paralelización
        args = [(i, ind, nodos_der, nodos_izq, nodos_file, output_path, n_puntos_der, n_puntos_izq) 
                for i, ind in enumerate(poblacion)]
        
        # Procesar en paralelo
        resultados_procesamiento = [None] * len(poblacion)
        puntos_der_list = [None] * len(poblacion)
        puntos_izq_list = [None] * len(poblacion)
        
        print(f"/nProcesando {len(poblacion)} individuos en paralelo...")
        start_time = time.time()
        
        with Pool(processes=n_procesos) as pool:
            for i, (individuo_id, resultados, puntos_der, puntos_izq) in enumerate(pool.imap_unordered(procesar_individuo, args)):
                resultados_procesamiento[individuo_id] = resultados
                puntos_der_list[individuo_id] = puntos_der
                puntos_izq_list[individuo_id] = puntos_izq
                
                if i % 10 == 0 or i == len(poblacion) - 1:
                    elapsed = time.time() - start_time
                    print(f"Procesados {i+1}/{len(poblacion)} individuos | Tiempo: {elapsed:.1f}s", end='/r')
        
        print("/nConstruyendo VTK final...")
        
        # Construir VTK final con los resultados
        global_point_id = 0
        for individuo_id in range(len(poblacion)):
            # Añadir puntos derechos (tipo 0)
            for i, punto in enumerate(puntos_der_list[individuo_id]):
                points.InsertNextPoint(punto)
                vertices.InsertNextCell(1)
                vertices.InsertCellPoint(global_point_id)
                
                punto_ids.InsertNextValue(poblacion[individuo_id][i])
                print(f'hola: {poblacion[individuo_id][i]}')
                
                
                individuo_ids.InsertNextValue(individuo_id)
                tipo_punto.InsertNextValue(0)
                etiquetas_array.InsertNextValue(resultados_procesamiento[individuo_id][i])
                
                global_point_id += 1
            
            # Añadir puntos izquierdos (tipo 1)
            for i, punto in enumerate(puntos_izq_list[individuo_id]):
                points.InsertNextPoint(punto)
                vertices.InsertNextCell(1)
                vertices.InsertCellPoint(global_point_id)
                
                punto_idx = poblacion[individuo_id][n_puntos_der + i]

                
                punto_ids.InsertNextValue(punto_idx)
                individuo_ids.InsertNextValue(individuo_id)
                tipo_punto.InsertNextValue(1)
                etiquetas_array.InsertNextValue(resultados_procesamiento[individuo_id][n_puntos_der+i])
                
                global_point_id += 1
        
        # Configurar malla final
        ugrid.SetPoints(points)
        ugrid.SetCells(vtk.VTK_VERTEX, vertices)
        
        # Añadir arrays de datos
        ugrid.GetPointData().AddArray(punto_ids)
        ugrid.GetPointData().AddArray(individuo_ids)
        ugrid.GetPointData().AddArray(tipo_punto)
        ugrid.GetPointData().AddArray(etiquetas_array)
        
        # Guardar archivo final
        output_vtk = os.path.join(output_path, 'poblacion_final.vtk')
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(output_vtk)
        writer.SetInputData(ugrid)
        writer.Write()
        
        print(f"/nArchivo VTK guardado en: {output_vtk}")
        return True, resultados_procesamiento
    
    except Exception as e:
        print(f"/nError en guardar_poblacion_completa_paralelo: {str(e)}")
        return False, None
   
   
   
##############################################################################################################################
###################################################################
#############################################

# Crear el directorio (si no existe)
try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directorio creado exitosamente en: {os.path.abspath(output_dir)}")
except Exception as e:
    print(f"Error al crear directorio: {str(e)}")
   
   

# ----------------------------
# 1. Cargar malla y nodos candidatos
# ----------------------------

#leer la malla, añadir globlIds y sacar los endos
vol_mesh = smart_reader('/home/nodo38/sandra/GA_OpenCarp/CINC_2025_OpenCarp/prueba2.vtu')
vol_mesh = addGlobalIds(vol_mesh)

         
surf_mesh = extract_surface(vol_mesh)
endo = threshold(surf_mesh, 0.9, 1, 'tm') 




# Extraer nodos desde el endocardio nodos= (globalID,[x,y,z])
has_global_ids = endo.GetPointData().HasArray("GlobalIds")
if not has_global_ids:
    raise ValueError("La malla no contiene 'GlobalIds'. Asegúrate de haber llamado a `addGlobalIds`.")

puntos = endo.GetPoints()
global_ids = endo.point_data['GlobalIds']

nodos = []
for i in range(puntos.GetNumberOfPoints()):
    nodo_id = int(global_ids[i])
    coords = np.array(puntos.GetPoint(i))
    nodos.append((nodo_id, coords))



#save GlobalIds complete mesh and endo to check
vtkWrite(vol_mesh,'./vol_mesh_globaIds.vtu')         
vtkWrite(endo,'./endo_globaIds.vtu')


# Sacamos el globalId del nodo más cercano al centroide de los segmentos AHA como candidatos de estimulación
vi, vd = get_aha_centroids_and_global_ids(endo)

nodos_der=[]
nodos_izq=[]

filename = os.path.join(output_dir, "aha_vi.txt")
with open(filename, 'w') as f:
    f.write("GlobalID\tAHA_segment\n")
    for seg in vi:
        f.write(f"{seg['closest_node_id']}\t{seg['AHA_segment']}\n")
        nodos_izq.append(seg['closest_node_id'])

filename = os.path.join(output_dir, "aha_vd.txt")
with open(filename, 'w') as f:
    f.write("GlobalID\tAHA_segment\n")
    for seg in vd:
        f.write(f"{seg['closest_node_id']}\t{seg['AHA_segment']}\n")
        nodos_der.append(seg['closest_node_id'])


# Combinar las dos listas
combinados = vi + vd

# Mezclar el orden de forma aleatoria
combinados = np.random.permutation(combinados).tolist()

# Escribir al archivo
filename = os.path.join(output_dir, "unique_aha_values.txt")
with open(filename, 'w') as f:
    for seg in combinados:
        f.write(f"{seg['closest_node_id']}\n")   


 
    
# ----------------------------
# 2. Cálculo de distancias geodésicas
# ----------------------------

print("/nIniciando cálculo de matriz geodésica derecha...")
dist_geodesica_der = calcular_matriz_geodesica_paralelo(endo, nodos_der)

np.save(os.path.join(output_dir,"dist_geodesica_der.npy"), dist_geodesica_der)
print(f"Matriz derecha guardada. Dimensiones: {dist_geodesica_der.shape}")


print("/nIniciando cálculo de matriz geodésica izquierda...")
dist_geodesica_izq = calcular_matriz_geodesica_paralelo(endo, nodos_izq)

np.save(os.path.join(output_dir,"dist_geodesica_izq.npy"), dist_geodesica_izq)
print(f"Matriz izquierda guardada. Dimensiones: {dist_geodesica_izq.shape}")


# ----------------------------
# 3. Generación de población
# ----------------------------

poblacion_der_idx = generar_poblacion(nodos_der, dist_geodesica_der,n_individuos=3000, n_puntos=3, min_dist=30.0)
poblacion_izq_idx = generar_poblacion(nodos_izq, dist_geodesica_izq,n_individuos=3000, n_puntos=4, min_dist=30.0)

np.save(os.path.join(output_dir,"poblacion_der_idx.npy"), poblacion_der_idx)
np.save(os.path.join(output_dir,"poblacion_izq_idx.npy"), poblacion_izq_idx)



# Convertir índices de nodos_der a globalIDs
der_idx_to_globalId = {i: item for i, item in enumerate(nodos_der)}
izq_idx_to_globalId = {i: item for i, item in enumerate(nodos_izq)}


poblacion_der_global = np.array([
    [der_idx_to_globalId[i] for i in individuo]
    for individuo in poblacion_der_idx
])

# Convertir índices de nodos_izq a globalIDs
poblacion_izq_global = np.array([
    [izq_idx_to_globalId[i] for i in individuo]
    for individuo in poblacion_izq_idx
])


poblacion_final = combinar_poblaciones(poblacion_der_global, poblacion_izq_global,  ptos_der=3, ptos_izq=4)


np.save(os.path.join(output_dir,"poblacion_der.npy"), poblacion_der_global)
np.save(os.path.join(output_dir,"poblacion_izq.npy"), poblacion_izq_global)
np.save(os.path.join(output_dir,"poblacion_final.npy"), poblacion_final)

print(f"La población_der tiene dimensiones {poblacion_der_global.shape}, La población_izq tiene dimensiones{poblacion_izq_global.shape} y la final tiene {poblacion_final.shape} individuos ")


# ----------------------------
# 4. Guardado de individuos (versión corregida)
# ----------------------------

'''
print("\nGuardando población...")
guardar_poblacion_simple(
    nodos_der=nodos_der,
    nodos_izq=nodos_izq,
    poblacion=poblacion_final,
    nodos=nodos,
    output_path=output_dir,
    n_puntos_der=3,
    n_puntos_izq=4
)
print("\n¡Guardado completado exitosamente!")
'''


# Convertir nodos a diccionario para acceso rápido
nodos_dict = {gid: coord for gid, coord in nodos}

# Inicializar estructura de puntos VTK
points = vtk.vtkPoints()
individuo_ids = vtk.vtkIntArray()
individuo_ids.SetName("individuo_id")

# Agregar puntos y asociar cada uno a su individuo
for idx_ind, individuo in enumerate(poblacion_final):
    for global_id in individuo:
        coord = nodos_dict.get(global_id)
        if coord is None:
            raise ValueError(f"GlobalID {global_id} no encontrado en nodos.")
        points.InsertNextPoint(coord)
        individuo_ids.InsertNextValue(idx_ind)

# Crear vtkPolyData con solo puntos
polydata = vtk.vtkPolyData()
polydata.SetPoints(points)
polydata.GetPointData().AddArray(individuo_ids)
polydata.GetPointData().SetActiveScalars("individuo_id")


glyphFilter = vtk.vtkVertexGlyphFilter()
glyphFilter.SetInputData(polydata)
glyphFilter.Update()

polydata_with_vertices = glyphFilter.GetOutput()


# Guardar como archivo VTP
writer = vtk.vtkXMLPolyDataWriter()
filename = os.path.join(output_dir, "poblacion.vtp")
writer.SetFileName(filename)
writer.SetInputData(polydata_with_vertices)
writer.Write()







































'''

print("/nGuardando individuos como VTK...")
# Después de generar tu población final (combinando derecha e izquierda)


globalID_to_coords = {item[0]: item[1] for item in nodos}







nodos_der_xyz = np.array([globalID_to_coords[gid] for gid in nodos_der])
nodos_izq_xyz = np.array([globalID_to_coords[gid] for gid in nodos_izq])


success, resultados_procesamiento = guardar_poblacion_completa_paralelo(
    nodos_der_xyz,
    nodos_izq_xyz,
    poblacion_final,
    nodos,n_puntos_der=3,n_puntos_izq=4,
    output_path=os.path.join(output_dir),
    n_procesos=112  # Ajusta según tus necesidades
)

# Guardar los resultados del procesamiento
if success and resultados_procesamiento is not None:
    # Convertir a array numpy para facilitar el guardado
    resultados_array = np.array(resultados_procesamiento, dtype=object)
    
    # Guardar como archivo .npy
    np.save(os.path.join(output_dir, "poblacion_total.npy"), resultados_array)
    valores_unicos=np.unique(resultados_array)
   
    saving_path = os.path.join(output_dir, "poblacion_unique_values.txt")
    np.savetxt(saving_path, valores_unicos, fmt='%d')
    
    # También guardar como texto legible
    with open(os.path.join(output_dir, "poblacion_total.txt"), "w") as f:
        f.write(f"Semilla: {SEMILLA}/n")
        for i, resultado_individuo in enumerate(resultados_procesamiento):
            f.write(f"Individuo {i}: {resultado_individuo}/n")
 

    print(f"/nResultados de procesamiento guardados en {output_dir}")

print("/n¡Proceso completado exitosamente!")


'''