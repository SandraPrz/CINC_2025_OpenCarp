#Created by: Jorge Sanchez

import vtk
from vtk.util import numpy_support
from pathlib import Path
import random
import math

from vtk.util.numpy_support import vtk_to_numpy

from typing import List

import json
from typing import Any, Optional

import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree

import os
import shutil
import filecmp

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

def farthest_point_sampling_ids(polydata, sample_distance=None, num_samples=None, seed=None):
    """
    Perform farthest-point sampling on a vtkPolyData surface and return the point IDs.
    
    Either `sample_distance` or `num_samples` must be provided:
      - If sample_distance is set, stops when the largest candidate distance 
        drops below sample_distance.
      - If num_samples is set, stops when that many points have been selected.
    
    Parameters
    ----------
    polydata : vtk.vtkPolyData
        The input surface.
    sample_distance : float, optional
        Minimum desired separation between sampled points. If provided,
        sampling continues until no candidate has distance >= sample_distance.
    num_samples : int, optional
        Fixed number of points to sample.
    seed : int, optional
        Random seed for reproducible initial point choice.
    
    Returns
    -------
    list of int
        The point IDs of the sampled points.
    """
    if (sample_distance is None and num_samples is None) or \
       (sample_distance is not None and num_samples is not None):
        raise ValueError("You must specify exactly one of sample_distance or num_samples.")
    
    pts = polydata.GetPoints()
    n_pts = pts.GetNumberOfPoints()
    if n_pts == 0:
        return []
    
    # Random initial seed index
    rng = random.Random(seed)
    first_idx = rng.randrange(n_pts)
    
    # distances[i] = min distance from point i to any selected sample so far
    distances = [float('inf')] * n_pts
    samples = [first_idx]
    
    # Initialize distances from the first sampled point
    p0 = pts.GetPoint(first_idx)
    for i in range(n_pts):
        d2 = vtk.vtkMath.Distance2BetweenPoints(p0, pts.GetPoint(i))
        distances[i] = math.sqrt(d2)
    
    # Iteratively pick the point farthest from all already sampled
    while True:
        # find the point with maximum "min distance" to current samples
        max_idx = max(range(n_pts), key=lambda i: distances[i])
        max_dist = distances[max_idx]
        
        # stopping criteria
        if sample_distance is not None:
            if max_dist < sample_distance:
                break
        else:  # num_samples is not None
            if len(samples) >= num_samples:
                break
        
        # accept this point
        samples.append(max_idx)
        p_new = pts.GetPoint(max_idx)
        
        # update distances array
        for i in range(n_pts):
            d2 = vtk.vtkMath.Distance2BetweenPoints(p_new, pts.GetPoint(i))
            d = math.sqrt(d2)
            if d < distances[i]:
                distances[i] = d
    
    return samples

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

def get_cobivecoaux(mesh, mesh_ids):
    """! Get the coordinates of the mesh ids.
    @param mesh_ids The mesh ids to get the coordinates.
    @return ab, tm, rt, tv, ts The coordinates of the mesh ids.
    """
    # Coordinates arrays
    ab = numpy_support.vtk_to_numpy(mesh.GetPointData().GetScalars("ab"))
    tm = numpy_support.vtk_to_numpy(mesh.GetPointData().GetScalars("tm")) 
    rt = numpy_support.vtk_to_numpy(mesh.GetPointData().GetScalars("rt"))
    tv = numpy_support.vtk_to_numpy(mesh.GetPointData().GetScalars("tv")) 
    ts = numpy_support.vtk_to_numpy(mesh.GetPointData().GetScalars("ts"))

    return ab[mesh_ids], tm[mesh_ids], rt[mesh_ids], tv[mesh_ids], ts[mesh_ids]

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

def farthest_point_sampling_global_ids(polydata, sample_distance=None, num_samples=None, seed=None):
    """
    Perform farthest-point sampling on a vtkPolyData surface and return the GlobalIds of sampled points.
    
    Parameters
    ----------
    polydata : vtk.vtkPolyData
        The input surface. Must contain a point array named 'GlobalIds'.
    sample_distance : float, optional
        Minimum desired separation between sampled points.
    num_samples : int, optional
        Fixed number of points to sample.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    list of int
        The GlobalIds of the sampled points.
    """
    if (sample_distance is None and num_samples is None) or \
       (sample_distance is not None and num_samples is not None):
        raise ValueError("You must specify exactly one of sample_distance or num_samples.")
    
    pts = polydata.GetPoints()
    n_pts = pts.GetNumberOfPoints()
    if n_pts == 0:
        return []
    
    # Get GlobalIds array
    gid_array = polydata.GetPointData().GetArray("GlobalIds")
    if gid_array is None:
        raise ValueError("polydata does not have a point array named 'GlobalIds'")
    
    global_ids = vtk_to_numpy(gid_array)

    # Random initial seed index
    rng = random.Random(seed)
    first_idx = rng.randrange(n_pts)
    
    distances = [float('inf')] * n_pts
    samples = [first_idx]
    
    p0 = pts.GetPoint(first_idx)
    for i in range(n_pts):
        d2 = vtk.vtkMath.Distance2BetweenPoints(p0, pts.GetPoint(i))
        distances[i] = math.sqrt(d2)
    
    while True:
        max_idx = max(range(n_pts), key=lambda i: distances[i])
        max_dist = distances[max_idx]
        
        if sample_distance is not None:
            if max_dist < sample_distance:
                break
        else:
            if len(samples) >= num_samples:
                break
        
        samples.append(max_idx)
        p_new = pts.GetPoint(max_idx)
        
        for i in range(n_pts):
            d2 = vtk.vtkMath.Distance2BetweenPoints(p_new, pts.GetPoint(i))
            d = math.sqrt(d2)
            if d < distances[i]:
                distances[i] = d
    
    # Return GlobalIds instead of internal indices
    return [int(global_ids[i]) for i in samples]

def get_closest_global_ids(points: vtk.vtkPolyData,
                           mesh: vtk.vtkPolyData,
                          ) -> List[int]:
    """
    For each point in `points`, find the closest point on `mesh`
    and return its global ID from the mesh's point data.

    Parameters
       points : vtk.vtkPolyData
        PolyData containing the points to locate.
       mesh : vtk.vtkPolyData
        PolyData representing the mesh with a global ID point data array.
       id_array_name : str, optional
        Name of the point data array on `mesh` storing global IDs.

    Returns
    -------
    List[int]
        Global IDs of the closest mesh points for each input point.
    """
    # Build a point locator for the mesh
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    # For each input point, find closest mesh point and record its ID
    global_ids: List[int] = []
    for i in range(points.GetNumberOfPoints()):
        pt = points.GetPoint(i)
        closest_id = locator.FindClosestPoint(pt)
        global_ids.append(int(closest_id))

    return global_ids

from typing import List
import vtk

def get_closest_global_ids_sorted_by_label(points: vtk.vtkPolyData,
                                           mesh: vtk.vtkPolyData,
                                           label_array_name: str = 'NumLeads') -> List[int]:
    """
    Finds the closest global mesh point for each point in `points`, ordered by the values
    in the given point data array (e.g., 'NumLeads').

    Parameters
    ----------
    points : vtk.vtkPolyData
        PolyData containing the electrode points.
    mesh : vtk.vtkPolyData
        PolyData mesh that contains global IDs.
    label_array_name : str
        Name of the scalar array used to sort the electrode points.

    Returns
    -------
    List[int]
        List of global IDs from `mesh` that correspond to sorted `points`.
    """
    # Extract the label array from point data
    label_array = points.GetPointData().GetArray(label_array_name)
    if label_array is None:
        raise ValueError(f"Array '{label_array_name}' not found in point data.")

    # Get (index, label) pairs and sort by label value
    labeled_points = [(i, label_array.GetValue(i)) for i in range(points.GetNumberOfPoints())]
    labeled_points.sort(key=lambda x: x[1])

    # Build a point locator for the mesh
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    # For each input point, find closest mesh point and record its ID
    global_ids: List[int] = []
    for i, _ in labeled_points:
        pt = points.GetPoint(i)
        closest_id = locator.FindClosestPoint(pt)
        global_ids.append(int(closest_id))

    
    return global_ids


def modify_json_entry(
    file_path: str,
    key: str,
    new_value: Any,
    output_file_path: Optional[str] = None,
    *,
    encoding: str = 'utf-8',
    indent: Optional[int] = 4
) -> None:
    """
    Reads a JSON file, updates the value for a given key, 
    and writes the result back to a file.

    :param file_path: Path to the input JSON file.
    :param key: Top-level key whose value should be modified.
    :param new_value: The new value to assign to key.
    :param output_file_path: If provided, writes to this file; otherwise overwrites file_path.
    :param encoding: File encoding (default 'utf-8').
    :param indent: Number of spaces to indent when writing JSON (None for compact).
    """
    # Load the existing data
    with open(file_path, 'r', encoding=encoding) as f:
        data = json.load(f)

    # Check that the key exists (optional)
    if key not in data:
        raise KeyError(f"Key {key!r} not found in JSON file.")

    # Modify the entry
    data[key] = new_value

    # Determine where to write
    target = output_file_path or file_path

    # Write back out
    with open(target, 'w', encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

    print(f"Updated key {key!r} in {target!r}.")
    





def point_selection(region_mesh,min_dist=6):
    """
    Take a selected region with the label GlobalIds and obtain equidistance nodes by their GlobalIds

    Parameters
    ----------
    region_mesh : vtk.vtkPolyData
        PolyData containing the selected region.
    min_dist : minimal distance between selected nodes
    """

    points = np.array([region_mesh.GetPoint(i) for i in range(region_mesh.GetNumberOfPoints())])
    
    
    
    # Crear un árbol para búsquedas eficientes
    tree = cKDTree(points)
    
    # Crear lista para guardar puntos seleccionados
    selected = []
    visited = np.zeros(len(points), dtype=bool)
    
    
    
    for i, pt in enumerate(points):
        if not visited[i]:
            selected.append(pt)
            idxs = tree.query_ball_point(pt, min_dist)
            visited[idxs] = True
    
    selected = np.array(selected)
    #selected_mesh = pv.PolyData(selected)
    #selected_mesh.plot()
    
    
    
    # GlobalIds están en region_mesh
    global_ids_array = region_mesh.GetPointData().GetArray("GlobalIds")
    
    
    # Obtener los puntos de region_mesh para comparar
    acc_points = np.array([region_mesh.GetPoint(i) for i in range(region_mesh.GetNumberOfPoints())])
    
    # Mapear coordenadas de puntos seleccionados a GlobalIds
    selected_global_ids = []
    
    for sel_pt in selected:
        # Buscar el índice del punto más cercano en region_mesh
        dists = np.linalg.norm(acc_points - sel_pt, axis=1)
        closest_idx = np.argmin(dists)
        
        # Obtener su GlobalId
        global_id = global_ids_array.GetValue(closest_idx)
        selected_global_ids.append(global_id)
    
    # Convertir a array y mostrar
    selected_global_ids = selected_global_ids
    
    
    return selected_global_ids


def extract_uvcs(mesh, uvc_names=['ab', 'tm', 'rt', 'tv', 'ts']):
    uvc_data = []
    for name in uvc_names:
        array = mesh.GetPointData().GetArray(name)
        if array is None:
            raise ValueError(f"No se encontró el array UVC '{name}' en el mesh.")
        values = np.array([array.GetValue(i) for i in range(mesh.GetNumberOfPoints())])
        uvc_data.append(values)
    return np.stack(uvc_data, axis=1)  # shape: (n_points, 5)
    
    
def preprocess_uvc_tree(mesh):
    """
    Extract UVC coordinates from mesh and build a KDTree for fast lookup.

    Returns:
        uvc_coords: np.ndarray of shape (n_points, 5)
        tree: cKDTree built from UVC coordinates
        global_ids: vtkDataArray of GlobalIds
    """
    uvc_coords = extract_uvcs(mesh)
    tree = cKDTree(uvc_coords)

    global_ids = mesh.GetPointData().GetArray("GlobalIds")
    if global_ids is None:
        raise ValueError("The mesh does not contain a 'GlobalIds' array.")

    return uvc_coords, tree, global_ids


def find_closest_nodes_batch(tree, global_ids, target_uvcs):
    """
    Query closest mesh nodes for multiple target UVC coordinates.

    Parameters:
        tree: KDTree built from mesh UVCs
        global_ids: vtkDataArray of GlobalIds
        target_uvcs: (N, 5) NumPy array of UVC targets

    Returns:
        matched_ids: list of GlobalIds
        distances: list of UVC-space distances
    """
    distances, indices = tree.query(target_uvcs)
    matched_ids = [global_ids.GetValue(i) for i in indices]
    return matched_ids, distances


def create_vtk_from_nodes(vol_mesh, global_ids, uvc_coords, stim_times, output_filename):
    """
    Crea y guarda un archivo VTK con los nodos seleccionados,
    sus coordenadas, opcionalmente UVC y tiempo de estimulación.

    Parámetros:
    - vol_mesh: vtkUnstructuredGrid original
    - global_ids: lista con los global IDs de los nodos seleccionados
    - uvc_coords: array (N x 5) con valores UVC o None si no se quiere añadir
    - stim_times: lista o array con tiempos de estimulación para cada nodo
    - output_filename: nombre de archivo VTK a guardar (.vtu)
    """
    points = vtk.vtkPoints()
    
    global_id_array = vtk.vtkIntArray()
    global_id_array.SetName("GlobalId")

    stim_time_array = vtk.vtkDoubleArray()
    stim_time_array.SetName("StimTime")

    # Solo crear arrays UVC si uvc_coords no es None
    if uvc_coords is not None:
        uvc_names = ['ab', 'tm', 'rt', 'tv', 'ts']
        uvc_arrays = []
        for name in uvc_names:
            arr = vtk.vtkDoubleArray()
            arr.SetName(name)
            uvc_arrays.append(arr)
    else:
        uvc_arrays = []

    for i, gid in enumerate(global_ids):
        x, y, z = vol_mesh.GetPoint(gid)
        points.InsertNextPoint(x, y, z)

        global_id_array.InsertNextValue(gid)
        stim_time_array.InsertNextValue(stim_times[i])

        if uvc_coords is not None:
            for j, arr in enumerate(uvc_arrays):
                arr.InsertNextValue(uvc_coords[i,j])

    selected_mesh = vtk.vtkUnstructuredGrid()
    selected_mesh.SetPoints(points)

    for i in range(points.GetNumberOfPoints()):
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, i)
        selected_mesh.InsertNextCell(vertex.GetCellType(), vertex.GetPointIds())

    selected_mesh.GetPointData().AddArray(global_id_array)
    selected_mesh.GetPointData().AddArray(stim_time_array)
    for arr in uvc_arrays:
        selected_mesh.GetPointData().AddArray(arr)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(selected_mesh)
    writer.Write()

    print(f"Archivo guardado: {output_filename}")
    

def copy_and_delete_folder(source, target_base):
    """
    Safely copies a folder from WSL to a mounted Windows drive.
    Ensures the copy is complete and correct before deleting the original.

    :param source: Path to the source folder in WSL (e.g., "/home/user/project")
    :param target_base: Path to the destination base on a Windows drive (e.g., "/mnt/e/backup")
    """
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source folder does not exist: {source}")
    
    # Create destination base if it doesn't exist
    if not os.path.exists(target_base):
        os.makedirs(target_base)
    
    folder_name = os.path.basename(source.rstrip("/"))
    full_target = os.path.join(target_base, folder_name)
    
    print(f"Copying '{source}' to '{full_target}'...")

    try:
        # Copy the entire folder
        shutil.copytree(source, full_target)

        # Compare content to ensure copy integrity
        comparison = filecmp.dircmp(source, full_target)
        if comparison.left_only or comparison.right_only or comparison.diff_files:
            raise Exception("Copy is not identical. Original folder will NOT be deleted.")
        
        # If everything is okay, delete the original
        shutil.rmtree(source)
        print(f"Copy verified. Original folder deleted: {source}")
    
    except Exception as e:
        print(f"Error during operation: {e}")
        if os.path.exists(full_target):
            print("Partial copy will be retained at the destination.")

