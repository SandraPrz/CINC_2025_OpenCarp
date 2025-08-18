import pyvista as pv
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from types import SimpleNamespace
import time
import os
import json

try:
    import pyamg
    PYAMG_AVAILABLE = True
except ImportError:
    PYAMG_AVAILABLE = False

# ==================================================
# Utility Functions
# ==================================================
def tic(message=""):
    """Start a timer."""
    print(f"{message} ... ", end="", flush=True)
    return time.time()

def toc(start_time):
    """Stop the timer and print duration."""
    elapsed = time.time() - start_time
    print(f"Done ({elapsed:.2f}s)")

# ==================================================
# Configuration Loading Function
# ==================================================
def load_config(config_file_path="config.json"):
    """Loads configuration from a JSON file."""
    print(f"Attempting to load configuration from {config_file_path}...")
    try:
        with open(config_file_path, 'r') as f:
            cfg_dict = json.load(f)
        cfg_data = SimpleNamespace(**cfg_dict) 
        print(f"Successfully loaded configuration from {config_file_path}")
        return cfg_data
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file_path}' not found. Please create it.")
        exit()
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse '{config_file_path}'. Invalid JSON: {e}")
        exit()
    except Exception as e:
        print(f"Error loading configuration from '{config_file_path}': {e}")
        exit()

# ==================================================
# Solver Callback
# ==================================================
class SolverCallback:
    def __init__(self, solver_type):
        self.niter = 0
        self.solver_type = solver_type.upper()
        self.last_print_iter = 0

    def __call__(self, rk_or_xk): # rk for CG/BiCGSTAB (residual), xk for GMRES (current solution)
        self.niter += 1
        if self.niter % 100 == 0 or self.niter == 1:
            print(f"          {self.solver_type} Iter {self.niter}...")
            self.last_print_iter = self.niter

# ==================================================
# Grounding Function (More Robust Version)
# ==================================================
def apply_grounding(A_orig, b_orig, ground_node_id):
    """
    Applies grounding (Dirichlet boundary condition) to A and b for a specific node.
    This version uses matrix transpose for potentially better robustness/performance.
    """
    print(f"          Applying grounding at node {ground_node_id}.")
    A_mod = A_orig.copy().tolil() 
    b_mod = b_orig.copy()

    # Set the ground_node row to 0, then add diagonal=1
    A_mod.rows[ground_node_id] = [ground_node_id]
    A_mod.data[ground_node_id] = [1.0]

    A_t = A_mod.T.tolil()
    A_t.rows[ground_node_id] = [] # Clear all entries in the row of A_t (which is the column of A_mod)
    A_t.data[ground_node_id] = []
    A_mod = A_t.T.tolil() # Transpose back
    A_mod[ground_node_id, ground_node_id] = 1.0
    b_mod[ground_node_id] = 0.0
    
    return A_mod.tocsr(), b_mod

# ==================================================
# Mesh Loading and Geometry Precomputation
# ==================================================
def _load_and_prepare_mesh_data(cfg):
    """Loads mesh, extracts tetrahedra, and computes geometric properties."""
    t0_mesh_load = tic("    Loading mesh and extracting tetrahedra")
    try:
        mesh0 = pv.read(cfg.VTK_FILE)
    except FileNotFoundError:
        toc(t0_mesh_load)
        print(f"\nError: Input mesh file '{cfg.VTK_FILE}' not found.")
        return None
    except Exception as e:
        toc(t0_mesh_load)
        print(f"\nError reading mesh file '{cfg.VTK_FILE}': {e}")
        return None

    tet_ids = np.where(mesh0.celltypes == pv.CellType.TETRA)[0]
    if not tet_ids.size:
        toc(t0_mesh_load)
        print(f"\nError: No tetrahedra found in '{cfg.VTK_FILE}'.")
        return None
    
    tet_mesh = mesh0.extract_cells(tet_ids)
    points = tet_mesh.points.astype(np.float64)
    cells_raw = tet_mesh.cells
    num_tets = tet_mesh.n_cells
    tets = cells_raw.reshape(num_tets, 5)[:, 1:5].astype(np.int32)

    if "cell_scalars" not in tet_mesh.cell_data:
        toc(t0_mesh_load)
        print(f"\nError: 'cell_scalars' not found in cell_data of '{cfg.VTK_FILE}'. Cannot determine tissue regions.")
        return None
    cell_labels = tet_mesh.cell_data["cell_scalars"]
    num_nodes = points.shape[0]
    toc(t0_mesh_load)
    print(f"          {num_nodes} nodes, {num_tets} tetrahedra")

    t0_geom = tic("    Computing geometric properties (gradients, volumes)")
    X = points[tets]
    
    B_mat = np.stack([X[:, 1, :] - X[:, 0, :],
                      X[:, 2, :] - X[:, 0, :],
                      X[:, 3, :] - X[:, 0, :]], axis=-1) 
    
    detB = np.linalg.det(B_mat)
    if np.any(np.abs(detB) < 1e-12):
        num_degenerate = np.sum(np.abs(detB) < 1e-12)
        print(f"          Warning: {num_degenerate} tetrahedra have zero or near-zero volume. This may cause issues.")
        detB[np.abs(detB) < 1e-12] = 1e-12

    volumes = np.abs(detB) / 6.0
    
    try:
        invB = np.linalg.inv(B_mat)
    except np.linalg.LinAlgError:
        toc(t0_geom)
        print("\nError: Singular matrix encountered during B_mat inversion (degenerate tetrahedra). Cannot compute grad_phi.")
        return None

    # Gradients of basis functions (grad_phi_local)
    invBT = np.transpose(invB, axes=(0, 2, 1))
    grad_phi_local = np.empty((num_tets, 4, 3), dtype=np.float64)
    grad_phi_local[:, 1:4, :] = invBT
    grad_phi_local[:, 0, :] = -np.sum(invBT, axis=1)
    
    elem_region = np.where(cell_labels == cfg.HEART_MATERIAL_ID, cfg.HEART_MATERIAL_ID, 1).astype(np.int32)
    heart_nodes_indices = np.unique(tets[elem_region == cfg.HEART_MATERIAL_ID].ravel())
    toc(t0_geom)

    return SimpleNamespace(
        points=points, tets=tets, cell_labels=cell_labels, num_nodes=num_nodes,
        num_tets=num_tets, grad_phi=grad_phi_local, volumes=volumes,
        elem_region=elem_region, heart_nodes=heart_nodes_indices,
    )

# ==================================================
# Lead Field Computation
# ==================================================
def compute_leadfield(config_file):
    """
    Computes or loads the lead field matrix C.
    Applies PDF conventions for source terms and uses Gi for C matrix assembly.
    """
    cfg = load_config(config_file)

    actual_precon_method = cfg.PRECON_METHOD.upper()
    if actual_precon_method == 'AMG':
        if not PYAMG_AVAILABLE:
            print("    Warning: Configured PRECON_METHOD='AMG' but pyamg library not found. Falling back to 'JACOBI'.")
            actual_precon_method = 'JACOBI'
    elif actual_precon_method not in ['JACOBI', 'NONE']:
        print(f"    Warning: Unknown PRECON_METHOD '{cfg.PRECON_METHOD}' in config. Using 'JACOBI'.")
        actual_precon_method = 'JACOBI'
    
    print(f"    Using Preconditioner: {actual_precon_method}")

    C = None
    num_leads_from_config = len(cfg.MEASUREMENT_LEADS)

    print("\n[Main] Phase 1: Lead Field Matrix (C) Computation/Loading")
    mesh_data = _load_and_prepare_mesh_data(cfg)
    if mesh_data is None:
        return None, None, 0, 0 # C_matrix, heart_nodes, num_nodes, num_leads

    # Attempt to load precomputed C matrix
    if os.path.exists(cfg.C_MATRIX_FILE):
        t0_cload = tic(f"    Attempting to load precomputed C matrix from {cfg.C_MATRIX_FILE}")
        try:
            C_loaded = np.load(cfg.C_MATRIX_FILE)
            num_leads_loaded, num_nodes_loaded_from_C = C_loaded.shape
            toc(t0_cload)

            if num_leads_loaded == num_leads_from_config and num_nodes_loaded_from_C == mesh_data.num_nodes:
                print(f"          Successfully loaded C matrix: {num_leads_loaded} leads, {num_nodes_loaded_from_C} nodes.")
                C = C_loaded
            else:
                print(f"          Warning: Loaded C matrix dimensions ({num_leads_loaded}L, {num_nodes_loaded_from_C}N) "
                      f"mismatch config/mesh ({num_leads_from_config}L, {mesh_data.num_nodes}N). Recomputing C.")
        except Exception as e:
            toc(t0_cload)
            print(f"          Warning: Could not load C matrix from {cfg.C_MATRIX_FILE} ({e}). Recomputing C.")

    if C is None:
        print("    Full C matrix computation required.")
        
        # --- Step: Building conductivity tensors ---
        t0_cond = tic("    [Step 2] Building conductivity tensors")
        I3 = np.eye(3, dtype=np.float64)
        
        G_bulk_per_elem = np.zeros((mesh_data.num_tets, 3, 3), dtype=np.float64)
        G_bulk_per_elem[mesh_data.elem_region == cfg.HEART_MATERIAL_ID] = cfg.SIGMA_HEART * I3
        G_bulk_per_elem[mesh_data.elem_region != cfg.HEART_MATERIAL_ID] = cfg.SIGMA_TORSO * I3 
        Gi_per_elem = np.zeros((mesh_data.num_tets, 3, 3), dtype=np.float64)
        Gi_per_elem[mesh_data.elem_region == cfg.HEART_MATERIAL_ID] = cfg.SIGMA_HEART * I3

        toc(t0_cond)

        # --- Step: Building lead configuration ---
        t0_lead = tic("    [Step 3] Building lead configuration (source terms for Z)")
        # Stores (positive_node_id, negative_node_id) for each lead based on PDF convention
        LEAD_ELECTRODE_PAIRS = [] 
        for meas_node_idx in cfg.MEASUREMENT_LEADS:
            if cfg.REFERENCE_IS_POSITIVE:
                LEAD_ELECTRODE_PAIRS.append({'positive': cfg.REFERENCE_AND_GROUND_NODE_ID, 'negative': meas_node_idx})
            else:
                LEAD_ELECTRODE_PAIRS.append({'positive': meas_node_idx, 'negative': cfg.REFERENCE_AND_GROUND_NODE_ID})
        
        print("") # Newline for cleaner lead printout
        toc(t0_lead)

        # --- Step: Assembling global stiffness matrix A (for Z computation) ---
        t0_assem = tic("    [Step 4] Assembling global stiffness matrix A (using bulk conductivity)")
        Kloc_all = (mesh_data.grad_phi @ G_bulk_per_elem @ mesh_data.grad_phi.transpose(0, 2, 1)) * mesh_data.volumes[:, None, None]
        # Assemble into sparse matrix A
        II = np.repeat(mesh_data.tets, 4, axis=1)
        JJ = np.tile(mesh_data.tets, (1, 4))
        rows = II.ravel().astype(np.int32)
        cols = JJ.ravel().astype(np.int32)
        data = Kloc_all.ravel()
        
        A = sp.coo_matrix((data, (rows, cols)), shape=(mesh_data.num_nodes, mesh_data.num_nodes)).tocsr()
        
        # Check for symmetry
        symmetry_check_nnz = (A - A.T).nnz
        if symmetry_check_nnz == 0:
            print(f"\n          Symmetry check (A - A.T).nnz: {symmetry_check_nnz} (Matrix is symmetric)")
        else:
            # Check norm of difference for practical symmetry
            diff_norm = spla.norm(A - A.T)
            a_norm = spla.norm(A)
            relative_diff_norm = diff_norm / a_norm if a_norm > 1e-9 else diff_norm
            print(f"\n          Symmetry check (A - A.T).nnz: {symmetry_check_nnz}")
            print(f"          Norm of (A - A.T): {diff_norm:.2e}, Relative norm: {relative_diff_norm:.2e}")
            if relative_diff_norm < 1e-9: # Tolerance for practical symmetry
                 print("          (Matrix is practically symmetric despite minor numerical asymmetries)")
            else:
                 print("          (Matrix is NOT symmetric!)")
        toc(t0_assem)

        # --- Step: Computing scalar lead fields (Z_node) for each lead ---
        t0_solve = tic("    [Step 5/6] Computing scalar lead fields (Z_node)")
        Z_node = np.zeros((num_leads_from_config, mesh_data.num_nodes), dtype=np.float64)

        for lead_idx, lead_pair in enumerate(LEAD_ELECTRODE_PAIRS):
            t1_lead_solve = tic(f"        Processing Lead {lead_idx+1}/{num_leads_from_config} (+N{lead_pair['positive']}, -N{lead_pair['negative']}) using {cfg.SOLVER_TYPE} (Precon: {actual_precon_method})")
            
            # Construct RHS vector b for this lead
            # PDF: Source = -1 at positive electrode, +1 at negative electrode.
            b = np.zeros(mesh_data.num_nodes, dtype=np.float64)
            if 0 <= lead_pair['positive'] < mesh_data.num_nodes:
                b[lead_pair['positive']] = -1.0 
            else:
                print(f"          Warning: Positive electrode node {lead_pair['positive']} for lead {lead_idx+1} is out of bounds ({mesh_data.num_nodes} nodes). Skipping source.")
            
            if 0 <= lead_pair['negative'] < mesh_data.num_nodes:
                b[lead_pair['negative']] = +1.0
            else:
                print(f"          Warning: Negative electrode node {lead_pair['negative']} for lead {lead_idx+1} is out of bounds ({mesh_data.num_nodes} nodes). Skipping source.")

            # Apply grounding (Dirichlet boundary condition)
            A_g, b_g = apply_grounding(A, b, cfg.REFERENCE_AND_GROUND_NODE_ID)
            
            # Setup preconditioner
            M_prec = None
            if actual_precon_method == 'JACOBI':
                diagA_g = A_g.diagonal()
                diagA_g[np.abs(diagA_g) < 1e-15] = 1.0 # Avoid division by zero
                M_prec = sp.diags(1.0 / diagA_g)
                print("              Using Jacobi Preconditioner.")
            elif actual_precon_method == 'AMG' and PYAMG_AVAILABLE:
                print("              Using AMG Preconditioner (pyamg).")
                M_prec = pyamg.smoothed_aggregation_solver(A_g.tocsr(), strength='classical', aggregate='standard').aspreconditioner(cycle='V')
            else: # 'NONE' or AMG not available
                print(f"              Using NO Preconditioner (Method: {actual_precon_method}).")
            
            cb = SolverCallback(cfg.SOLVER_TYPE)
            solution = None; info = -1 # Default error info

            try:
                if cfg.SOLVER_TYPE.upper() == 'GMRES':
                    restart_val = min(30, A_g.shape[0] - 1) if A_g.shape[0] > 1 else 1
                    solution, info = spla.gmres(A_g, b_g, rtol=cfg.SOLVER_TOL, maxiter=cfg.SOLVER_MAXITER, M=M_prec, callback=cb, restart=restart_val)
                elif cfg.SOLVER_TYPE.upper() == 'BICGSTAB':
                    solution, info = spla.bicgstab(A_g, b_g, rtol=cfg.SOLVER_TOL, maxiter=cfg.SOLVER_MAXITER, M=M_prec, callback=cb)
                elif cfg.SOLVER_TYPE.upper() == 'CG':
                    print("              ******* ERROR: CG solver selected, but A_g is non-symmetric due to grounding. CG will likely fail or give incorrect results. Use GMRES or BiCGSTAB. *******")
                    solution, info = spla.cg(A_g, b_g, rtol=cfg.SOLVER_TOL, maxiter=cfg.SOLVER_MAXITER, M=M_prec, callback=cb)

                else:
                    print(f"              Error: Unknown SOLVER_TYPE '{cfg.SOLVER_TYPE}'. Skipping this lead.")
                    toc(t1_lead_solve); continue 
                
                if solution is not None: Z_node[lead_idx] = solution

            except Exception as e:
                print(f"              Solver {cfg.SOLVER_TYPE} failed with exception: {e}"); info = -99 # Custom error code for exception

            final_residual_norm = np.linalg.norm(b_g - A_g @ Z_node[lead_idx]) if solution is not None and hasattr(A_g, 'dot') else float('inf')
            print(f"\n              Solver Info: {info}, Iterations: {cb.niter}, Final Residual Norm: {final_residual_norm:.2e}")
            
            if info != 0:
                if cb.niter >= cfg.SOLVER_MAXITER and info > 0 : 
                    print("              ******* WARNING: Solver reached MAXITER without desired tolerance! *******")
                else:
                    print(f"              ******* WARNING: Solver finished with non-zero info code = {info} *******")
            elif cb.niter == 0 and info == 0: 
                if np.linalg.norm(b_g) < 1e-14 : print("              Converged in 0 iterations (RHS was likely zero).")
                else: print("              Converged in 0 iterations (check problem setup).")
            toc(t1_lead_solve)
        toc(t0_solve)

        # --- Step: Computing element gradients L_fields = grad(Z) for each lead ---
        t0_L = tic("    [Step 7] Computing element gradients L_fields = grad(Z)")
        Z_e = Z_node[:, mesh_data.tets] # (num_leads, num_tets, 4 nodes)
        L_fields = np.einsum('lei,eij->lej', Z_e, mesh_data.grad_phi, optimize=True)
        toc(t0_L)

        # --- Step: Assembling C matrix for ECG: V_ecg = integral( grad(Z) . Gi . grad(Vm) dx ) ---
        t0_C = tic("    [Step 8] Assembling C matrix (using intracellular conductivity Gi)")
        
        M_element_contributions = np.einsum('led,edj,eij,e->lei',
                                             L_fields,          # grad(Z_l) for lead l, element e, dim d
                                             Gi_per_elem,       # Gi for element e, dim d, dim j'
                                             mesh_data.grad_phi,# grad(phi_i) for element e, dim j', local node i
                                             mesh_data.volumes, # d(Vol)_e
                                             optimize=True)

        # Assemble C from M_element_contributions
        M_flat = M_element_contributions.ravel()
        l_idx = np.repeat(np.arange(num_leads_from_config), mesh_data.num_tets * 4) # 4 nodes per tet
        n_idx = np.tile(mesh_data.tets.ravel(), num_leads_from_config) 
        
        C_sparse = sp.coo_matrix((M_flat, (l_idx, n_idx)),
                                 shape=(num_leads_from_config, mesh_data.num_nodes)).tocsr()
        C = C_sparse.toarray()
        toc(t0_C)
        
        t0_save = tic(f"        Saving C matrix to {cfg.C_MATRIX_FILE}")
        try:
            np.save(cfg.C_MATRIX_FILE, C)
            toc(t0_save)
        except Exception as e:
            toc(t0_save)
            print(f"Error: Could not save C matrix to {cfg.C_MATRIX_FILE}: {e}")

    if C is None:
        print("Error: C matrix could not be loaded or computed. Aborting.")
        return None, mesh_data.heart_nodes if mesh_data else None, mesh_data.num_nodes if mesh_data else 0, 0

    return C, mesh_data.heart_nodes, mesh_data.num_nodes, num_leads_from_config

# ==================================================
# ECG Computation
# ==================================================
def compute_ecg(C_matrix, heart_nodes_ids, total_num_nodes, config_file):
    """Computes ECG using the C matrix and Vm time series."""
    cfg = load_config(config_file) # Load configuration for VM_FILE and ECG_FILE

    print("\n[Main] Phase 2: ECG Simulation")

    t0_vm = tic(f"    [Step 9] Loading Vm time series from {cfg.VM_FILE}")
    try:
        vm_data = np.loadtxt(cfg.VM_FILE)
    except FileNotFoundError:
        toc(t0_vm)
        print(f"\nError: Vm input file '{cfg.VM_FILE}' not found.")
        return
    except Exception as e:
        toc(t0_vm)
        print(f"\nError loading Vm data from '{cfg.VM_FILE}': {e}.")
        return

    times = vm_data[:, 0]
    Vmat_raw = vm_data[:, 1:] # Vm values
    n_steps, n_vm_cols = Vmat_raw.shape
    toc(t0_vm)

    V_full_mesh = np.zeros((n_steps, total_num_nodes), dtype=np.float64)

    if n_vm_cols == total_num_nodes:
        print("          Vm data found for all nodes in the mesh.")
        V_full_mesh = Vmat_raw.astype(np.float64)
    elif heart_nodes_ids is not None and n_vm_cols == heart_nodes_ids.size:
        print(f"          Vm data found for {heart_nodes_ids.size} heart nodes. Mapping to full mesh ({total_num_nodes} nodes).")
        if heart_nodes_ids.max() >= total_num_nodes:
            print("          Error: Max heart node ID from Vm mapping exceeds total_num_nodes. Check Vm data and mesh consistency.")
            return
        V_full_mesh[:, heart_nodes_ids] = Vmat_raw.astype(np.float64)
    else:
        print(f"          Error: Vm data columns ({n_vm_cols}) do not match total mesh nodes ({total_num_nodes}) "
              f"or heart_nodes count ({heart_nodes_ids.size if heart_nodes_ids is not None else 'N/A'}). Cannot proceed.")
        return
    
    print(f"          Loaded {n_steps} time steps for Vm.")

    t0_ecg_sim = tic("    [Step 10] Computing ECG trace (Vm @ C.T)")
    ecg_trace = V_full_mesh @ C_matrix.T 
    toc(t0_ecg_sim)

    t0_write = tic(f"    [Step 11] Writing ECG trace to {cfg.ECG_FILE}")
    output_data = np.column_stack((times, ecg_trace))
    # header_str = "time " + " ".join([f"Lead{i+1}" for i in range(ecg_trace.shape[1])])
    try:
        np.savetxt(cfg.ECG_FILE, output_data, fmt="%.6e", comments="") #header=header_str,
        toc(t0_write)
        print(f"[Main] ECG computation complete. Output saved to {cfg.ECG_FILE}")
    except Exception as e:
        toc(t0_write)
        print(f"Error: Could not write ECG output to {cfg.ECG_FILE}: {e}")


