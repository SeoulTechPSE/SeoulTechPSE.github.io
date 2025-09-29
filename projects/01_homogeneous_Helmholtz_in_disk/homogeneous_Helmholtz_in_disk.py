import numpy as np
import gmsh
from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.io import gmshio
from ufl import TrialFunction, TestFunction, dx, grad, inner
from slepc4py import SLEPc
from petsc4py import PETSc

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def process_eigenvalues(eps, nconv, min_threshold=3.0, degeneracy_tol=0.2, verbose=True):
    """Process eigenvalues from SLEPc solver: filter, sort, and group degenerate modes
    
    Parameters
    ----------
    eps : SLEPc.EPS
        Solved eigenvalue problem
    nconv : int
        Number of converged eigenvalues
    min_threshold : float, optional
        Minimum eigenvalue to consider (filters spurious modes), default=3.0
    degeneracy_tol : float, optional
        Tolerance for grouping degenerate eigenvalues, default=0.2
    verbose : bool, optional
        Print detailed output, default=True
        
    Returns
    -------
    tuple
        (grouped_eigenvalues, raw_eigenvalues)
        - grouped_eigenvalues: list of (eigenvalue, degeneracy, indices) tuples
        - raw_eigenvalues: list of (eigenvalue, index) tuples for all filtered values
    """
    # Collect raw eigenvalues
    raw_eigenvalues = []
    if verbose and MPI.COMM_WORLD.rank == 0:
        print(f"\nRaw eigenvalues from solver (all {nconv}):\n")
    
    for i in range(nconv):
        eigval = eps.getEigenvalue(i)
        if verbose and MPI.COMM_WORLD.rank == 0:
            print(f"  {i+1:2d}: {eigval.real:10.6f}")
        # Filter physically meaningful eigenvalues
        if eigval.real > min_threshold:
            raw_eigenvalues.append((eigval.real, i))
    
    # Sort eigenvalues by magnitude
    raw_eigenvalues.sort(key=lambda x: x[0])
    
    # Group degenerate eigenvalues
    grouped_eigenvalues = []
    i = 0
    while i < len(raw_eigenvalues):
        eigval, idx = raw_eigenvalues[i]
        degeneracy = 1
        indices = [idx]
        
        # Check for degenerate eigenvalues
        j = i + 1
        while j < len(raw_eigenvalues) and abs(raw_eigenvalues[j][0] - eigval) < degeneracy_tol:
            degeneracy += 1
            indices.append(raw_eigenvalues[j][1])
            j += 1
        
        grouped_eigenvalues.append((eigval, degeneracy, indices))
        i = j
    
    if verbose and MPI.COMM_WORLD.rank == 0:
        print(f"\nFiltered eigenvalues (> {min_threshold}): {len(raw_eigenvalues)}")
        print(f"Total unique modes found after grouping: {len(grouped_eigenvalues)}")
        print("\nGrouped eigenvalues:\n")
        for idx, (eigval, deg, _) in enumerate(grouped_eigenvalues[:20]):
            if deg > 1:
                print(f"λ_{idx+1:2d} = {eigval:10.6f}  (degeneracy: {deg})")
            else:
                print(f"λ_{idx+1:2d} = {eigval:10.6f}")
    
    return grouped_eigenvalues, raw_eigenvalues


def create_symmetric_eigenfunction(V, A, eps, eigval, deg, indices):
    """Create y-axis symmetric eigenfunction from eigenvalue data
    
    For degenerate eigenvalues, finds the optimal linear combination of the 
    degenerate eigenfunctions that exhibits y-axis symmetry u(x,y) = u(-x,y).
    For non-degenerate cases, returns the eigenfunction with normalized sign.
    
    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
        Function space for the eigenfunctions
    A : PETSc.Mat
        Stiffness matrix (used to create vectors)
    eps : SLEPc.EPS
        Solved eigenvalue problem
    eigval : float
        Eigenvalue for this mode
    deg : int
        Degeneracy of the eigenvalue (1 for non-degenerate, 2+ for degenerate)
    indices : list of int
        Indices of the eigenvalue/eigenvector pairs from the solver
        
    Returns
    -------
    dolfinx.fem.Function
        Y-axis symmetric eigenfunction with normalized sign (positive on upper y-axis)
    """
    if deg > 1 and len(indices) >= 2:
        # Get two degenerate eigenvectors
        vr1, vi1 = A.createVecs()
        vr2, vi2 = A.createVecs()
        eps.getEigenpair(indices[0], vr1, vi1)
        eps.getEigenpair(indices[1], vr2, vi2)
        
        u1 = fem.Function(V)
        u2 = fem.Function(V)
        u1.x.array[:] = vr1.array
        u2.x.array[:] = vr2.array
        
        x = V.tabulate_dof_coordinates()
        from scipy.spatial import cKDTree
        tree = cKDTree(x[:, :2])
        
        # Find best linear combination for y-axis symmetry
        best_combo = u1.x.array.copy()
        best_symmetry = float('inf')
        
        for alpha in np.linspace(0, 2*np.pi, 36):
            test_array = np.cos(alpha) * u1.x.array + np.sin(alpha) * u2.x.array
            
            symmetry_error = 0
            count = 0
            for i in range(len(x)):
                if abs(x[i, 0]) > 0.05:
                    mirror_point = np.array([-x[i, 0], x[i, 1]])
                    dist, j = tree.query(mirror_point)
                    if dist < 0.1:
                        symmetry_error += (test_array[i] - test_array[j])**2
                        count += 1
            
            if count > 0:
                symmetry_error = np.sqrt(symmetry_error / count)
                if symmetry_error < best_symmetry:
                    best_symmetry = symmetry_error
                    best_combo = test_array.copy()
        
        u_eigen = fem.Function(V)
        u_eigen.x.array[:] = best_combo
        
        # Enforce perfect symmetry
        for i in range(len(x)):
            if x[i, 0] > 0.01:
                mirror_point = np.array([-x[i, 0], x[i, 1]])
                dist, j = tree.query(mirror_point)
                if dist < 0.05:
                    avg_val = (u_eigen.x.array[i] + u_eigen.x.array[j]) / 2
                    u_eigen.x.array[i] = avg_val
                    u_eigen.x.array[j] = avg_val
    else:
        # Non-degenerate eigenfunction
        vr, vi = A.createVecs()
        eps.getEigenpair(indices[0], vr, vi)
        u_eigen = fem.Function(V)
        u_eigen.x.array[:] = vr.array
    
    # Normalize sign (positive on upper y-axis)
    x = V.tabulate_dof_coordinates()
    y_axis_mask = (np.abs(x[:, 0]) < 0.05) & (x[:, 1] > 0.5)
    if np.any(y_axis_mask):
        y_axis_sum = np.sum(u_eigen.x.array[y_axis_mask])
        if y_axis_sum < 0:
            u_eigen.x.array[:] *= -1
    
    return u_eigen


# ============================================================================
# MAIN CODE: EIGENVALUE PROBLEM ON UNIT DISK
# ============================================================================

# Create disk mesh using Gmsh
gmsh.initialize()
gmsh.model.add("disk")

# Create disk (radius 1.0, center at origin)
disk = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
gmsh.model.occ.synchronize()

# Define physical groups
gmsh.model.addPhysicalGroup(2, [disk], 1)
gmsh.model.setPhysicalName(2, 1, "Disk")

# Define boundary
boundary = gmsh.model.getBoundary([(2, disk)], oriented=False)
boundary_tags = [b[1] for b in boundary]
gmsh.model.addPhysicalGroup(1, boundary_tags, 1)
gmsh.model.setPhysicalName(1, 1, "Boundary")

# Set mesh size (coarser mesh with P2 elements)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.04)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.04)

# Generate mesh
gmsh.model.mesh.generate(2)

# Import mesh to FEniCSx
domain, cell_markers, facet_markers = gmshio.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, 0, gdim=2
)

gmsh.finalize()

# Define function space (P2 Lagrange elements for better accuracy)
V = fem.functionspace(domain, ("Lagrange", 2))

# Define boundary condition (Dirichlet: u = 0)
boundary_facets = facet_markers.find(1)
boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V)

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Assemble stiffness and mass matrices
# Eigenvalue problem: -Δu = λu
a_form = fem.form(inner(grad(u), grad(v)) * dx)
b_form = fem.form(inner(u, v) * dx)

# Assemble matrices with boundary conditions
A = assemble_matrix(a_form, bcs=[bc])
A.assemble()

B = assemble_matrix(b_form, bcs=[bc])
B.assemble()

# Configure SLEPc eigenvalue solver
eps = SLEPc.EPS().create(MPI.COMM_WORLD)
eps.setOperators(A, B)
eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)

# Use shift-and-invert method to find eigenvalues near a target
st = eps.getST()
st.setType(SLEPc.ST.Type.SINVERT)

eps.setDimensions(nev=50, ncv=100)
eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
eps.setTarget(60.0)
eps.setFromOptions()

# Solve eigenvalue problem
if MPI.COMM_WORLD.rank == 0:
    print("\nComputing eigenvalues...")
eps.solve()

# Process eigenvalue results
nconv = eps.getConverged()
if MPI.COMM_WORLD.rank == 0:
    print(f"\nNumber of converged eigenvalues: {nconv}")
    print("Eigenvalues (theoretical values are squares of Bessel zeros):")
    print("-" * 50)

grouped_eigenvalues, eigenvalues = process_eigenvalues(eps, nconv)

# Print theoretical comparison
if MPI.COMM_WORLD.rank == 0:
    print("\nTheoretical eigenvalues for disk:\n")
    print("(m,n): j_{01}^2 =  5.783 (×1),  j_{11}^2 = 14.682 (×2),  j_{21}^2 = 26.375 (×2)")
    print("       j_{02}^2 = 30.471 (×1),  j_{31}^2 = 40.706 (×2),  j_{12}^2 = 49.218 (×2)")
    print("       j_{41}^2 = 57.583 (×2),  j_{22}^2 = 70.850 (×2),  j_{03}^2 = 74.887 (×1)")
    print("       j_{51}^2 = 76.939 (×2),  j_{32}^2 = 95.278 (×2),  j_{61}^2 = 98.726 (×2)")
    print("in which j_{mn} denotes m-th angular mode with n radial nodes")
    
    print("\nAccuracy comparison (first 12 modes):\n")
    
    # Theoretical values
    theoretical = [
        (5.783185962947,  1, "(0,1)"),   # j_{01}^2
        (14.681970642124, 2, "(1,1)"),   # j_{11}^2
        (26.374616427163, 2, "(2,1)"),   # j_{21}^2
        (30.471262343662, 1, "(0,2)"),   # j_{02}^2
        (40.706465818200, 2, "(3,1)"),   # j_{31}^2
        (49.218456321695, 2, "(1,2)"),   # j_{12}^2
        (57.582940903291, 2, "(4,1)"),   # j_{41}^2
        (70.849998919096, 2, "(2,2)"),   # j_{22}^2
        (74.887006790695, 1, "(0,3)"),   # j_{03}^2
        (76.938928333647, 2, "(5,1)"),   # j_{51}^2
        (95.277572544037, 2, "(3,2)"),   # j_{32}^2
        (98.726272477249, 2, "(6,1)")    # j_{61}^2
    ]
    
    for idx, (eigval, deg, _) in enumerate(grouped_eigenvalues[:min(12, len(grouped_eigenvalues))]):
        if idx < len(theoretical):
            theo_val, theo_deg, mode_label = theoretical[idx]
            error = abs(eigval - theo_val) / theo_val * 100
            deg_match = "✓" if deg == theo_deg else "✗"
            if error > 1.0:
                print(f"  Mode {idx+1:2d} {mode_label}: computed = {eigval:7.3f} (×{deg}), theoretical = {theo_val:7.3f} (×{theo_deg}), error = {error:5.2f}% {deg_match} ⚠️")
            else:
                print(f"  Mode {idx+1:2d} {mode_label}: computed = {eigval:7.3f} (×{deg}), theoretical = {theo_val:7.3f} (×{theo_deg}), error = {error:5.2f}% {deg_match}")

# Save and visualize eigenfunctions
if len(grouped_eigenvalues) > 0:
    from dolfinx.io import VTKFile
    
    num_modes = min(12, len(grouped_eigenvalues))  #**********
    
    # Compute all eigenfunctions once and store them
    eigenfunctions = []
    for idx in range(num_modes):
        eigval, deg, indices = grouped_eigenvalues[idx]
        u_eigen = create_symmetric_eigenfunction(V, A, eps, eigval, deg, indices)
        eigenfunctions.append((eigval, deg, u_eigen))
    
    # Save to VTK files
    for idx, (eigval, deg, u_eigen) in enumerate(eigenfunctions):
        if deg > 1:
            u_eigen.name = f"Eigenmode_{idx+1}_deg{deg}"
        else:
            u_eigen.name = f"Eigenmode_{idx+1}"
        
        with VTKFile(MPI.COMM_WORLD, f"eigenmode_{idx+1}.pvd", "w") as vtk:
            vtk.write_function(u_eigen)
    
    if MPI.COMM_WORLD.rank == 0:
        print(f"\nSaved first {num_modes} unique eigenfunctions to VTK files.")
        print("Visualize with ParaView: paraview eigenmode_*.pvd")
    
    # Visualize using PyVista
    if MPI.COMM_WORLD.rank == 0:
        try:
            import pyvista as pv
            from dolfinx import plot
            
            print("\nGenerating PyVista visualizations...")
            
            pv.OFF_SCREEN = True
            
            # Individual eigenmode images
            for idx, (eigval, deg, u_eigen) in enumerate(eigenfunctions):
                topology, cell_types, geometry = plot.vtk_mesh(V)
                grid = pv.UnstructuredGrid(topology, cell_types, geometry)
                grid.point_data["u"] = u_eigen.x.array.real
                
                plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
                plotter.add_mesh(grid, show_edges=False, scalars="u", 
                               cmap="RdBu_r", clim=[-abs(u_eigen.x.array).max(), 
                                                     abs(u_eigen.x.array).max()],
                               scalar_bar_args={'title': 'Amplitude', 
                                              'vertical': True,
                                              'position_x': 0.85,
                                              'position_y': 0.1,
                                              'width': 0.08,
                                              'height': 0.8})
                plotter.view_xy()
                title = f"Eigenmode {idx+1}: lambda = {eigval:.3f}"
                if deg > 1:
                    title += f" (degeneracy: {deg})"
                plotter.add_title(title, font_size=14)
                
                plotter.screenshot(f"eigenmode_{idx+1}.png", transparent_background=False)
                plotter.close()
            
            print(f"Saved {num_modes} unique eigenmode images as eigenmode_*.png")
            
            # Comparison plot
            plotter = pv.Plotter(shape=(3, 4), off_screen=True, window_size=[2000, 1500], border=False)
            
            for idx, (eigval, deg, u_eigen) in enumerate(eigenfunctions):
                topology, cell_types, geometry = plot.vtk_mesh(V)
                grid = pv.UnstructuredGrid(topology, cell_types, geometry)
                grid.point_data["u"] = u_eigen.x.array.real
                
                row = idx // 4
                col = idx % 4
                plotter.subplot(row, col)
                plotter.add_mesh(grid, show_edges=False, scalars="u", 
                               cmap="RdBu_r", clim=[-abs(u_eigen.x.array).max(), 
                                                     abs(u_eigen.x.array).max()],
                               show_scalar_bar=False)
                plotter.view_xy()
                title = f"Mode {idx+1}: lambda={eigval:.2f}"
                if deg > 1:
                    title += f" (×{deg})"
                plotter.add_title(title, font_size=12)
            
            plotter.screenshot("eigenmodes_comparison.png", transparent_background=False)
            plotter.close()
            
            print("Saved comparison plot as eigenmodes_comparison.png")
            
        except ImportError:
            print("\nPyVista not available. Install with: pip install pyvista")
        except Exception as e:
            print(f"\nWarning: Could not create PyVista visualization: {e}")