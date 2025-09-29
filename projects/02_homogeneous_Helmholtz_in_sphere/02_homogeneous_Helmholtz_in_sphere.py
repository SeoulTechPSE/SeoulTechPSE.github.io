import numpy as np
import gmsh
from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.io import gmshio
from ufl import TrialFunction, TestFunction, dx, grad, inner
from slepc4py import SLEPc

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def process_eigenvalues(eps, nconv, min_threshold=3.0, degeneracy_tol=0.5, verbose=True):
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
        Tolerance for grouping degenerate eigenvalues, default=0.5
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


# ============================================================================
# MESH GENERATION
# ============================================================================

# Create sphere mesh using Gmsh
gmsh.initialize()
gmsh.model.add("sphere")

# Create sphere (radius 1.0, center at origin)
sphere = gmsh.model.occ.addSphere(0, 0, 0, 1)
gmsh.model.occ.synchronize()

# Define physical groups
gmsh.model.addPhysicalGroup(3, [sphere], 1)
gmsh.model.setPhysicalName(3, 1, "Sphere")

# Define boundary (surface of sphere)
boundary = gmsh.model.getBoundary([(3, sphere)], oriented=False)
boundary_tags = [b[1] for b in boundary]
gmsh.model.addPhysicalGroup(2, boundary_tags, 1)
gmsh.model.setPhysicalName(2, 1, "Boundary")

# Set mesh size (finer mesh for better accuracy)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.10)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.10)

# Generate mesh
gmsh.model.mesh.generate(3)

# Import mesh to FEniCSx
domain, cell_markers, facet_markers = gmshio.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, 0, gdim=3
)

gmsh.finalize()

# ============================================================================
# FINITE ELEMENT SETUP
# ============================================================================

# Define function space (P2 Lagrange elements)
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

# ============================================================================
# EIGENVALUE SOLVER CONFIGURATION
# ============================================================================

# Configure SLEPc eigenvalue solver
eps = SLEPc.EPS().create(MPI.COMM_WORLD)
eps.setOperators(A, B)
eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)

# Use shift-and-invert method
st = eps.getST()
st.setType(SLEPc.ST.Type.SINVERT)

eps.setDimensions(nev=60, ncv=120)
eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
eps.setTarget(40.0)  # Target around 40 for better coverage
eps.setFromOptions()

# Solve eigenvalue problem
if MPI.COMM_WORLD.rank == 0:
    print("\n" + "="*70)
    print("EIGENVALUE PROBLEM ON UNIT SPHERE")
    print("="*70)
    print("\nComputing eigenvalues...")
eps.solve()

# ============================================================================
# PROCESS AND ANALYZE RESULTS
# ============================================================================

# Process eigenvalue results
nconv = eps.getConverged()
if MPI.COMM_WORLD.rank == 0:
    print(f"\nNumber of converged eigenvalues: {nconv}")
    print("Eigenvalues for unit sphere:")
    print("-" * 70)

grouped_eigenvalues, eigenvalues = process_eigenvalues(eps, nconv, degeneracy_tol=0.5)

# ============================================================================
# THEORETICAL VALUES AND COMPARISON
# ============================================================================

if MPI.COMM_WORLD.rank == 0:
    print("\n" + "="*70)
    print("THEORETICAL COMPARISON")
    print("="*70)
    print("\nTheoretical eigenvalues for unit sphere:")
    print("λ = j²_{l+1/2,n} where j_{l+1/2,n} is spherical Bessel zero")
    print("Degeneracy is (2l+1) for each (l,n) mode")
    print("(l: angular momentum, n: radial quantum number)\n")
    
    print("Accuracy comparison (first 10 unique modes):\n")
    
    # Theoretical values: (eigenvalue, degeneracy, label)
    theoretical = [
        ( 9.869604,  1, "(l=0, n=1)"), 
        (20.190729,  3, "(l=1, n=1)"),   
        (33.217462,  5, "(l=2, n=1)"),   
        (39.478418,  1, "(l=0, n=2)"),   
        (48.831194,  7, "(l=3, n=1)"),   
        (59.679516,  3, "(l=1, n=2)"),   
        (66.954312,  9, "(l=4, n=1)"),   
        (82.719231,  5, "(l=2, n=2)"),   
        (87.531220, 11, "(l=5, n=1)"),   
        (88.826440,  1, "(l=0, n=3)"),   
    ]
    
    for idx, (eigval, deg, _) in enumerate(grouped_eigenvalues[:min(10, len(grouped_eigenvalues))]):
        if idx < len(theoretical):
            theo_val, theo_deg, mode_label = theoretical[idx]
            error = abs(eigval - theo_val) / theo_val * 100
            deg_match = "✓" if abs(deg - theo_deg) <= 2 else "✗"
            if error > 2.0:
                print(f"  Mode {idx+1:2d} {mode_label}: computed = {eigval:8.2f} (×{deg:2d}), theoretical = {theo_val:8.2f} (×{theo_deg:2d}), error = {error:5.2f}% {deg_match} ⚠️")
            else:
                print(f"  Mode {idx+1:2d} {mode_label}: computed = {eigval:8.2f} (×{deg:2d}), theoretical = {theo_val:8.2f} (×{theo_deg:2d}), error = {error:5.2f}% {deg_match}")

# ============================================================================
# SAVE EIGENFUNCTIONS
# ============================================================================

if len(grouped_eigenvalues) > 0:
    from dolfinx.io import VTKFile
    
    num_modes = min(10, len(grouped_eigenvalues))
    
    if MPI.COMM_WORLD.rank == 0:
        print(f"\n{'='*70}")
        print(f"SAVING EIGENFUNCTIONS")
        print(f"{'='*70}")
        print(f"\nSaving first {num_modes} unique eigenfunctions to VTK files...")
    
    for idx in range(num_modes):
        eigval, deg, indices = grouped_eigenvalues[idx]
        
        # Get first eigenfunction from the group
        vr, vi = A.createVecs()
        eps.getEigenpair(indices[0], vr, vi)
        
        u_eigen = fem.Function(V)
        u_eigen.x.array[:] = vr.array
        
        if deg > 1:
            u_eigen.name = f"Eigenmode_{idx+1}_deg{deg}"
        else:
            u_eigen.name = f"Eigenmode_{idx+1}"
        
        with VTKFile(MPI.COMM_WORLD, f"sphere_eigenmode_{idx+1}.pvd", "w") as vtk:
            vtk.write_function(u_eigen)
    
    if MPI.COMM_WORLD.rank == 0:
        print(f"Saved {num_modes} eigenfunctions successfully.")
        print("\nVisualization instructions:")
        print("  1. Open with ParaView: paraview sphere_eigenmode_*.pvd")
        print("  2. Recommended filters:")
        print("     - 'Slice' filter: view cross-sections")
        print("     - 'Contour' filter: show isosurfaces")
        print("     - 'Warp By Scalar': deform surface by eigenfunction values")
        print("  3. Color mapping: use eigenfunction values with diverging colormap")

# ============================================================================
# SUMMARY
# ============================================================================

if MPI.COMM_WORLD.rank == 0:
    print("\n" + "="*70)
    print("COMPUTATION COMPLETE")
    print("="*70)
    print(f"\nSummary:")
    print(f"  - Converged eigenvalues: {nconv}")
    print(f"  - Unique modes (grouped): {len(grouped_eigenvalues)}")
    print(f"  - VTK files saved: {num_modes}")
    print(f"\nNote: 3D eigenfunctions are best visualized in ParaView.")
    print(f"      Use slice and contour filters to explore the 3D structure.")
    print("="*70 + "\n")