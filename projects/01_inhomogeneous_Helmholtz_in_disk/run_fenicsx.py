import numpy as np
import gmsh
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.io import gmshio
from ufl import TrialFunction, TestFunction, dx, grad, inner
from slepc4py import SLEPc
from petsc4py import PETSc

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
    print("Computing eigenvalues...")
eps.solve()

# Process results and group degenerate eigenvalues
nconv = eps.getConverged()
if MPI.COMM_WORLD.rank == 0:
    print(f"Number of converged eigenvalues: {nconv}")
    print("\nEigenvalues (theoretical values are squares of Bessel zeros):")
    print("-" * 50)

eigenvalues = []
if MPI.COMM_WORLD.rank == 0:
    print(f"\nRaw eigenvalues from solver (all {nconv}):")
for i in range(nconv):
    eigval = eps.getEigenvalue(i)
    if MPI.COMM_WORLD.rank == 0:
        print(f"  {i+1:2d}: {eigval.real:10.6f}")
    # Select only physically meaningful eigenvalues (> 3.0 to avoid spurious modes)
    if eigval.real > 3.0:
        eigenvalues.append((eigval.real, i))

# Sort eigenvalues by magnitude
eigenvalues.sort(key=lambda x: x[0])

# Group degenerate eigenvalues (within tolerance)
grouped_eigenvalues = []
tol = 0.2
i = 0
while i < len(eigenvalues):
    eigval, idx = eigenvalues[i]
    degeneracy = 1
    indices = [idx]
    
    # Check for degenerate eigenvalues
    j = i + 1
    while j < len(eigenvalues) and abs(eigenvalues[j][0] - eigval) < tol:
        degeneracy += 1
        indices.append(eigenvalues[j][1])
        j += 1
    
    grouped_eigenvalues.append((eigval, degeneracy, indices))
    i = j

if MPI.COMM_WORLD.rank == 0:
    print(f"\nFiltered eigenvalues (> 3.0): {len(eigenvalues)}")
    print(f"Total unique modes found after grouping: {len(grouped_eigenvalues)}")
    print("\nGrouped eigenvalues:")
    for idx, (eigval, deg, _) in enumerate(grouped_eigenvalues[:20]):
        if deg > 1:
            print(f"λ_{idx+1:2d} = {eigval:10.6f}  (degeneracy: {deg})")
        else:
            print(f"λ_{idx+1:2d} = {eigval:10.6f}")

if MPI.COMM_WORLD.rank == 0:
    print("\nTheoretical eigenvalues for disk (reference from paper):")
    print("Note: j_{mn} denotes m-th angular mode with n radial nodes")
    print("(m,n): j_{01}^2 =  5.783,  j_{11}^2 = 14.682 (×2),  j_{21}^2 = 26.375 (×2)")
    print("       j_{02}^2 = 30.471,  j_{31}^2 = 40.706 (×2),  j_{12}^2 = 49.218 (×2)")
    print("       j_{41}^2 = 57.583 (×2),  j_{22}^2 = 70.850 (×2),  j_{03}^2 = 74.887 (×1)")
    print("       j_{51}^2 = 76.939 (×2),  j_{32}^2 = 95.278 (×2),  j_{61}^2 = 98.726 (×2)")
    print("\nAccuracy comparison (first 12 modes):")
    
    # Theoretical values from paper (exact)
    theoretical = [
        (5.783185962947, 1, "(0,1)"),    # j_{01}^2
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
                print(f"  Mode {idx+1:2d} {mode_label}: computed = {eigval:9.3f} (×{deg}), theoretical = {theo_val:9.3f} (×{theo_deg}), error = {error:5.2f}% {deg_match} ⚠️")
            else:
                print(f"  Mode {idx+1:2d} {mode_label}: computed = {eigval:9.3f} (×{deg}), theoretical = {theo_val:9.3f} (×{theo_deg}), error = {error:5.2f}% {deg_match}")

# Save eigenfunctions to VTK files (only unique modes)
if len(grouped_eigenvalues) > 0:
    from dolfinx.io import VTKFile
    
    num_modes = min(10, len(grouped_eigenvalues))
    for idx in range(num_modes):
        eigval, deg, indices = grouped_eigenvalues[idx]
        
        # For degenerate eigenvalues, find the y-axis symmetric combination
        if deg > 1 and len(indices) >= 2:
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
            
            # Explicitly enforce symmetry
            for i in range(len(x)):
                if x[i, 0] > 0.01:
                    mirror_point = np.array([-x[i, 0], x[i, 1]])
                    dist, j = tree.query(mirror_point)
                    if dist < 0.05:
                        avg_val = (u_eigen.x.array[i] + u_eigen.x.array[j]) / 2
                        u_eigen.x.array[i] = avg_val
                        u_eigen.x.array[j] = avg_val
        else:
            vr, vi = A.createVecs()
            eps.getEigenpair(indices[0], vr, vi)
            u_eigen = fem.Function(V)
            u_eigen.x.array[:] = vr.array
        
        # Normalize sign
        x = V.tabulate_dof_coordinates()
        y_axis_mask = (np.abs(x[:, 0]) < 0.05) & (x[:, 1] > 0.5)
        if np.any(y_axis_mask):
            y_axis_sum = np.sum(u_eigen.x.array[y_axis_mask])
            if y_axis_sum < 0:
                u_eigen.x.array[:] *= -1
        
        if deg > 1:
            u_eigen.name = f"Eigenmode_{idx+1}_deg{deg}"
        else:
            u_eigen.name = f"Eigenmode_{idx+1}"
        
        with VTKFile(MPI.COMM_WORLD, f"eigenmode_{idx+1}.pvd", "w") as vtk:
            vtk.write_function(u_eigen)
    
    if MPI.COMM_WORLD.rank == 0:
        print(f"\nSaved first {num_modes} unique eigenfunctions to VTK files.")
        print("Visualize with ParaView: paraview eigenmode_1.pvd")

# Visualize eigenfunctions using PyVista
if len(grouped_eigenvalues) > 0 and MPI.COMM_WORLD.rank == 0:
    try:
        import pyvista as pv
        from dolfinx import plot
        
        print("\nGenerating PyVista visualizations...")
        
        pv.OFF_SCREEN = True
        
        num_viz = min(10, len(grouped_eigenvalues))
        
        for idx in range(num_viz):
            eigval, deg, indices = grouped_eigenvalues[idx]
            
            if deg > 1 and len(indices) >= 2:
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
                
                for i in range(len(x)):
                    if x[i, 0] > 0.01:
                        mirror_point = np.array([-x[i, 0], x[i, 1]])
                        dist, j = tree.query(mirror_point)
                        if dist < 0.05:
                            avg_val = (u_eigen.x.array[i] + u_eigen.x.array[j]) / 2
                            u_eigen.x.array[i] = avg_val
                            u_eigen.x.array[j] = avg_val
            else:
                vr, vi = A.createVecs()
                eps.getEigenpair(indices[0], vr, vi)
                u_eigen = fem.Function(V)
                u_eigen.x.array[:] = vr.array
            
            x = V.tabulate_dof_coordinates()
            y_axis_mask = (np.abs(x[:, 0]) < 0.05) & (x[:, 1] > 0.5)
            if np.any(y_axis_mask):
                y_axis_sum = np.sum(u_eigen.x.array[y_axis_mask])
                if y_axis_sum < 0:
                    u_eigen.x.array[:] *= -1
            
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
            plotter.add_title(title, font_size=16)
            
            plotter.screenshot(f"eigenmode_{idx+1}.png", transparent_background=False)
            plotter.close()
        
        print(f"Saved {num_viz} unique eigenmode images as eigenmode_*.png")
        
        plotter = pv.Plotter(shape=(2, 5), off_screen=True, window_size=[2400, 1000])
        
        for idx in range(min(10, num_viz)):
            eigval, deg, indices = grouped_eigenvalues[idx]
            
            if deg > 1 and len(indices) >= 2:
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
                
                for i in range(len(x)):
                    if x[i, 0] > 0.01:
                        mirror_point = np.array([-x[i, 0], x[i, 1]])
                        dist, j = tree.query(mirror_point)
                        if dist < 0.05:
                            avg_val = (u_eigen.x.array[i] + u_eigen.x.array[j]) / 2
                            u_eigen.x.array[i] = avg_val
                            u_eigen.x.array[j] = avg_val
            else:
                vr, vi = A.createVecs()
                eps.getEigenpair(indices[0], vr, vi)
                u_eigen = fem.Function(V)
                u_eigen.x.array[:] = vr.array
            
            x = V.tabulate_dof_coordinates()
            y_axis_mask = (np.abs(x[:, 0]) < 0.05) & (x[:, 1] > 0.5)
            if np.any(y_axis_mask):
                y_axis_sum = np.sum(u_eigen.x.array[y_axis_mask])
                if y_axis_sum < 0:
                    u_eigen.x.array[:] *= -1
            
            topology, cell_types, geometry = plot.vtk_mesh(V)
            grid = pv.UnstructuredGrid(topology, cell_types, geometry)
            grid.point_data["u"] = u_eigen.x.array.real
            
            row = idx // 5
            col = idx % 5
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