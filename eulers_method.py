# December 17th, 2023
# minimizes need for small angle approximation (uses multiple steps/iterations to determine solution)
# uses Euler's method
# increase number of iterations for more accuracy
# Solves for displacements, given a force.

import numpy as np

# Change dataname to any linear or nonlinear input file! 
dataname = "nonLinear3DTetrahedron.txt"
iterations = 100

def split_cols(lines, index):
    data = []
    while index < len(lines):
        if lines[index].strip():
            columns = lines[index].split()
            data.append(columns)
            index += 1
        else:
            break
    return data

with open(dataname, 'r') as file: # 2d or 3d input text file
    lines = file.readlines()

lineno = 0
nodelm = 2 # number of nodes per element
for line in lines:
    if "Number of spatial dimensions:" in line:
        ndim = int(lines[lineno + 1])
        ndpn = ndim
    elif "Number of joints/nodes:" in line:
        nodes = int(lines[lineno + 1])
    elif "Node #, x-location, y-location, (z-location if 3D)" in line:
        node_data = []
        index = lineno + 1
        data = split_cols(lines, index)
        for columns in data:
            nodeno = int(columns[0])
            if ndim == 2:
                x, y= float(columns[1]), float(columns[2])
                node_data.append([nodeno, x, y])
            else:
                x, y, z = float(columns[1]), float(columns[2]), float(columns[3])
                node_data.append([nodeno, x, y, z])
        node = np.array(node_data)
        node_pos = node[:,1:]
    elif "Number of bars/elements" in line:
        neles = int(lines[lineno + 1])
    # elif "Number of local nodes:" in line:
    #     nlnodes = int(lines[lineno+1])
    elif "Element#" in line:
        ele_data = []
        ele = np.ndarray((neles,5), dtype = object)
        index = lineno + 1
        data = split_cols(lines, index)
        for columns in data:
            element_number = int(columns[0])
            # for i in range(1,nlnodes+1):
            #     local_nodes
            local_node1 = int(columns[1])
            local_node2 = int(columns[2])
            youngs_modulus = float(columns[3])
            area = float(columns[4])
            ele_data.append([element_number, local_node1, local_node2, youngs_modulus, area])
        ele = np.array(ele_data)#, dtype=[('element_number', int), ('local_node1', int), ('local_node2', int), ('youngs_modulus', float), ('area', float)])
        local_nodes_arr = ele[:,1:3].astype(int)-1
        E = ele[:,3]
        A = ele[:,4]
    elif "Number of applied forces" in line:
        nfbcs = int(lines[lineno + 1])
    elif "Node#, Force direction, Force value" in line:
        fnode = []
        fdof = []
        fval = []
        force = []
        index = lineno + 1
        data = split_cols(lines, index)
        for columns in data:
            fnode.append(int(columns[0]))
            fdof.append(int(columns[1]))
            fval.append(float(columns[2]))
            force.append([int(columns[0]), int(columns[1]), float(columns[2])]) # array of forces info
        force_index_bc = np.array((fnode, fdof))-1
        force_values = np.array(fval)
    elif "Number of known/applied displacements" in line:
        ndbcs = int(lines[lineno + 1])
    elif "Node#, Displacement direction, Displacement value" in line:
        index = lineno + 1
        data = split_cols(lines, index)
        disp_bc=[]
        dbcdof=[]
        disp_val_bc=[]
        dbc=[] # array of displacment info
        for columns in data:
            disp_bc.append(int(columns[0]))
            dbcdof.append(int(columns[1]))
            disp_val_bc.append(float(columns[2]))
            dbc.append([int(columns[0]), int(columns[1]), float(columns[2])])
        disp_bc = np.array((disp_bc, dbcdof))-1
        disp_val_bc = np.array(disp_val_bc)
    lineno += 1

# Initialize length for bars, cosine/direction matrix, 

coord_1 = node[local_nodes_arr[:,0], 1:]
coord_2 = node[local_nodes_arr[:,1], 1:]
element_dir = coord_2 - coord_1 # n1 to n2 vector
L = np.linalg.norm(element_dir, axis =1)
strain_constant = E*A/L

def calculate(node, strain_constant):
    coord_1 = node[local_nodes_arr[:,0], 1:]
    coord_2 = node[local_nodes_arr[:,1], 1:]
    element_dir = coord_2 - coord_1 # n1 to n2 vector
    currentL = np.linalg.norm(element_dir, axis =1)

    # finding angles/basis (B) matrix
    normalized_dir = element_dir/currentL[:,np.newaxis] # n1 basis, element component to nodes
    angles = np.stack([-normalized_dir, normalized_dir]) # local nodes to element
    # connectivity: global nodes to local nodes
    con = np.zeros((neles, nodelm, nodes)) # nodes per element, total nodes, total elements
    ele_index = np.arange(len(local_nodes_arr))[:, np.newaxis].repeat(nodelm, axis = 1)
    node_index = np.arange(nodelm)[np.newaxis, :]
    con[ele_index, node_index, local_nodes_arr] = 1

    # Basis (B) matrix
    B = np.einsum("med, emn -> edn", angles, con) # nodelm(m), elements(e), ndims(d) x elements(e), nodelm (m), nodes (n)

    # Stiffness (K) matrix
    K = np.einsum("edn, e, ebm->ndmb",B,strain_constant,B) # nodes (n or m), elements(e), axis/ndim (d or b)
    K = np.reshape(K, (ndim*nodes, ndim*nodes)) # per row, node1x node1y  node2x node2y
  
    # Known Nodal Displacements (Indexes) 
    kdi = disp_bc[0]*ndim+disp_bc[1] # k = known, u = unknown, di = displacement indexes
    udi = np.setdiff1d(np.arange(ndim*nodes), kdi, assume_unique = True)

    # Known Forces (Indexes) (per Rank-Nullity theorem (?))
    kfi = udi # fi = force indexes
    ufi = kdi

    # Stiffness Matrix, order:
    # K1 K2  // K1 previously known as Kred (K Reduced)
    # K3 K4
    K1=K[kfi,:] [:,udi]
    K2=K[kfi,:] [:,kdi]
    K3=K[ufi,:] [:,udi]
    K4=K[ufi,:] [:,kdi]

    # Input Known Forces
    Fk = np.zeros(len(kfi)) # Force matrix 
    Fk[kfi == force_index_bc[0]*ndim + force_index_bc[1]]=force_values

    # Solve: F = K@U
    Uk = disp_val_bc # U = displacements 

    # Fk = K1 Uu + K2 Uk 
    Uu = np.linalg.solve(K1, Fk-K2@Uk)

    # Fu = K3 Uu + K4 Uk
    Fu = K3@Uu+K4@Uk

    # Final Nodal Displacement Matrix
    displacement_3d = np.zeros((nodes*ndim))
    displacement_3d[udi]= Uu
    displacement_3d[kdi]= Uk 
    displacement_3d = displacement_3d.reshape((nodes,ndim))
    return B, Fu, Fk, Uu, Uk, ufi, kfi, displacement_3d

def update(displacement_3d, node, step, net_displacement): 
    node[:,1:] += displacement_3d*step
    net_displacement += displacement_3d*step


step = 1/iterations 
net_displacement = np.zeros((nodes*ndim)).reshape(nodes, ndim)
for i in range(iterations):
    B, Fu, Fk, Uu, Uk, ufi, kfi, displacement_3d = calculate(node,strain_constant)
    update(displacement_3d, node, step, net_displacement)

#############################
###### Post Processing ######
#############################

# Final Forces Matrix
F = np.zeros((nodes*ndim))
F[ufi]= Fu
F[kfi]= Fk 

strain = np.einsum("edn, nd->e",B, net_displacement)/L # elements(e), axis/ndim (d), nodes (n)
stress = strain*E # E = young's modulus
bar_force = stress*A
N = bar_force

print("Internal Bar Forces (N):", N, "\n")
print("Final Bar Lengths (L):", L, "\n")
print("Net Nodal Displacements:", net_displacement)