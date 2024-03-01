import numpy as np
from math import sqrt
from numpy.linalg import inv
import itertools
import matplotlib.pyplot as plt
import operator
from functools import reduce

# Change dataname to any linear or nonlinear input file!
dataname = "nonLinear3DCube.txt"


### DATA PROCESSING ###
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
        node_pos = np.copy(node[:,1:])
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
            local_node1 = int(columns[1])
            local_node2 = int(columns[2])
            youngs_modulus = float(columns[3])
            area = float(columns[4])
            ele_data.append([element_number, local_node1, local_node2, youngs_modulus, area])
        ele = np.array(ele_data)
        local_nodes_arr = ele[:,1:3].astype(int)-1
        E = ele[:,3]
        a = ele[:,4]
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
            force.append([int(columns[0]), int(columns[1]), float(columns[2])]) # array of forces info0
        force_index_bc = (np.array((fnode, fdof))-1).T
        fnode = np.array(fnode)
        fdof = np.array(fdof)
        force_values = np.array(fval)
    elif "Number of known/applied displacements" in line:
        ndbcs = int(lines[lineno + 1])
    elif "Node#, Displacement direction, Displacement value" in line:
        index = lineno + 1
        data = split_cols(lines, index)
        disp_bc=[]
        dbcdof=[]
        disp_val_bc=[]
        dbc=[] 
        for columns in data:
            disp_bc.append(int(columns[0]))
            dbcdof.append(int(columns[1]))
            disp_val_bc.append(float(columns[2]))
            dbc.append([int(columns[0]), int(columns[1]), float(columns[2])])
        dnode = np.array(disp_bc)
        ddof = np.array(dbcdof)
        dval = np.array(disp_val_bc)
    lineno += 1
currentNode = np.copy(node[:,1:]) # nodal positions after each iteration

### FUNCTIONS ###

def count2List(lst1, lst2):
    return np.array([[i, j] for i, j in zip(lst1, lst2)]).ravel()
def count3List(lst1, lst2, lst3):
    return np.array([[i, j, k] for i, j, k in zip(lst1, lst2, lst3)]).ravel()
def convert_to_2d_array(lst):
    return np.array(lst).reshape(-1, ndim)
def calcC():
    # connectivity: global nodes to local nodes
    C = np.zeros((neles, nodelm, nodes)) # nodes per element, total nodes, total elements
    ele_index = np.arange(len(local_nodes_arr))[:, np.newaxis].repeat(nodelm, axis = 1)
    node_index = np.arange(nodelm)[np.newaxis, :]
    C[ele_index, node_index, local_nodes_arr] = 1
    return C

def calcB(currentNode):
    # B matrix takes global nodes to elements = Connectivity (C) then Angles (A) matrix
    coord_1 = currentNode[local_nodes_arr[:,0], :]
    coord_2 = currentNode[local_nodes_arr[:,1], :]
    element_dir = coord_2 - coord_1 # n1 to n2 vector
    L = np.linalg.norm(element_dir, axis =1) # bar length
  
    # Angles (A) matrix
    normalized_dir = element_dir/L[:,np.newaxis] # n1 basis, element component to nodes
    A = np.stack([-normalized_dir, normalized_dir]) # local nodes to element

    # Basis (B) matrix
    B = np.einsum("med, emn -> edn", A, C) # nodelm(m), elements(e), ndims(d) x elements(e), nodelm (m), nodes (n)
    return A, B, L

def calcS():
    if ndim == 2:
        arr = np.array([[[0,1],[0,-1]], [[1,0],[-1,0]], [[0,-1],[0,1]], [[-1,0],[1,0]]], dtype = float)#: for 2d
    elif ndim == 3:
        arr = np.array([[[0,0,1],[0,0,-1]], [[0,1,0],[0,-1,0]], [[1,0,0],[-1, 0,0]],[[0,0,-1],[0,0,1]], [[0,-1,0],[0,1,0]], [[-1,0,0],[1, 0,0]]], dtype = float)
    S = np.tile(arr, (neles, 1, 1))
    S = S.reshape(neles, nodelm, ndim, nodelm, ndim)
    return S

nsteps = 300 # load steps
maxnewt = 5 # max newton raphson iterations

# define unit steps (based on loading)
unitdval = dval/nsteps
unitfval = force_values/nsteps

# force tolerance:
ftol = 1e-9

# initialize forces, displacements
F = np.zeros((ndim, nodes))
bar_forces = np.zeros((nsteps, neles))
N = np.zeros((neles))
Ru = np.ones(1)
node_displacements = np.zeros((nsteps,nodes, ndim))

# fist set of values
C = calcC()
L0 = calcB(currentNode)[2]
S = calcS()
Ea = E*a
dNdd = Ea/L0

newtno=0
for load in range(nsteps): # 300, nsteps
    newtno = 0

    # input known forces
    if nfbcs!=0: 
        if newtno == 0:
            F[fdof[:nfbcs]-1, fnode-1] +=unitfval
        
    while (((max(Ru) >=ftol or abs(min(Ru))>=ftol) or newtno == 0) and not(newtno>=maxnewt)): 
        A, B, L = calcB(currentNode)

        # -e = local, nodal (Fe is forces at local nodes)
        # global nodal, local nodal, elements:
        # u, ue, d
        # F, Fe, N
        # R, Re, Rb
        # Ktangent, Ktangente

        # Connectivity (C) matrix takes from global nodal to local nodal
        # Angles (A) matrix takes from local nodal to elements
        scale = dNdd-N/L # neles
        # Ktangente = A.T(dNdd - N/L)A + N*S /L
        # # Ktangent = C.T(Ktangente)C

        start = np.einsum("mnd,n, MnD -> nmdMD ", A,scale, A)
        subtraction = np.einsum("nmdMD, n -> nmdMD", S, N/L) # elements, nodelm, ndim, nodelm, ndim
        Ktangente = start + subtraction # e for local nodal; elements, nodelm, ndim, nodelm, ndim
        Ktangent = np.einsum("nmo, nmdMD, nMO -> odOD ", C, Ktangente, C) # global nodal; nodes, ndim, nodes, ndim
        Ktangent = np.reshape(Ktangent, (ndim*nodes, ndim*nodes)) # flatten, 2d 

        # Input Known Forces
        # get expected/calculated bar forces from elemental to global nodal form: CtAtN
        expectedForcese = np.einsum("mnd, n -> mnd", A, N)
        expectedForces = np.einsum("nmo, mnd -> do", C, expectedForcese)
   
        # flatten forces and expected forces
        if ndim == 2:
            F_flat = count2List(F[0], F[1]) # creates xy xy patern instead of x then y values
            expectedForces_flat = count2List(expectedForces[0], expectedForces[1])
        elif ndim == 3:
            F_flat = count3List(F[0], F[1],F[2])
            expectedForces_flat = count3List(expectedForces[0], expectedForces[1],expectedForces[2])

        R = F_flat - expectedForces_flat
        # print("r", R)
        kdi = (dnode-1,ddof[:ndbcs]-1) # at which nodes displacements are known and at what direction
        kdi_list = np.vstack(kdi).T.tolist() 

        # get "unraveled" indexes; 
        kdi_flat = np.ravel_multi_index(np.array(kdi_list).T, ( nodes, ndim))
        udi_flat = np.setdiff1d(np.arange(ndim*nodes), kdi_flat, assume_unique = True) # every index not found in kdi_flat

        # Known Forces (Indexes) (per Rank-Nullity theorem (?)) 
        # To solve for each element: at known displacements, there is unknown supportive forces; at unknown displacements, there must be known forces.
        kfi_flat = udi_flat
        ufi_flat = kdi_flat
        
        # R = Ktangent * du
        # Stiffness Tangent Matrix, order:
        # K1 K2  
        # K3 K4

        K1=Ktangent[kfi_flat,:] [:,udi_flat]
        K2=Ktangent[kfi_flat,:] [:,kdi_flat]
        K3=Ktangent[ufi_flat,:] [:,udi_flat]
        K4=Ktangent[ufi_flat,:] [:,kdi_flat]

        Rk = R[kfi_flat] # same shape as known forces

        # Solve: R = KTangent@dU
        dUk = np.zeros(ndbcs)
        if newtno == 0:
            dUk = unitdval 

        dUu = np.linalg.solve(K1, Rk-K2@dUk)
        Ru = K3@dUu+K4@dUk

        # Final Nodal Displacement Matrix
        displacement_3d = np.zeros((nodes*ndim))
        displacement_3d[udi_flat]= dUu # flattened, 2D
        displacement_3d[kdi_flat]= dUk
        
        displacement_3d = convert_to_2d_array(displacement_3d)
        currentNode+=displacement_3d
        net_displacement = currentNode-node_pos # net length change in each bar
        
        A, B, L = calcB(currentNode)

        strain = (L - L0)/L0
        stress = strain*E # E = young's modulus
        bar_force = stress*a
        N = bar_force

        ### For Visualization ###
        node_displacements[load, :, :] = net_displacement
        bar_forces[load, :] = N 
        newtno+=1

print("Internal Bar Forces (N):", N, "\n")
print("Final Bar Lengths (L):", L, "\n")
print("Net Nodal Displacements:", net_displacement)