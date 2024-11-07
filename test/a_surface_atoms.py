import numpy as np
import copy

vdw_radii = {
    'H': 1.20, 'He': 1.43, 'Li': 2.12, 'Be': 1.98, 'B': 1.91,
    'C': 1.77, 'N': 1.66, 'O': 1.50, 'F': 1.46, 'Ne': 1.58,
    'Na': 2.50, 'Mg': 2.51, 'Al': 2.25, 'Si': 2.19, 'P': 1.90,
    'S': 1.89, 'Cl': 1.82, 'Ar': 1.83, 'K': 2.73, 'Ca': 2.62,
    'Sc': 2.58, 'Ti': 2.46, 'V': 2.42, 'Cr': 2.45, 'Mn': 2.45,
    'Fe': 2.44, 'Co': 2.40, 'Ni': 2.40, 'Cu': 2.38, 'Zn': 2.39,
    'Ga': 2.32, 'Ge': 2.29, 'As': 1.88, 'Se': 1.82, 'Br': 1.86,
    'Kr': 2.25, 'Rb': 3.21, 'Sr': 2.84, 'Y': 2.75, 'Zr': 2.52,
    'Nb': 2.56, 'Mo': 2.45, 'Tc': 2.44, 'Ru': 2.46, 'Rh': 2.44,
    'Pd': 2.15, 'Ag': 2.53, 'Cd': 2.49, 'In': 2.43, 'Sn': 2.42,
    'Sb': 2.47, 'Te': 1.99, 'I': 2.04, 'Xe': 2.06, 'Cs': 3.48,
    'Ba': 3.03, 'La': 2.98, 'Ce': 2.88, 'Pr': 2.92, 'Nd': 2.95,
    'Sm': 2.90, 'Eu': 2.87, 'Gd': 2.83, 'Tb': 2.79, 'Dy': 2.87,
    'Ho': 2.81, 'Er': 2.83, 'Tm': 2.79, 'Yb': 2.80, 'Lu': 2.74,
    'Hf': 2.63, 'Ta': 2.53, 'W': 2.57, 'Re': 2.49, 'Os': 2.48,
    'Ir': 2.41, 'Pt': 2.29, 'Au': 2.32, 'Hg': 2.45, 'Tl': 2.47,
    'Pb': 2.60, 'Bi': 2.54, 'Th': 2.93, 'Pa': 2.88, 'U': 2.71,
    'Np': 2.82, 'Pu': 2.81, 'Am': 2.83, 'Cm': 3.05, 'Bk': 3.40,
    'Cf': 3.05, 'Es': 2.70

}
covalent_radii = {
    'H': 0.53, 'He': 0.31, 'Li': 1.67, 'Be': 1.12, 'B': 0.87, 'C': 0.67, 'N': 0.56, 'O': 0.48, 'F': 0.42, 'Ne': 0.38,
    'Na': 1.9, 'Mg': 1.45, 'Al': 1.18, 'Si': 1.11, 'P': 0.98, 'S': 0.88, 'Cl': 0.79, 'Ar': 0.71,
    'K': 2.43, 'Ca': 1.94, 'Sc': 1.84, 'Ti': 1.76, 'V': 1.71, 'Cr': 1.66, 'Mn': 1.61, 'Fe': 1.56, 'Co': 1.52, 'Ni': 1.49,
    'Cu': 1.45, 'Zn': 1.42, 'Ga': 1.36, 'Ge': 1.25, 'As': 1.14, 'Se': 1.03, 'Br': 0.94, 'Kr': 0.88,
    'Rb': 2.65, 'Sr': 2.19, 'Y': 2.12, 'Zr': 2.06, 'Nb': 1.98, 'Mo': 1.9, 'Tc': 1.83, 'Ru': 1.78, 'Rh': 1.73, 'Pd': 1.69,
    'Ag': 1.65, 'Cd': 1.61, 'In': 1.56, 'Sn': 1.45, 'Sb': 1.33, 'Te': 1.23, 'I': 1.15, 'Xe': 1.08,
    'Cs': 2.98, 'Ba': 2.53, 'Hf': 2.08, 'Ta': 2.0, 'W': 1.93, 'Re': 1.88, 'Os': 1.85, 'Ir': 1.8, 'Pt': 1.77, 'Au': 1.74,
    'Hg': 1.71, 'Tl': 1.56, 'Pb': 1.54, 'Bi': 1.43, 'Po': 1.35, 'At': 1.27, 'Rn': 1.2,
    'Fr': None, 'Ra': None, 'Rf': None, 'Db': None, 'Sg': None, 'Bh': None, 'Hs': None, 'Mt': None, 'Ds': None,
    'Rg': None, 'Cn': None, 'Nh': None, 'Fl': None, 'Mc': None, 'Lv': None, 'Ts': None, 'Og': None,
    'La': 2.26, 'Ce': 2.1, 'Pr': 2.47, 'Nd': 2.06, 'Pm': 2.05, 'Sm': 2.38, 'Eu': 2.31, 'Gd': 2.33, 'Tb': 2.25, 'Dy': 2.28,
    'Ho': 2.26, 'Er': 2.26, 'Tm': 2.22, 'Yb': 2.22, 'Lu': 2.17,
    'Ac': None, 'Th': None, 'Pa': None, 'U': None, 'Np': None, 'Pu': None, 'Am': None, 'Cm': None, 'Bk': None,
    'Cf': None, 'Es': None, 'Fm': None, 'Md': None, 'No': None, 'Lr': None
}


def parse_vasp_contcar(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    scale_factor = float(lines[1].strip())
    lattice_vectors = np.array([list(map(float, lines[i].split())) for i in range(2, 5)]) * scale_factor
    atomic_types_line = lines[5].split()
    numbers_of_atoms = list(map(int, lines[6].split()))

    # 生成包含每个原子类型正确数量的列表
    atomic_types = []
    for atom_type, count in zip(atomic_types_line, numbers_of_atoms):
        atomic_types.extend([atom_type] * count)

    total_atoms = sum(numbers_of_atoms)
    coordinate_type = lines[7].strip().lower()

    base_index = 8
    atomic_positions = np.array([list(map(float, lines[base_index + i].split()[:3])) for i in range(total_atoms)])
    if coordinate_type == 'direct':
        cartesian_positions = np.dot(atomic_positions, lattice_vectors)
    else:
        cartesian_positions = atomic_positions

    return lattice_vectors, atomic_types, numbers_of_atoms, cartesian_positions



def parse_vasp_contcar(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    scale_factor = float(lines[1].strip())
    lattice_vectors = np.array([list(map(float, lines[i].split())) for i in range(2, 5)]) * scale_factor
    atomic_types_line = lines[5].split()
    numbers_of_atoms = list(map(int, lines[6].split()))

    # 生成包含每个原子类型正确数量的列表
    atomic_types = []
    for atom_type, count in zip(atomic_types_line, numbers_of_atoms):
        atomic_types.extend([atom_type] * count)

    total_atoms = sum(numbers_of_atoms)
    coordinate_type = lines[7].strip().lower()

    base_index = 8
    atomic_positions = np.array([list(map(float, lines[base_index + i].split()[:3])) for i in range(total_atoms)])
    if coordinate_type == 'direct':
        cartesian_positions = np.dot(atomic_positions, lattice_vectors)
    else:
        cartesian_positions = atomic_positions

    return lattice_vectors, atomic_types, numbers_of_atoms, cartesian_positions


def create_supercell(positions, lattice_vectors,atomic_types, replication=(-1, 1, -1, 1, -1, 1)):

    supercell_positions = []
    supercell_atomic_types = copy.deepcopy(atomic_types)

    ix_start, ix_end, iy_start, iy_end, iz_start, iz_end = replication
    shift = (0 - ix_start) * lattice_vectors[0] + (0 - iy_start) * lattice_vectors[1] + (0 - iz_start) * \
            lattice_vectors[2]

    for pos in positions:
        supercell_positions.append(pos + shift)

    # 计算超胞的晶格向量
    supercell_lattice_vectors = np.zeros_like(lattice_vectors)
    supercell_lattice_vectors[0] = lattice_vectors[0] * (ix_end - ix_start + 1)
    supercell_lattice_vectors[1] = lattice_vectors[1] * (iy_end - iy_start + 1)
    supercell_lattice_vectors[2] = lattice_vectors[2] * (iz_end - iz_start + 1)

    for i in range(ix_start, ix_end + 1):
        for j in range(iy_start, iy_end + 1):
            for k in range(iz_start, iz_end + 1):
                shift = (i-ix_start) * lattice_vectors[0] +  (j-iy_start)  * lattice_vectors[1] +  (k-iz_start)  * lattice_vectors[2]
                if i == 0 and j ==0 and k ==0:
                    continue
                else:
                    for pos in positions:
                        supercell_positions.append(pos + shift)
                    supercell_atomic_types+=atomic_types
    return np.array(supercell_positions),supercell_atomic_types,supercell_lattice_vectors


def calculate_distances(positions):
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    return distances

def calculate_dynamic_cutoff(index1, index2, atomic_types,cutoff_factor = 1.3):
    # 找出两个索引对应的原子类型

    # 计算阈值
    radius_sum = covalent_radii[atomic_types[index1]] + covalent_radii[atomic_types[index2]]
    vdw_sum = vdw_radii[atomic_types[index1]] + vdw_radii[atomic_types[index2]]
    return (radius_sum+vdw_sum)/2*cutoff_factor

def layer_marking(atomic_types, numbers_of_atoms, positions, all_positions, distances, cutoff_factor=1.2):
    num_original = len(positions)
    num_total = len(all_positions)
    layers = np.zeros(num_total, dtype=int)

    layer_number = 1

    for start_index in range(num_original):
        if layers[start_index] != 0:
            continue

        current_indices = [start_index]
        layers[start_index] = layer_number

        while current_indices:
            next_indices = []
            for index in current_indices:
                for neighbor_index in range(num_total):
                    if layers[neighbor_index] == 0:
                        dynamic_cutoff = calculate_dynamic_cutoff(index, neighbor_index, atomic_types,cutoff_factor = cutoff_factor)
                        if distances[index, neighbor_index] < dynamic_cutoff:
                            layers[neighbor_index] = layer_number
                            next_indices.append(neighbor_index)

            current_indices = list(set(next_indices))

        layer_number += 1

    return layers



name = f"POSCAR_2D"
lattice_vectors, atomic_types, numbers_of_atoms, positions = parse_vasp_contcar(name)
# 建立超胞
supercell_boundry = (-2, 2, -2, 2, -2, 2)


supercell_positions,supercell_atomic_types,supercell_lattice_vectors = create_supercell(positions, lattice_vectors,atomic_types,replication=supercell_boundry)

supercell_positions_ini = []
ix_start, ix_end, iy_start, iy_end, iz_start, iz_end = supercell_boundry
shift = (0 - ix_start) * lattice_vectors[0] + (0 - iy_start) * lattice_vectors[1] + (0 - iz_start) * \
        lattice_vectors[2]
for pos in supercell_positions:
    supercell_positions_ini.append(pos - shift)


distances = calculate_distances(supercell_positions)
# 超胞分层
layers = layer_marking(supercell_atomic_types, numbers_of_atoms, positions, supercell_positions, distances, cutoff_factor=1)
print(layers)



import numpy as np


def find_surface_atoms(positions, layers, atomic_types, supercell_positions, supercell_atomic_types, supercell_layers,
                       covalent_radii):
    # Initialize an empty list to store surface atom markers
    surface_markers = np.zeros(len(positions), dtype=int)  # 1 for top surface, -1 for bottom surface, 0 for neither


    for i, pos in enumerate(positions):
        # Consider only atoms in the primary cell layer 1
        if layers[i] != 1:
            continue

        # Get the atomic radius for the element type
        element_type1 = supercell_atomic_types[i]
        radius_element1 = covalent_radii[element_type1]


        # Define a flag to mark if atoms are found above or below
        found_above = False
        found_below = False

        for j, supercell_pos in enumerate(supercell_positions):
            # Consider only atoms with layers value 1 in the supercell
            if supercell_layers[j] != 1:
                continue
            if i ==j:
                continue

            # Check if the atom is within the cylindrical region above or below the current atom
            distance_xy = np.linalg.norm(supercell_pos[:2] - pos[:2])  # distance in the x-y plane
            element_type2 = supercell_atomic_types[i]
            radius_element2 = covalent_radii[element_type2]
            radius = radius_element1 + radius_element2
            if distance_xy <= radius:
                if supercell_pos[2] > pos[2]:  # atom above
                    found_above = True
                elif supercell_pos[2] < pos[2]:  # atom below
                    found_below = True

        # Mark surface atoms based on above/below conditions
        if not found_above and not found_below:
            surface_markers[i] = 3  # top surface
        elif not found_below:
            surface_markers[i] = -1  # bottom surface
        elif not found_above:
            surface_markers[i] = 1  # atom surrounded above and below
        else:
            surface_markers[i] = 0  # not on the surface

    return surface_markers



surface_atoms = find_surface_atoms(positions, layers, atomic_types, supercell_positions_ini, supercell_atomic_types, layers,
                                   covalent_radii)
print(surface_atoms)

# # Define points A, B, C, and P
# A = np.array([0.72510, 7.14735, 8.18064])
# B = np.array([-0.72510, 4.20822, 8.18064])
# C = np.array([3.70861, 5.64899, 9.2571])
# P = np.array([1.40340, 4.98177, 8.60293])
#
# # Calculate vectors AB and AC
# AB = B - A
# AC = C - A
#
# # Calculate the normal vector by taking the cross product of AB and AC
# normal_vector = np.cross(AB, AC)
#
# # Calculate the constant term D in the plane equation
# D = -np.dot(normal_vector, A)
#
# # Calculate the distance from point P to the plane
# distance = abs(np.dot(normal_vector, P) + D) / np.linalg.norm(normal_vector)
# print(distance)
