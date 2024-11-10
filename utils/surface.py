import numpy as np
from config import vdw_radii, covalent_radii
import itertools
from utils.geometry import calculate_distances,calculate_dynamic_cutoff

class SurfaceAtomIdentifier:
    def __init__(self, processor):
        """
        Initialize the SurfaceAtomIdentifier class using processed structure data from a StructureProcessor instance.

        Parameters:
        - processor (StructureProcessor): An instance with processed structure data.
        """
        self.lattice_vectors = processor.lattice_vectors
        self.atomic_types = processor.atomic_types
        self.positions = processor.positions
        self.supercell_positions = processor.supercell_positions
        self.supercell_atomic_types = processor.supercell_atomic_types
        self.layers = processor.layers  # Layers for both primary and supercell
        self.supercell_boundry = processor.supercell_boundry
        self.cutoff_factor = processor.cutoff_factor
        self._shift_supercell_positions()


    def _shift_supercell_positions(self):
        """
        Shift supercell positions to initial reference position based on supercell boundaries.
        """
        ix_start, ix_end, iy_start, iy_end, iz_start, iz_end = self.supercell_boundry
        shift = (
                (0 - ix_start) * self.lattice_vectors[0]
                + (0 - iy_start) * self.lattice_vectors[1]
                + (0 - iz_start) * self.lattice_vectors[2]
        )
        # Apply the shift to all supercell positions
        self.supercell_positions = [pos - shift for pos in self.supercell_positions]

    def find_surface_atoms(self):
        """
        Identify surface atoms by checking if atoms in the primary cell are surrounded above or below by atoms
        in the supercell. Uses atomic covalent radii for defining proximity in the x-y plane.

        Returns:
        - np.array: An array with surface markers where 1 = top surface, -1 = bottom surface, 0 = neither.
        """

        surface_markers = np.zeros(len(self.positions), dtype=int)  # 1 for top surface, -1 for bottom, 0 for neither

        for i, pos in enumerate(self.positions):
            # Only consider atoms in the primary cell with layer marked as 1

            # Get covalent radius of the current atom type
            element_type1 = self.supercell_atomic_types[i]
            radius_element1 = covalent_radii[element_type1]

            # Flags to mark if atoms are found above or below the current atom
            found_above = False
            found_below = False

            for j, supercell_pos in enumerate(self.supercell_positions):
                # Skip the same atom
                if i == j:
                    continue

                # Calculate distance in x-y plane and get the combined radius for both atoms
                distance_xy = np.linalg.norm(supercell_pos[:2] - pos[:2])
                element_type2 = self.supercell_atomic_types[j]
                radius_element2 = covalent_radii[element_type2]
                radius = radius_element1 + radius_element2

                # Check if atom is in the defined region above or below the current atom
                if distance_xy <= radius:
                    if supercell_pos[2] > pos[2]:  # atom above
                        found_above = True
                    elif supercell_pos[2] < pos[2]:  # atom below
                        found_below = True

            # Mark surface atoms based on above/below conditions
            if not found_above and not found_below:
                surface_markers[i] = 3  # isolated atom (top surface)
            elif not found_below:
                surface_markers[i] = -1  # bottom surface
            elif not found_above:
                surface_markers[i] = 1  # top surface
            else:
                surface_markers[i] = 0  # not on the surface

        self.surface_markers = surface_markers
        return surface_markers

    def analyze_bonded_surface_atoms(self):
        """
        Confirm true surface atoms among candidates marked by surface_markers.

        Returns:
        - dict: Dictionary with keys 'top_surface' and 'bottom_surface' listing confirmed atom details.
        """
        top_surface = []
        bottom_surface = []
        self.distances = calculate_distances(positions=self.supercell_positions)

        for i, marker in enumerate(self.surface_markers):
            if marker == 1 or marker == -1:
                bonding_atoms = []
                bonding_info = []  # List to store bonding details for each atom

                # Find bonding atoms within the cutoff
                for j in range(len(self.supercell_positions)):
                    if i == j:
                        continue
                    distance = self.distances[i, j]
                    cutoff = calculate_dynamic_cutoff(
                        self.supercell_atomic_types[i],
                        self.supercell_atomic_types[j],
                        self.cutoff_factor
                    )
                    if distance < cutoff:
                        bonding_atoms.append(j)
                        bonding_info.append({
                            "bonded_atom_index": j,
                            "bonded_atom_type": self.supercell_atomic_types[j],
                            "bond_distance": distance
                        })

                # Build atom info dictionary
                atom_info = {
                    "index": i,
                    "element": self.supercell_atomic_types[i],
                    "bond_count": len(bonding_atoms),
                    "bonded_atoms": bonding_info
                }

                # Check bonding configuration and update top_surface or bottom_surface
                if len(bonding_atoms) == 1:
                    if marker == 1:
                        top_surface.append(atom_info)
                    elif marker == -1:
                        bottom_surface.append(atom_info)

                elif len(bonding_atoms) == 2:
                    avg_z = np.mean([self.supercell_positions[j][2] for j in bonding_atoms])
                    if marker == 1 and avg_z > self.positions[i][2]:
                        continue  # Not top surface
                    elif marker == -1 and avg_z < self.positions[i][2]:
                        continue  # Not bottom surface

                    if marker == 1:
                        top_surface.append(atom_info)
                    elif marker == -1:
                        bottom_surface.append(atom_info)

                elif len(bonding_atoms) >= 3:
                    for combination in itertools.combinations(bonding_atoms, 3):
                        points = [self.supercell_positions[idx] for idx in combination]
                        normal_vector = np.cross(points[1] - points[0], points[2] - points[0])
                        if normal_vector[2] > 0 and marker == 1 and all(p[2] > self.positions[i][2] for p in points):
                            break
                        elif normal_vector[2] < 0 and marker == -1 and all(p[2] < self.positions[i][2] for p in points):
                            break

                    else:
                        if marker == 1:
                            top_surface.append(atom_info)
                        elif marker == -1:
                            bottom_surface.append(atom_info)

        return {"top_surface": top_surface, "bottom_surface": bottom_surface}


