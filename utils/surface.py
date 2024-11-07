import numpy as np
from config import vdw_radii, covalent_radii


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
        self._shift_supercell_positions()
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
                if  i == j:
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

        return surface_markers
