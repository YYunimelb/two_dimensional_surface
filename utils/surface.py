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

    @staticmethod
    def point_to_plane_z(x, y, plane_params):
        """
        Calculate the z-coordinate of a point (x, y) on a plane.

        Parameters:
        - x, y: Coordinates of the point.
        - plane_params: Tuple (a, b, c, d) defining the plane equation ax + by + cz + d = 0.

        Returns:
        - float: Calculated z-coordinate on the plane.
        """
        a, b, c, d = plane_params
        return -(a * x + b * y + d) / c

    @staticmethod
    def calculate_plane(points):
        """
        Calculate the plane equation parameters from three points.

        Parameters:
        - points: List of three points, each a numpy array [x, y, z].

        Returns:
        - tuple: (a, b, c, d) coefficients of the plane equation ax + by + cz + d = 0.
        """
        p1, p2, p3 = points
        # Compute two vectors on the plane
        v1 = p2 - p1
        v2 = p3 - p1
        # Compute the normal vector
        normal = np.cross(v1, v2)
        a, b, c = normal
        # Compute d using one of the points
        d = -np.dot(normal, p1)
        return a, b, c, d

    def check_layer_connectivity(self):
        """
        Determine if the layers in the structure are connected.

        For each atom, identify bonding atoms within the same layer.
        Then check if atoms in other layers are bonded using an adjusted cutoff
        based on the most tightly bonded element type.

        Returns:
        - bool: True if no inter-layer connections are found, otherwise False.
        """
        self.distances = calculate_distances(positions=self.supercell_positions)

        for i in range(len(self.positions)):
            bonding_atoms = []
            bonding_ratios = {}  # Store bonding tightness ratios grouped by element type

            # Identify bonding atoms in the same layer
            for j in range(len(self.supercell_positions)):
                if i == j or self.layers[i] != self.layers[j]:  # Skip different layers
                    continue

                distance = self.distances[i, j]
                cutoff = calculate_dynamic_cutoff(
                    self.supercell_atomic_types[i],
                    self.supercell_atomic_types[j],
                    self.cutoff_factor
                )

                if distance < cutoff:
                    bonding_atoms.append(j)

                    element_type = self.supercell_atomic_types[j]
                    radius_sum = covalent_radii[self.supercell_atomic_types[i]] + covalent_radii[element_type]
                    ratio = distance  # Bond tightness ratio

                    if element_type not in bonding_ratios or ratio < bonding_ratios[element_type]:
                        bonding_ratios[element_type] = ratio

            # Find the most tightly bonded element type
            if not bonding_ratios:
                continue  # Skip atoms with no bonds in the same layer

            most_tightly_bonded_element = min(bonding_ratios, key=bonding_ratios.get)

            # Filter bonding atoms to only include the most tightly bonded element type
            tightly_bonded_distances = [
                self.distances[i, j]
                for j in bonding_atoms
                if self.supercell_atomic_types[j] == most_tightly_bonded_element
            ]

            if not tightly_bonded_distances:
                continue  # No bonds with the tightly bonded element type

            # Calculate adjusted cutoff
            adjusted_cutoff = 1.15 * max(tightly_bonded_distances)

            # Check for bonding to atoms in other layers
            for j in range(len(self.supercell_positions)):
                if i == j or self.layers[i] == self.layers[j]:  # Skip atoms in the same layer
                    continue

                element_type = self.supercell_atomic_types[j]
                if element_type == most_tightly_bonded_element:
                    distance = self.distances[i, j]
                    if distance < adjusted_cutoff:
                        # Found an inter-layer bond
                        return False
        # No inter-layer connections found

        return True

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
                        # Get the three points defining the plane
                        points = [self.supercell_positions[idx] for idx in combination]
                        # Calculate the plane equation parameters
                        plane_params = self.calculate_plane(points)
                        if abs(plane_params[2]) < 0.1:
                            continue
                        current_x,current_y,current_z = self.supercell_positions[i]

                        # Compute the z value on the plane for the current atom's x, y
                        plane_z = self.point_to_plane_z(current_x, current_y, plane_params)

                        # Compare the plane z with the current atom's z
                        if marker == 1 and plane_z > current_z+ 0.01 :
                            break  # Plane is above the current atom
                        elif marker == -1 and plane_z < current_z-0.01 :
                            break  # Plane is below the current atom

                    else:
                        if marker == 1:
                            top_surface.append(atom_info)
                        elif marker == -1:
                            bottom_surface.append(atom_info)

        return {"top_surface": top_surface, "bottom_surface": bottom_surface}


