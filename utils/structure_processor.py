# utils/structure_processor.py
from parsers import VaspParser
from utils import SupercellBuilder, LayerAnalyzer



class StructureProcessor:
    def __init__(self, file_path, supercell_boundry=(-2, 2, -2, 2, -2, 2), cutoff_factor=1.0):
        self.file_path = file_path
        self.supercell_boundry = supercell_boundry
        self.cutoff_factor = cutoff_factor
        self.lattice_vectors = None
        self.atomic_types = None
        self.numbers_of_atoms = None
        self.positions = None
        self.supercell_positions = None
        self.supercell_atomic_types = None
        self.supercell_lattice_vectors = None
        self.layers = None

        # parser structure
        self.parser = VaspParser(self.file_path)

    def process_structure(self):
        # parse POSCAR
        self.lattice_vectors, self.atomic_types, self.numbers_of_atoms, self.positions = self.parser.parse()

        # build supercell
        supercell_builder = SupercellBuilder(self.positions, self.lattice_vectors, self.atomic_types, replication=self.supercell_boundry)
        self.supercell_positions, self.supercell_atomic_types, self.supercell_lattice_vectors = supercell_builder.create_supercell()

        # calculate the distance for all atoms
        distances = supercell_builder.calculate_distances(self.supercell_positions)

        # use LayerAnalyzer for layering
        layer_analyzer = LayerAnalyzer(self.supercell_atomic_types, self.positions, self.supercell_positions, distances, self.cutoff_factor)
        self.layers = layer_analyzer.layer_marking()

    def get_results(self):
        """Return the calculation result for external calls."""
        return {
            "lattice_vectors": self.lattice_vectors,
            "atomic_types": self.atomic_types,
            "numbers_of_atoms": self.numbers_of_atoms,
            "positions": self.positions,
            "supercell_positions": self.supercell_positions,
            "supercell_atomic_types": self.supercell_atomic_types,
            "supercell_lattice_vectors": self.supercell_lattice_vectors,
            "layers": self.layers
        }


import numpy as np
from utils.geometry import (
    find_central_atom,
    find_equivalent_atoms,
    calculate_basis_vectors,
    find_third_basis_vector,
    standardization_basis,
    fill_to_new_basis
)
from parsers.write_poscar import create_poscar


class StructureNormalizer:
    def __init__(self, processor):
        """Initialize the structure normalization class and directly use the processed structure data."""
        self.lattice_vectors = processor.lattice_vectors
        self.atomic_types = processor.atomic_types
        self.positions = processor.positions
        self.supercell_positions = processor.supercell_positions
        self.supercell_atomic_types = processor.supercell_atomic_types
        self.layers = processor.layers

    def convert_to_normal_structure(self, output_path="POSCAR_bulk"):
        """Perform structure normalization transformation and save the result."""

        # 1. Find the central atom of the structure and its equivalent atoms
        central_atom_index = find_central_atom(self.positions, self.lattice_vectors)
        layer_of_central_atom = self.layers[central_atom_index]
        equivalent_atoms = find_equivalent_atoms(
            central_atom_index, self.atomic_types, len(self.supercell_positions), self.layers, layer_of_central_atom
        )

        # 2. Calculate two basis vectors
        basis_vector_1, basis_vector_2 = calculate_basis_vectors(
            self.supercell_positions, central_atom_index, equivalent_atoms
        )

        # 3. Calculate the third basis vector based on the first two
        basis_vector_3 = find_third_basis_vector(basis_vector_1, basis_vector_2, self.lattice_vectors)
        new_basis = np.array([basis_vector_1, basis_vector_2, basis_vector_3])

        # 4. Normalize the basis vectors
        final_basis = standardization_basis(new_basis)

        # 5. Transform to the normalized coordinate system and remove duplicates
        positions, atomic_types = fill_to_new_basis(self.supercell_positions, self.supercell_atomic_types,
                                                           new_basis)

        # 6. Write to POSCAR file
        create_poscar(final_basis, atomic_types, positions,output_path)



    def _write_poscar(self, unique_atoms, final_basis):
        # Extract all unique types
        unique_types = np.unique(unique_atoms['type'])

        # Construct POSCAR content
        poscar_content = "Generated POSCAR\n1.0\n"
        for vec in final_basis:
            poscar_content += " ".join(f"{v:.10f}" for v in vec) + "\n"
        poscar_content += " ".join(unique_types) + "\n"

        # Calculate and sort the number of atoms of each type
        counts = [np.sum(unique_atoms['type'] == typ) for typ in unique_types]
        poscar_content += " ".join(map(str, counts)) + "\n"

        poscar_content += "Direct\n"
        for typ in unique_types:
            for atom in unique_atoms[unique_atoms['type'] == typ]:
                poscar_content += f"{atom['x']:.10f} {atom['y']:.10f} {atom['z']:.10f}\n"

        return poscar_content
