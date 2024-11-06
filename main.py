# main.py
from parsers import VaspParser
from utils import SupercellBuilder, LayerAnalyzer

import numpy as np
from parsers import VaspParser
from utils import SupercellBuilder, LayerAnalyzer,LayerChecker,StructureProcessor



def convert_to_normal_structure(initial_structure_path="POSCAR", final_structure_path="POSCAR_bulk"):
    # 解析 POSCAR 文件
    parser = VaspParser(initial_structure_path)
    lattice_vectors, atomic_types, numbers_of_atoms, positions = parser.parse()

    # 建立超胞
    supercell_builder = SupercellBuilder(positions, lattice_vectors, atomic_types)
    supercell_positions, supercell_atomic_types, supercell_lattice_vectors = supercell_builder.create_supercell()
    distances = supercell_builder.calculate_distances(supercell_positions)

    # 超胞分层
    layer_analyzer = LayerAnalyzer(supercell_atomic_types, numbers_of_atoms, positions, supercell_positions, distances)
    layers = layer_analyzer.layer_marking()
    print(f"Layer info: {layers}")

    # 判断材料类型
    type_of_materials = check_layers(layers)
    if type_of_materials == "bulk":
        print("This material is not layered")
        return

    # 找到中心原子和等价原子
    central_atom_index = find_central_atom(positions, lattice_vectors)
    layer_of_central_atom = layers[central_atom_index]
    equivalent_atoms = find_equivalent_atoms(central_atom_index, atomic_types, len(supercell_positions), layers, layer_of_central_atom)

    # 计算基矢量
    basis_calculator = BasisCalculator(lattice_vectors)
    basis_vector_1, basis_vector_2 = basis_calculator.calculate_basis_vectors(supercell_positions, central_atom_index, equivalent_atoms)
    basis_vector_3 = basis_calculator.find_third_basis_vector(basis_vector_1, basis_vector_2)
    new_basis = np.array([basis_vector_1, basis_vector_2, basis_vector_3])

    # 规范化基矢量
    final_basis = basis_calculator.compute_new_basis(new_basis)

    # 转换为相对规范化坐标，并保存为 POSCAR
    converter = CoordinateConverter(final_basis)
    unique_atoms = converter.convert_to_relative_normalize(supercell_positions, supercell_atomic_types)
    poscar_content = write_poscar(unique_atoms, final_basis)
    with open(final_structure_path, "w") as f:
        f.write(poscar_content)



def main():
    # 定义输入和输出路径
    file_path = "data/POSCAR_mp-224"  # 替换为实际POSCAR文件路径

    # 解析 POSCAR 文件
    processor = StructureProcessor(file_path, supercell_boundry=(-2, 2, -2, 2, -2, 2), cutoff_factor=1.0)
    processor.process_structure()
    layer_checker = LayerChecker(processor)
    result = layer_checker.analyze_structure()
    print("The structure is:", result)

    # 建立超胞

if __name__ == "__main__":
    main()
