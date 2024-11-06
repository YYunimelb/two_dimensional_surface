# utils/structure_processor.py
from parsers import VaspParser
from utils import SupercellBuilder, LayerAnalyzer



class StructureProcessor:
    def __init__(self, file_path, supercell_boundry=(-2, 2, -2, 2, -2, 2), cutoff_factor=1.0):
        self.file_path = file_path
        self.supercell_boundry = supercell_boundry
        self.cutoff_factor = cutoff_factor

        # 初始化属性
        self.lattice_vectors = None
        self.atomic_types = None
        self.numbers_of_atoms = None
        self.positions = None
        self.supercell_positions = None
        self.supercell_atomic_types = None
        self.supercell_lattice_vectors = None
        self.layers = None

        # 初始化解析器
        self.parser = VaspParser(self.file_path)

    def process_structure(self):
        """执行超胞创建、距离计算和层标记的完整流程，并将结果存储在类属性中"""
        # 解析 POSCAR 文件
        self.lattice_vectors, self.atomic_types, self.numbers_of_atoms, self.positions = self.parser.parse()

        # 建立超胞
        supercell_builder = SupercellBuilder(self.positions, self.lattice_vectors, self.atomic_types, replication=self.supercell_boundry)
        self.supercell_positions, self.supercell_atomic_types, self.supercell_lattice_vectors = supercell_builder.create_supercell()

        # 计算超胞中各原子间的距离矩阵
        distances = supercell_builder.calculate_distances(self.supercell_positions)

        # 使用 LayerAnalyzer 进行超胞分层
        layer_analyzer = LayerAnalyzer(self.supercell_atomic_types, self.positions, self.supercell_positions, distances, self.cutoff_factor)
        self.layers = layer_analyzer.layer_marking()

    def get_results(self):
        """返回计算结果，供外部调用"""
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
    standardization_basis
)

class StructureNormalizer:
    def __init__(self, processor):
        """初始化结构标准化类，直接使用已处理的结构数据"""
        self.lattice_vectors = processor.lattice_vectors
        self.atomic_types = processor.atomic_types
        self.positions = processor.positions
        self.supercell_positions = processor.supercell_positions
        self.supercell_atomic_types = processor.supercell_atomic_types
        self.layers = processor.layers

    def convert_to_normal_structure(self, output_path="POSCAR_bulk"):
        """执行结构标准化转换并保存结果"""

        # 1. 找到结构中心的原子和等价原子
        central_atom_index = find_central_atom(self.positions, self.lattice_vectors)
        layer_of_central_atom = self.layers[central_atom_index]
        equivalent_atoms = find_equivalent_atoms(
            central_atom_index, self.atomic_types, len(self.supercell_positions), self.layers, layer_of_central_atom
        )

        # 2. 计算两个基向量
        basis_vector_1, basis_vector_2 = calculate_basis_vectors(
            self.supercell_positions, central_atom_index, equivalent_atoms
        )

        # 3. 根据前两个基向量计算第三个基向量
        basis_vector_3 = find_third_basis_vector(basis_vector_1, basis_vector_2, self.lattice_vectors)
        new_basis = np.array([basis_vector_1, basis_vector_2, basis_vector_3])

        # 4. 规范化基向量
        final_basis = standardization_basis(new_basis)

        # 5. 转换到规范化的坐标系并去重
        unique_atoms = self._convert_to_relative_normalize(self.supercell_positions, self.supercell_atomic_types,
                                                           new_basis)

        # 6. 写入 POSCAR 文件
        poscar_content = self._write_poscar(unique_atoms, final_basis)
        with open(output_path, "w") as f:
            f.write(poscar_content)

    def _convert_to_relative_normalize(self, supercell_positions, supercell_atomic_types, original_basis):
        """将坐标转换为相对坐标并归一化"""
        inverse_basis = np.linalg.inv(original_basis)
        relative_positions = np.dot(supercell_positions, inverse_basis)
        normalized_positions = np.mod(relative_positions, 1)

        # 将坐标归一化到 0-1 范围内，避免浮点误差
        epsilon = 1e-3
        normalized_positions[normalized_positions > 1 - epsilon] = 0
        rounded_positions = np.round(normalized_positions, decimals=5)

        # 组合坐标和原子类型，去重
        combined = np.core.records.fromarrays(
            [rounded_positions[:, 0], rounded_positions[:, 1], rounded_positions[:, 2], supercell_atomic_types],
            names='x, y, z, type'
        )
        unique_atoms = np.unique(combined)
        return unique_atoms

    def _write_poscar(self, unique_atoms, final_basis):
        # 提取所有唯一的类型
        unique_types = np.unique(unique_atoms['type'])

        # 构建POSCAR内容
        poscar_content = "Generated POSCAR\n1.0\n"
        for vec in final_basis:
            poscar_content += " ".join(f"{v:.10f}" for v in vec) + "\n"
        poscar_content += " ".join(unique_types) + "\n"

        # 计算每种类型的原子数量并排序
        counts = [np.sum(unique_atoms['type'] == typ) for typ in unique_types]
        poscar_content += " ".join(map(str, counts)) + "\n"

        poscar_content += "Direct\n"
        for typ in unique_types:
            for atom in unique_atoms[unique_atoms['type'] == typ]:
                poscar_content += f"{atom['x']:.10f} {atom['y']:.10f} {atom['z']:.10f}\n"

        return poscar_content