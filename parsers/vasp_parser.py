# parsers/vasp_parser.py
import numpy as np

class VaspParser:
    def __init__(self, file_path):
        self.file_path = file_path

    def parse(self):
        """
        解析 VASP POSCAR 文件，返回晶格向量、原子类型、原子数量、原子位置（以笛卡尔坐标表示）。
        """
        with open(self.file_path, 'r') as file:
            lines = file.readlines()

        # 解析比例因子和晶格向量
        scale_factor = float(lines[1].strip())
        lattice_vectors = np.array([list(map(float, lines[i].split())) for i in range(2, 5)]) * scale_factor

        # 解析原子类型和数量
        atomic_types_line = lines[5].split()
        numbers_of_atoms = list(map(int, lines[6].split()))

        # 生成包含每个原子类型的完整列表
        atomic_types = []
        for atom_type, count in zip(atomic_types_line, numbers_of_atoms):
            atomic_types.extend([atom_type] * count)

        # 获取坐标类型和原子位置
        total_atoms = sum(numbers_of_atoms)
        coordinate_type = lines[7].strip().lower()
        atomic_positions = np.array([list(map(float, lines[8 + i].split()[:3])) for i in range(total_atoms)])

        # 如果坐标是直接坐标，将其转换为笛卡尔坐标
        if coordinate_type == 'direct':
            cartesian_positions = np.dot(atomic_positions, lattice_vectors)
        else:
            cartesian_positions = atomic_positions

        return lattice_vectors, atomic_types, numbers_of_atoms, cartesian_positions
