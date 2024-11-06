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
