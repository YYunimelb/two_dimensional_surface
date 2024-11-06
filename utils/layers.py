# utils/layers.py
import numpy as np
from utils.supercell import SupercellBuilder
from utils.geometry import are_points_collinear
from config import vdw_radii  # 假设 van der Waals 半径数据在 config.py 中

class LayerAnalyzer:
    def __init__(self, atomic_types, positions, all_positions, distances, cutoff_factor=1.2):
        self.atomic_types = atomic_types
        self.positions = positions
        self.all_positions = all_positions
        self.distances = distances
        self.cutoff_factor = cutoff_factor

    def calculate_dynamic_cutoff(self, index1, index2):
        # 获取两个索引的原子类型
        radius_sum = vdw_radii[self.atomic_types[index1]] + vdw_radii[self.atomic_types[index2]]
        # 计算动态的阈值
        return radius_sum * self.cutoff_factor

    def layer_marking(self):
        num_original = len(self.positions)
        num_total = len(self.all_positions)
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
                            dynamic_cutoff = self.calculate_dynamic_cutoff(index, neighbor_index)
                            if self.distances[index, neighbor_index] < dynamic_cutoff:
                                layers[neighbor_index] = layer_number
                                next_indices.append(neighbor_index)

                current_indices = list(set(next_indices))

            layer_number += 1

        return layers



# utils/layer_checker.py
import numpy as np
from utils.geometry import are_points_collinear

class LayerChecker:
    def __init__(self, structure_processor):
        """初始化 LayerChecker，直接接收 StructureProcessor 实例"""
        self.atomic_types = structure_processor.atomic_types
        self.lattice_vectors = structure_processor.lattice_vectors
        self.positions = structure_processor.positions
        self.supercell_positions = structure_processor.supercell_positions
        self.supercell_atomic_types = structure_processor.supercell_atomic_types
        self.layers = structure_processor.layers
        self.cutoff_factor = structure_processor.cutoff_factor

    def find_equivalent_atoms(self, central_atom_index, total_atoms, target_layer):
        """查找等效原子"""
        num_types = len(self.atomic_types)
        potential_equivalents = [(central_atom_index + i * num_types) for i in range(total_atoms // num_types)]
        equivalent_atoms = [index for index in potential_equivalents if
                            index < total_atoms and self.layers[index] == target_layer]
        return equivalent_atoms

    def check_layers(self):
        """检查 layers 的唯一性，判断是否为 bulk"""
        unique_elements, counts = np.unique(self.layers, return_counts=True)
        if len(unique_elements) == 1:
            return "bulk"
        if len(unique_elements) == 2:
            index_of_1 = np.where(unique_elements == 1)
            if index_of_1[0].size > 0:
                count_of_1 = counts[index_of_1[0][0]]
                if count_of_1 > counts.sum() / 2:
                    return "bulk"
        return "layered"

    def analyze_structure(self):
        """判断是否为层状结构的主要逻辑"""
        result = self.check_layers()

        if result == "bulk":
            return "bulk"

        unique_layers = []
        idx = []
        for index, layer in enumerate(self.layers[:len(self.positions)]):
            if layer not in unique_layers:
                idx.append(index)
                unique_layers.append(layer)

        all_layered = True
        for index in idx:
            central_atom_index = index
            layer_of_central_atom = self.layers[central_atom_index]
            equivalent_atoms = self.find_equivalent_atoms(central_atom_index, len(self.supercell_positions),
                                                          layer_of_central_atom)

            # 检查等效原子数量并判断是否共线
            if len(equivalent_atoms) >= 3:
                # 如果等效原子共线，则符合层状结构
                if  are_points_collinear([self.supercell_positions[i] for i in equivalent_atoms]):
                    all_layered = False
                    break
            else:
                all_layered = False
                break

        return "layered" if all_layered else "bulk"
