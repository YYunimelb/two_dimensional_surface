# main.py
import shutil

from parsers import VaspParser
from utils import SupercellBuilder, LayerAnalyzer
from pathlib import Path
import numpy as np
from parsers import VaspParser
from utils import SupercellBuilder, LayerAnalyzer,LayerChecker,StructureProcessor,StructureNormalizer

import os
from pymatgen.io.vasp import Poscar
from mp_api.client import MPRester
from utils.structure_processor import StructureProcessor,BulkTo2DTransformer
from utils.surface import SurfaceAtomIdentifier
from utils.out_format_control import print_surface_summary,format_surface_atoms_output

def print_surface_summary(surface_markers):
    """
    Print the positions (1-based index) of atoms on the top surface (1), bottom surface (-1), and isolated (3).

    Parameters:
    - surface_markers (list): List of surface markers where 1 = top surface, -1 = bottom surface, 3 = isolated atom.
    """
    top_surface_indices = [i + 1 for i, marker in enumerate(surface_markers) if marker == 1]
    bottom_surface_indices = [i + 1 for i, marker in enumerate(surface_markers) if marker == -1]
    isolated_indices = [i + 1 for i, marker in enumerate(surface_markers) if marker == 3]

    print("Top surface atoms (1):", top_surface_indices)
    print("Bottom surface atoms (-1):", bottom_surface_indices)
    print("Isolated atoms (3):", isolated_indices)



def get_structure_from_mp(api_key):
    """查询Materials Project数据库，获取结构数据并判断是否为层状结构,"""

    # 初始化 MPRester，使用 Materials Project 的 API 密钥
    with MPRester(api_key) as mpr:
        # 从 Materials Project 查询结构，过滤掉超过49个原子的结构
        docs = mpr.summary.search(
            theoretical=False,
            fields=["material_id", "structure"],
            chunk_size=200,
            num_sites=(None, 49),  # 设置原子个数上限为49
            material_ids=[f"mp-{i}" for i in range(0,1000)]  # 包含从mp-0到mp-50000的所有Materials Project ID
        )


        # 定义需要排除的锕系元素集合
        actinides = {
            "Ac", "Th", "Pa", "U", "Np", "Pu", "Pm", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",  # 锕系元素
        }

        # 遍历查询结果
        for doc in docs:
            mp_structure = doc.structure
            contains_actinides = any(element.symbol in actinides for element in mp_structure.species)

            # 如果结构中包含锕系元素，则跳过
            if contains_actinides:
                continue

            # 获取材料的ID
            mp_id = doc.material_id
            print(mp_id)

            try:
                # 将结构写入 POSCAR 文件
                poscar = Poscar(mp_structure)
                poscar_file_path = f"data/structure/POSCAR_{mp_id}"
                poscar.write_file(poscar_file_path)

                # 使用 StructureProcessor 和 LayerChecker 检查结构是否为层状
                processor = StructureProcessor(poscar_file_path, supercell_boundry=(-2, 2, -2, 2, -2, 2),
                                               cutoff_factor=1.0)
                processor.process_structure()
                layer_checker = LayerChecker(processor)
                result = layer_checker.analyze_structure()

                # 根据结果处理文件
                if result == "layered":
                    print(f"{mp_id} : layered")
                else:
                    os.remove(poscar_file_path)  # 如果不是层状结构，删除该 POSCAR 文件

            except Exception as e:
                print(f"Error processing {mp_id}: {e}")
def verify_surface_atom_rule(surface_result):
    """
    Verify the rule for surface atoms based on valence electrons and bonded atoms.

    Parameters:
    - surface_result (dict): Dictionary with 'top_surface' and 'bottom_surface' containing atom details.

    Returns:
    - bool: True if the rule is satisfied for all surface atoms, otherwise False.
    """
    # Define valence electrons for relevant elements
    valence_electrons = {"H":1,
        "B": 3, "C": 4, "N": 5, "O": 6, "F": 7,  # Group 13 to 17
        "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7,
        "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7,
        "In": 3, "Sn": 4, "Sb": 5, "Te": 6, "I": 7,
        "Tl": 3, "Pb": 4, "Bi": 5
    }

    def check_atoms(surface_atoms):
        """
        Check whether all atoms in the given surface satisfy the rules.

        Parameters:
        - surface_atoms (list): List of atom dictionaries.

        Returns:
        - bool: True if all atoms satisfy the rule, False otherwise.
        """

        for atom in surface_atoms:
            element = atom["element"]
            if element == "H" or element == "O"  or element == "S":
                continue
            bond_count = min(atom["bond_count"], 3)  # Limit bonded atoms to a maximum of 3
            if element not in valence_electrons:
                return False  # Element not allowed on the surface
            if valence_electrons[element] + bond_count < 8:
                return False  # Rule not satisfied for this atom
        return True

    # Check top surface atoms
    if not check_atoms(surface_result["top_surface"]):
        return False

    # Check bottom surface atoms
    if not check_atoms(surface_result["bottom_surface"]):
        return False

    return True

def single_structure_check(file_path):
    #   file_path = "test/POSCAR_mp-341"  #
    processor = StructureProcessor(file_path, supercell_boundry=(-2, 2, -2, 2, -2, 2), cutoff_factor=1.0)
    processor.process_structure()
    #print(processor.layers)
    layer_checker = LayerChecker(processor)
    result = layer_checker.analyze_structure()
    print("The structure is:", result)

def main(file_path):
    #API_KEY = "mY30L5L7yZr48BNMeqkS9U9Zum6MHNpK"
    #get_structure_from_mp(API_KEY)
    single_structure_check(file_path)

    #file_path = f"data/structure_not_satisfy/POSCAR_mp-{i}"

    processor = StructureProcessor(file_path, supercell_boundry=(-2, 2, -2, 2, -2, 2), cutoff_factor=1.0)
    processor.process_structure()
    surface_identifier = SurfaceAtomIdentifier(processor)
    print(surface_identifier.check_layer_connectivity())


    # normalizer = StructureNormalizer(processor)
    # normalizer.convert_to_normal_structure(output_path="POSCAR_bulk")
    #
    #
    # file_path = "POSCAR_bulk"
    # processor = StructureProcessor(file_path, supercell_boundry=(-2, 2, -2, 2, -2, 2), cutoff_factor=1.0)
    # processor.process_structure()
    #
    #
    #
    #
    # transformer = BulkTo2DTransformer(processor)
    # transformer.transform_to_2d(output_path="POSCAR_2D")
    #
    # file_path = "POSCAR_2D"
    # processor = StructureProcessor(file_path, supercell_boundry=(-2, 2, -2, 2, 0, 0), cutoff_factor=1.0)
    # processor.process_structure()
    # surface_identifier = SurfaceAtomIdentifier(processor)
    # surface_markers = surface_identifier.find_surface_atoms()
    # surface_result = surface_identifier.analyze_bonded_surface_atoms()

    #print_surface_summary(surface_markers)
    #print(format_surface_atoms_output(surface_result))

    #
    #

if __name__ == "__main__":
    """ 2578 ,  3439,  3468  ,   3849,   4160, 4906, 5824 ,6023, 7049 , 7277 ,  7784  ,8093,  8094 , 8190, 8378 ,
              8586 ,   8800,  8806, 8946, 9396, 9622, 9815  """
    path  =  "test/POSCAR_mp-541775"
    main(path)
