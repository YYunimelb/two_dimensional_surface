import numpy as np


def create_poscar(lattice_vectors, atomic_types, positions, output_path):
    """Create POSCAR file content with sorted atomic types and corresponding positions."""
    # 确保 atomic_types 和 positions 的顺序一致，按原子类型分组
    unique_types = np.unique(atomic_types)

    # 创建 POSCAR 文件内容
    poscar_content = "Generated POSCAR\n1.0\n"
    for vec in lattice_vectors:
        poscar_content += " ".join(f"{v:.10f}" for v in vec) + "\n"

    # 写入原子类型和数量
    poscar_content += " ".join(unique_types) + "\n"
    counts = [np.sum(np.array(atomic_types) == typ) for typ in unique_types]
    poscar_content += " ".join(map(str, counts)) + "\n"
    poscar_content += "Cartesian\n"

    # 按原子类型分组并排序坐标
    for typ in unique_types:
        for i, atom_type in enumerate(atomic_types):
            if atom_type == typ:
                poscar_content += " ".join(f"{x:.10f}" for x in positions[i]) + "\n"

    # 写入文件
    with open(output_path, "w") as f:
        f.write(poscar_content)

    return poscar_content
