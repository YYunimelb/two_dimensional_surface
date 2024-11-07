def create_poscar(lattice_vectors, atomic_types, positions,output_path):
    # 创建POSCAR文件内容
    content = "Generated POSCAR\n1.0\n"
    for vector in lattice_vectors:
        content += " ".join(map(str, vector)) + "\n"

    # 计算每种原子类型的个数
    atomic_counts = {atom: atomic_types.count(atom) for atom in set(atomic_types)}
    sorted_atomic_types = sorted(atomic_counts.keys())

    content += " ".join(sorted_atomic_types) + "\n"
    content += " ".join(str(atomic_counts[atom]) for atom in sorted_atomic_types) + "\n"
    content += "Cartesian\n"
    for pos in positions:
        content += " ".join(map(str, pos)) + "\n"
    with open(output_path, "w") as f:
        f.write(content)
    return content