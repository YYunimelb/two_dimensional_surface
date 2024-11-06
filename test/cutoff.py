from config import vdw_radii,covalent_radii  # 假设 van der Waals 半径数据在 config.py 中



def calculate_dynamic_cutoff( type1, type2):
    # 获取两个索引的原子类型
    radius_sum = covalent_radii[type1] + covalent_radii[type2]
    vdw_sum = vdw_radii[type1] + vdw_radii[type2]
    # 计算动态的阈值
    return (radius_sum + vdw_sum) / 2 * 1


type1 = "S"
type2 = "Au"
print(calculate_dynamic_cutoff(type1,type2))