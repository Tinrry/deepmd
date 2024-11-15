from ase import Atoms
from ase.io import read, write
from deepmd.calculator import DP
from ase.build.supercells import make_supercell

from datetime import datetime
import numpy as np

import os



# 获取当前工作目录
current_directory = os.getcwd()
print("当前目录:", current_directory)

model = '01_tc6001/frozen_model_compressed.pb'


# get memory usage
def get_memory_usage():
    # return: the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024   # in MB
    return mem

def gen_supercel(cell):
    system = read('geometry/POSCAR')
    ix, iy, iz = zip(cell)
    print(f"create supercel", [ix,iy,iz])

    supercell_name = f'geometry/POSCAR_alloy_{ix}x{iy}x{iz}.vasp'
    if not os.path.exists(supercell_name):
        # 创建超包
        M = [[ix, 0, 0], [0, iy, 0], [0, 0, iz]]
        sc=make_supercell(system, M)
        write(supercell_name, sc, vasp5=True, sort=True, direct=False)
    return supercell_name
    

# @profile(precision=2,stream=open('memory_profiler.log','w+'))
def main():
    with open('output.txt', 'a') as f:    #  todo wirte to <jobid>_energy.txt          
        f.write(f"ix\tiy\tiz\tn_supercell\tn_atoms\t\tenergy\t\ttime\tmemory(MB)\n")

        ix, iy, iz = 2, 2, 2
        supercell_name = gen_supercel(cell=[ix, iy, iz])

        sc = read(supercell_name, format='vasp')
        sc.calc = DP(model)
        start_time = datetime.now()
        # 计算能量
        print("开始计算能量值。")
        en = sc.get_potential_energy()
        # 记录结束时间
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        # 保存
        
        f.write(f"{ix}\t{iy}\t{iz}\t{ix*iy*iz}\t\t{len(sc)}\t\t{en:.3f}\t{elapsed_time.total_seconds():.1f}\t{get_memory_usage():.1f}\n")
    
if __name__=='__main__':
    main()
