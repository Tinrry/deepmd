import subprocess


# ===== gloabl parameters configuration =====

cell_list = [
    [1,1,1],
    [2,1,1],
    # [3,1,1],
    # [2,2,2],
    # [3,2,2],
    # [3,3,3],
    # [4,3,3],
    # [4,4,4],
    # [5,5,5],
    # [6,6,6],
    # [7,6,6],
    # [7,7,6],
    # [7,7,7],
    # [8,7,7],
    # [8,8,8],
    # [1,1,30],
    # [10,10,10],
    # [10,10,30],
    # [10,10,40],
    # [10,10,63],
    # [10,10,90],
    # [10,10,130],
    # [10,10,150],
    # [100,100,100],
  ]


# ===== submit jobs =====

# each cell generates a job
for i, cell in enumerate(cell_list):
    subprocess.run(["sbatch", "run_job.sh", str(cell[0]), str(cell[1]), str(cell[2])])
    print(f"{i}-th Job submitted")
    print("")
