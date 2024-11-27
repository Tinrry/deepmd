import dpdata
from ase import Atoms
from ase.io import read, write
from deepmd.calculator import DP
from ase.build.supercells import make_supercell

from datetime import datetime
import numpy as np
import pandas as pd
from ase import Atoms


import os
import glob

# calculator = DP(model="/home/shengbi/workspace/sciData/DeepMD/acc/alloy/pretrained/AlMgCu_fintune_3E6.pb")
calculator = DP(model="01train/frozen_model_compressed.pb")

# Define the base pattern with wildcards
# systems = ["AlMgCu.sys"]
systems = ['AlMgCu.cmcm']
for sys in systems:
    pattern = f"01train/POSCAR.{sys}.*/02.md/sys-*/deepmd"

    # Use glob to find all matching paths
    matching_paths = glob.glob(pattern)

    # Filter out only existing paths
    existing_paths = [path for path in matching_paths if os.path.exists(path)]


    df = pd.DataFrame(columns=('atoms', 'err_energy', 'err_force_x','err_force_y','err_force_z'))
    # Print the existing matching paths
    for path in existing_paths:
        print("system: ", path)
        multi_systems_dp = dpdata.LabeledSystem(path,fmt="deepmd/npy")
        multi_systems_ase = multi_systems_dp.to_ase_structure()

        for s_ase,s_dp in zip(multi_systems_ase,multi_systems_dp):
            # s_ase = s_dp.to_ase_structure()
            # print(s_ase)
            s_ase.set_calculator(calculator)
            en = s_ase.get_potential_energy()
            force = s_ase.get_forces()
            err_force = s_dp["forces"][0]-force
            # Adding the new row using append
            atoms = len(s_ase)
            new_row={ 
                'atoms': atoms, 
                'err_energy': (s_dp["energies"][0]-en)/atoms*1000., 
                'err_force_x': np.mean(err_force[:][0]*1000.),
                'err_force_y': np.mean(err_force[:][1]*1000.),
                'err_force_z': np.mean(err_force[:][2]*1000.)}
            df.loc[len(df)] = new_row
    df.to_csv(f'dp_{sys}_error.csv', index=False)  
    # df.to_csv(f'dp_{sys}_error.dat', sep='\t', index=False) 

    df_mean = df.groupby('atoms').mean().reset_index()
    df_mean.to_csv(f'dp_analyse_{sys}_error.csv', index=False)  
    # df_mean.to_csv(f'dp_analyse_{sys}_error.dat', sep='\t', index=False) 
