import numpy as np
from ase import Atoms
from ase.io import read, write

# get memory usage
def get_memory_usage():
    # return: the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024   # in MB
    return mem


# expand cubic supercell
def expand_cubic_supercell(atoms, n):
    # atoms: the atoms in the unit cell
    # n: the number of supercell in each direction
    # return: the atoms in the supercell
    cell = atoms.get_cell()
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    natoms = len(positions)
    new_positions = []
    new_symbols = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(natoms):
                    new_positions.append(positions[l] + i * cell[0] + j * cell[1] + k * cell[2])
                    new_symbols.append(symbols[l])
    new_atoms = Atoms(symbols=new_symbols, positions=new_positions, cell=n * cell, pbc=True)
    return new_atoms

# predict atoms in the supercell, use dp.get_energyforces_stress() to predict the energy, forces and stress
def predict_atoms_in_supercell(model_file: "Path", atoms: "ase.Atoms"):
    # dp: the deep potential model
    # atoms: the atoms in the supercell
    # return: the atoms in the supercell with predicted energy, forces and stress
    from deepmd.calculator import DP
    dp = DP(model=model_file)
    atoms.set_calculator(dp)
    energy = atoms.get_potential_energy()
    return energy

import os

START_SUPER_CELL = 1
SUPER_CELL = 2

if __name__ == "__main__":
    atoms = read('POSCAR_MgAlCu', format='vasp')         # read the atoms from the POSCAR file, 464 atoms, mp-1200279
    # expand the cubic supercell
    for i in range(1, SUPER_CELL):
        if os.path.exists(f'POSCAR_{i}**3'):
            continue
        new_atoms = expand_cubic_supercell(atoms, i)            # expand the cubic supercell, 464 * i**3 atoms
        write(f'POSCAR_{i}**3', new_atoms, format='vasp')

    with open("atoms_vs_mem.csv", "a") as f:
        # read the atoms from the POSCAR file
        f.write(f"n_atoms\tenergy_per_atom\tmemory_usage\tprocess_time\n")
        for i in range(START_SUPER_CELL, SUPER_CELL):
            new_atoms = read(f'POSCAR_{i}**3', format='vasp')    # read the atoms from the POSCAR file, 464 * i**3 atoms
            # use deepmd as the calculator to predict the energy, forces and stress
            import time
            start = time.time()
            energy = predict_atoms_in_supercell("01.train/graph_compressed.pb", new_atoms)
            # write the predicted atoms to csv file
            n_atoms = new_atoms.get_global_number_of_atoms()
            energy_per_atom = energy / n_atoms

            f.write(f"{n_atoms}\t{energy_per_atom}\t{get_memory_usage()}\t{time.time() - start}\n")
