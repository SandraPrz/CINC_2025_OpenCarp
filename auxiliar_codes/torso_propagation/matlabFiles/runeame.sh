#!/bin/bash
#SBATCH --job-name=torso
#SBATCH -D .
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH -A upv100
#SBATCH --qos=gp_debug  # colas: gp_debug / gp_resa
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --constraint=highmem
#SBATCH --mail-type=all
#SBATCH --mail-user=sandra.perez.upv@gmail.com

WORKDIR=$(pwd)


# Definir los parámetros
matricesFile='/gpfs/scratch/upv100/WORK-SANDRA/pruebas_autoSim/real_ecg_ID4/auxiliar_codes/torso_propagation/Torso/torsoPropagation_allNormal/matricesForIterativeSolver.mat'
potentialFilesFolder='$WORKDIR/post_S2/ens/'
ecgNodeFile='/gpfs/scratch/upv100/WORK-SANDRA/pruebas_autoSim/real_ecg_ID4/auxiliar_codes/torso_propagation/Torso/PSM4_torso_organLabel_preECGNodesFile_ECG_NODES.mat'
output_id='torso_propagation_allNormal'
outputPath='$WORKDIR/post_S2/torso_propagation/'
auxPath='/gpfs/scratch/upv100/WORK-SANDRA/pruebas_autoSim/real_ecg_ID4/auxiliar_codes/torso_propagation/auxiliaryFunctions/'
parallel=1
solver=1
tol_pcg=1e-6
maxIter=500

mkdir -p "$outputPath"
mkdir -p "${outputPath}BSPM"

# Ejecutar MATLAB pasando los parámetros como variables
matlab -nodisplay -r "/gpfs/scratch/upv100/WORK-SANDRA/pruebas_autoSim/real_ecg_ID4/auxiliar_codes/torso_propagation/matlabFiles/step3_runTorsoPropagation('$matricesFile', '$potentialFilesFolder', '$ecgNodeFile', '$output_id', '$outputPath', $parallel, $solver, $tol_pcg, $maxIter); exit;"

wait

matlab -nodisplay -r "Torso_potential('$ecgNodeFile', '${outputPath}${output_id}', '${outputPath}BSPM'); exit;"





