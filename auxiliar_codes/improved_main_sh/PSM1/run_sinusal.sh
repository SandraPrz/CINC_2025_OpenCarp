#!/bin/bash

	
	
	
# Asignar argumentos a variables
MANUAL_IDS="$1"
MANUAL_TIMES="$2"
MODULES_PATH_OPENCARP="$3"
EVAL_DIR="$4"
GEOM="$5"
SIM_FILES="$6"
LEAD_FILE="$7"
TORSO_FILE_UM="$8"
TORSO_FILE="$9"
CONFIG_TEMPLATE="${10}"


# Definir ruta del archivo de log dentro del EVAL_DIR
LOGFILE="$EVAL_DIR/run_sinusal.log"


# Ejecutar el script Python con todos los argumentos
python "$MODULES_PATH_OPENCARP/run_sinusal.py" \
  --job-name "$EVAL_DIR/simulation_results" \
  --geom "$GEOM" \
  --simulation-files "$SIM_FILES" \
  --sinusal-mode manual \
  --manual-ids "$MANUAL_IDS" \
  --manual-times "$MANUAL_TIMES" \
  --save-vtk False \
  --lead-file "$LEAD_FILE" \
  --torso-file-um "$TORSO_FILE_UM" \
  --torso-file "$TORSO_FILE" \
  --config-template "$CONFIG_TEMPLATE" \
  --initial-time 10 