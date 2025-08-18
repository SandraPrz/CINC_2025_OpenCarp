import os
import re 
import numpy as np
import matplotlib
#matplotlib.use('Agg')  # Usar backend sin GUI para entornos sin interfaz gráfica
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.signal import correlate
from scipy.stats import pearsonr
import shutil
###################################################################################################
####
####                    PROCESAMIENTO DE FICHEROS
####
###################################################################################################

def extraer_info_desde_archivo(ruta_archivo):
    """
    Extrae el nodo_archivo desde un archivo que contiene el ID en la tercera línea.
    """
    try:
        with open(ruta_archivo, 'r') as f:
            lineas = f.readlines()
            if len(lineas) >= 3:
                nodo_archivo = int(lineas[2].strip())
                return nodo_archivo
            else:
                print(f"[AVISO] Formato inesperado en {ruta_archivo}.")
    except Exception as e:
        print(f"[ERROR] No se pudo leer {ruta_archivo}: {e}")
    
    return None

def calcular_preexcitacion(nombre_carpeta,global_id, nodo_archivo, log_path="errores_preexcitacion.txt"):
    """Devuelve el prefijo numérico del global_id antes del nodo_archivo."""
    global_str = str(global_id)
    nodo_str = str(nodo_archivo)
    
    if global_str.endswith(nodo_str):
        preexcitacion_str = global_str[:-len(nodo_str)]
        if preexcitacion_str:  # si hay algo antes del nodo
            return int(preexcitacion_str)
        else:
            return 0  # si global_id es igual al nodo_archivo
    else:
        # Guardar discrepancia en log
        if nombre_carpeta is not None:
            with open(log_path, 'a') as f:
                f.write(f"{nombre_carpeta} | PreexcitationTIme + GlobalID: {global_str} | Nodo_archivo: {nodo_str}\n")
    return None
def procesar_carpetas(directorio_raiz, archivo_objetivo='node_acc_region0.vtx', guardar_ecg='yes'):
    resultados = []

    for carpeta in os.listdir(directorio_raiz):
        ruta_carpeta = os.path.join(directorio_raiz, carpeta)

        # Saltar si no es una carpeta
        if not os.path.isdir(ruta_carpeta):
            print(f"[INFO] No es una carpeta: {ruta_carpeta}. Se omite.")
            continue

        # Verificar si la carpeta está vacía
        if not os.listdir(ruta_carpeta):
            print(f"[AVISO] Carpeta vacía: {ruta_carpeta}. Se omite.")
            continue

        try:
            partes = carpeta.split("_")
            if len(partes) < 5:
                print(f"[AVISO] Nombre de carpeta no válido (formato incorrecto): {carpeta}")
                continue

            nodo_preexcitation_GlobalID = re.search(r'(\d+)$', partes[-1])
            if not nodo_preexcitation_GlobalID:
                print(f"[AVISO] No se encontró ID en el nombre de la carpeta: {carpeta}")
                continue
            nodo_preexcitation_GlobalID = int(nodo_preexcitation_GlobalID.group(1))

            ruta_stim = os.path.join(ruta_carpeta, "stim")
            ruta_leads = os.path.join(ruta_carpeta, "ecg_output.dat")

            if not os.path.isdir(ruta_stim):
                print(f"[AVISO] Carpeta 'stim' no encontrada en {carpeta}.")
                continue

            archivos_stim = os.listdir(ruta_stim)
            if not archivos_stim:
                print(f"[AVISO] Carpeta 'stim' vacía en {carpeta}.")
                continue

            # Procesar TODOS los archivos en 'stim' que coincidan con el patrón
            for archivo_stim in archivos_stim:
                if archivo_stim == archivo_objetivo:  # Solo procesar el archivo objetivo
                    ruta_archivo = os.path.join(ruta_stim, archivo_stim)
                    
                    if os.path.isfile(ruta_archivo):
                        nodo_archivo = extraer_info_desde_archivo(ruta_archivo)
                        if nodo_archivo is not None:
                            preexcitacion = calcular_preexcitacion(carpeta, nodo_preexcitation_GlobalID, nodo_archivo)
                            
                            # Verificar si el archivo ECG existe
                            if not os.path.isfile(ruta_leads):
                                print(f"[ERROR] Archivo ECG no encontrado: {ruta_leads}")
                                continue
                                
                            ecg_calculated = ecg_calcul(ruta_leads)
                            ecg_calculated_normalized = ecg_calcul_normalized(ruta_leads)
                            
                            resultados.append((carpeta, nodo_archivo, preexcitacion, ecg_calculated, ecg_calculated_normalized))
                            
                            graficar_simple(ecg_calculated, nodo_archivo, preexcitacion, ruta_carpeta, guardar=guardar_ecg)
                    else:
                        print(f"[AVISO] Archivo {ruta_archivo} no encontrado.")

        except Exception as e:
            print(f"[ERROR] Fallo al procesar carpeta {carpeta}: {str(e)}")
            continue

    return resultados


def recopilar_pngs(directorio_raiz, carpeta_salida='ecg-output_figure'):
    """
    Recorre todas las carpetas dentro de 'directorio_raiz', busca archivos .png
    y los copia a una carpeta de salida.
    """
    # Crear carpeta de salida si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    # Contador opcional
    contador = 0

    # Recorrer todas las subcarpetas y archivos
    for carpeta_actual, subdirs, archivos in os.walk(directorio_raiz):
        for archivo in archivos:
            if archivo.lower().endswith('.png'):
                ruta_origen = os.path.join(carpeta_actual, archivo)

                # Si hay archivos con el mismo nombre, se renombran con un índice único
                nuevo_nombre = f"{contador:05d}_{archivo}"
                ruta_destino = os.path.join(carpeta_salida, nuevo_nombre)

                shutil.copy2(ruta_origen, ruta_destino)
                contador += 1



def borrar_pngs(directorio, incluir_subcarpetas=False):
    """
    Elimina todos los archivos .png dentro del directorio especificado.

    Parámetros:
    -----------
    directorio : str
        Ruta al directorio donde buscar los archivos .png.
    incluir_subcarpetas : bool
        Si True, busca también en subdirectorios (recursivamente).
    """
    if incluir_subcarpetas:
        for raiz, _, archivos in os.walk(directorio):
            for archivo in archivos:
                if archivo.lower().endswith('.png'):
                    ruta_archivo = os.path.join(raiz, archivo)
                    try:
                        os.remove(ruta_archivo)
                        print(f"[BORRADO] {ruta_archivo}")
                    except Exception as e:
                        print(f"[ERROR] No se pudo borrar {ruta_archivo}: {e}")
    else:
        for archivo in os.listdir(directorio):
            ruta_archivo = os.path.join(directorio, archivo)
            if os.path.isfile(ruta_archivo) and archivo.lower().endswith('.png'):
                try:
                    os.remove(ruta_archivo)
                    print(f"[BORRADO] {ruta_archivo}")
                except Exception as e:
                    print(f"[ERROR] No se pudo borrar {ruta_archivo}: {e}")


###################################################################################################
####
####                    PROCESAMIENTO DE SEÑALES
####
###################################################################################################

def normalizar_señales(matriz):
    """
    Normaliza una matriz de señales ECG. Para cada señal (columna), 
    resta la media y divide por la desviación estándar, excepto la primera columna (tiempo).
    
    Parámetros:
    matriz : numpy.ndarray - Matriz de tamaño (n, 13), donde cada columna representa una señal.
        
    Retorna:
    numpy.ndarray - Matriz normalizada donde la primera columna queda intacta 
                    y las demás tienen media 0 y desviación estándar 1.
    """
    # Copiar la matriz original para no modificarla directamente
    matriz_normalizada = matriz.copy()

    # Calcular la media y la desviación estándar para cada columna (excepto la primera)
    media = np.mean(matriz[:, 1:], axis=0)
    desviacion_estandar = np.std(matriz[:, 1:], axis=0)

    # Evitar división por cero: si la desviación estándar es 0, reemplazar por 1
    desviacion_estandar[desviacion_estandar == 0] = 1  

    # Normalizar todas las columnas excepto la primera (tiempo)
    matriz_normalizada[:, 1:] = (matriz[:, 1:] - media) / desviacion_estandar

    return matriz_normalizada


def ecg_calcul(ecgLeads_path):
    """
    Calcula las 12 derivaciones del ECG a partir del fichero ecgLeads_path con los potenciales
    proporcionados por elvira, sin normalizar los resultados.
    
    Parámetros:
    ecgLeads_path : str - Ruta al archivo de potenciales
    
    Retorna:
    numpy.ndarray - Matriz con las 12 derivaciones del ECG sin normalizar
    """
    # Cargar el archivo de datos
    ecgLeads = np.loadtxt(ecgLeads_path, skiprows=1)

    # Crear un array vacío para almacenar las derivaciones
    ECG = np.zeros((ecgLeads.shape[0], 13))

    # Asignar la primera columna como el tiempo
    ECG[:, 0] = ecgLeads[:, 0]

    # Cálculo de las derivaciones estándar
    ECG[:, 1] = ecgLeads[:, 7] - ecgLeads[:, 8]  # I = LA - RA
    ECG[:, 2] = ecgLeads[:, 9] - ecgLeads[:, 8]  # II = LL - RA
    ECG[:, 3] = ecgLeads[:, 9] - ecgLeads[:, 7]  # III = LL - LA

    # Cálculo de las derivaciones aumentadas
    ECG[:, 4] = ecgLeads[:, 8] - 0.5 * (ecgLeads[:, 7] + ecgLeads[:, 9])  # aVR = RA - (1/2)(LA + LL)
    ECG[:, 5] = ecgLeads[:, 7] - 0.5 * (ecgLeads[:, 8] + ecgLeads[:, 9])  # aVL = LA - (1/2)(RA + LL)
    ECG[:, 6] = ecgLeads[:, 9] - 0.5 * (ecgLeads[:, 8] + ecgLeads[:, 7])  # aVF = LL - (1/2)(RA + LA)

    # Cálculo de las derivaciones precordiales
    ECG[:, 7] = ecgLeads[:, 1] - (1/3) * (ecgLeads[:, 8] + ecgLeads[:, 7] + ecgLeads[:, 9])  # V1
    ECG[:, 8] = ecgLeads[:, 2] - (1/3) * (ecgLeads[:, 8] + ecgLeads[:, 7] + ecgLeads[:, 9])  # V2
    ECG[:, 9] = ecgLeads[:, 3] - (1/3) * (ecgLeads[:, 8] + ecgLeads[:, 7] + ecgLeads[:, 9])  # V3
    ECG[:, 10] = ecgLeads[:, 4] - (1/3) * (ecgLeads[:, 8] + ecgLeads[:, 7] + ecgLeads[:, 9])  # V4
    ECG[:, 11] = ecgLeads[:, 5] - (1/3) * (ecgLeads[:, 8] + ecgLeads[:, 7] + ecgLeads[:, 9])  # V5
    ECG[:, 12] = ecgLeads[:, 6] - (1/3) * (ecgLeads[:, 8] + ecgLeads[:, 7] + ecgLeads[:, 9])  # V6
    
    return ECG
    


def ecg_calcul_normalized(ecgLeads_path):
    """
    Calcula las 12 derivaciones del ECG a partir del fichero ecgLeads_path con los potenciales
    proporcionados por elvira y normaliza las señales resultantes.
    
    Parámetros:
    ecgLeads_path : str - Ruta al archivo de potenciales
    
    Retorna:
    numpy.ndarray - Matriz con las 12 derivaciones normalizadas del ECG
    """
    # Cargar el archivo de datos
    ecgLeads = np.loadtxt(ecgLeads_path, skiprows=1)

    # Crear un array vacío para almacenar las derivaciones
    ECG = np.zeros((ecgLeads.shape[0], 13))

    # Asignar la primera columna como el tiempo
    ECG[:, 0] = ecgLeads[:, 0]

    # Cálculo de las derivaciones estándar
    ECG[:, 1] = ecgLeads[:, 7] - ecgLeads[:, 8]  # I = LA - RA
    ECG[:, 2] = ecgLeads[:, 9] - ecgLeads[:, 8]  # II = LL - RA
    ECG[:, 3] = ecgLeads[:, 9] - ecgLeads[:, 7]  # III = LL - LA

    # Cálculo de las derivaciones aumentadas
    ECG[:, 4] = ecgLeads[:, 8] - 0.5 * (ecgLeads[:, 7] + ecgLeads[:, 9])  # aVR = RA - (1/2)(LA + LL)
    ECG[:, 5] = ecgLeads[:, 7] - 0.5 * (ecgLeads[:, 8] + ecgLeads[:, 9])  # aVL = LA - (1/2)(RA + LL)
    ECG[:, 6] = ecgLeads[:, 9] - 0.5 * (ecgLeads[:, 8] + ecgLeads[:, 7])  # aVF = LL - (1/2)(RA + LA)

    # Cálculo de las derivaciones precordiales
    ECG[:, 7] = ecgLeads[:, 1] - (1/3) * (ecgLeads[:, 8] + ecgLeads[:, 7] + ecgLeads[:, 9])  # V1
    ECG[:, 8] = ecgLeads[:, 2] - (1/3) * (ecgLeads[:, 8] + ecgLeads[:, 7] + ecgLeads[:, 9])  # V2
    ECG[:, 9] = ecgLeads[:, 3] - (1/3) * (ecgLeads[:, 8] + ecgLeads[:, 7] + ecgLeads[:, 9])  # V3
    ECG[:, 10] = ecgLeads[:, 4] - (1/3) * (ecgLeads[:, 8] + ecgLeads[:, 7] + ecgLeads[:, 9])  # V4
    ECG[:, 11] = ecgLeads[:, 5] - (1/3) * (ecgLeads[:, 8] + ecgLeads[:, 7] + ecgLeads[:, 9])  # V5
    ECG[:, 12] = ecgLeads[:, 6] - (1/3) * (ecgLeads[:, 8] + ecgLeads[:, 7] + ecgLeads[:, 9])  # V6

    # Normalizar las señales resultantes
    ECG = normalizar_señales(ECG)
    return ECG



def align_and_cross_correlate(signal1, signal2, reference_lead=0, reference='signal2', 
                              write_corr_images=None, save_path='correlaciones'):
    """
    Alinea temporalmente dos señales de ECG de forma multicanal y calcula la correlación cruzada. Programado para solo tener en cuenta las precordiales

    Parameters:
        signal1, signal2: np.ndarray de forma (derivaciones), se espera 12 derivaciones.
            # Cálculo de las derivaciones estándar
             ECG[:, 0] =I 
             ECG[:, 1] =II 
             ECG[:, 2] =III
    
            # Cálculo de las derivaciones aumentadas
            ECG[:, 3] =  aVR 
            ECG[:, 4] = aVL 
            ECG[:, 5] = aVF 
    
            # Cálculo de las derivaciones precordiales
            ECG[:, 6] =  V1
            ECG[:, 7] =  V2
            ECG[:, 8] =  V3
            ECG[:, 9] = V4
            ECG[:, 10] = V5
            ECG[:, 11] = V6
            
        reference_lead: señal que se tiene en cuenta para calcular el desplzamiento
        reference: 'signal1' o 'signal2'. Especifica cuál es la referencia fija.
        save_path: str, carpeta donde guardar los plots

    Returns:
        aligned_signal1, aligned_signal2: señales alineadas
        cross_corrs: correlaciones por canal
        mean_cross_corr: media de las correlaciones por canal
    """
    
    assert signal1.shape == signal2.shape, "Las señales tienen que tener la misma longitud"
    assert reference in ['signal1', 'signal2'], "Referencia debe ser 'signal1' o 'signal2'"

    signal1_total=signal1
    signal2_total=signal2
    
    signal1=signal1[:,7:]
    signal2=signal2[:,7:]
    
       
    
    sig_ref = signal1 if reference == 'signal1' else signal2
    sig_mov = signal2 if reference == 'signal1' else signal1
    
    sig_ref_total = signal1_total if reference == 'signal1' else signal2_total
    sig_mov_total = signal2_total if reference == 'signal1' else signal1_total
    
    

    num_leads = sig_ref.shape[1]
    num_leads_total = sig_ref_total.shape[1]
    
    
    aligned_moving = np.zeros_like(sig_mov)
    aligned_moving_total = np.zeros_like(sig_mov_total)
    

    corr = correlate(sig_ref[:, reference_lead], sig_mov[:, reference_lead], mode='full')
    lag = np.argmax(corr) - (len(sig_ref) - 1)
    
    for i in range(0, num_leads):
        # Alinear aplicando desfase
        if lag > 0:
            aligned = np.pad(sig_mov[:, i], (lag, 0), mode='constant')[:len(sig_mov)]
            
        else:
            aligned = np.pad(sig_mov[:, i], (0, -lag), mode='constant')[-lag:len(sig_mov)-lag]
        aligned_moving[:, i] = aligned
        
        
    for i in range(0, num_leads_total):
        # Alinear aplicando desfase
        if lag > 0:
            aligned_total = np.pad(sig_mov_total[:, i], (lag, 0), mode='constant')[:len(sig_mov_total)]
            
        else:
            aligned_total = np.pad(sig_mov_total[:, i], (0, -lag), mode='constant')[-lag:len(sig_mov_total)-lag]
        aligned_moving_total[:, i] = aligned_total

    # Reasignar según cuál era la señal móvil
    aligned_signal1 = signal1 if reference == 'signal1' else aligned_moving
    aligned_signal2 = aligned_moving if reference == 'signal1' else signal2
    
    aligned_signal1_total = signal1_total if reference == 'signal1' else aligned_moving_total
    aligned_signal2_total = aligned_moving_total if reference == 'signal1' else signal2_total

    # Correlación por canal
    cross_corrs = [
        pearsonr(aligned_signal1[:, i], aligned_signal2[:, i])[0]
        for i in range(0, num_leads)
    ]
    mean_cross_corr = np.mean(cross_corrs)

    # --- PLOT de señales alineadas ---
    if write_corr_images:
        # Crear la carpeta si no existe
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        deriv_names = ['I','II','III','aVR','aVL','aVF','V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(15, 20))
        axes = axes.flatten()

        for i in range(0, num_leads_total-1):  # Solo V1-V6 (índices del 6 al 11 en las señales, pero en el plot son 0-5)
            #axes[i].plot(aligned_signal1[:, i ], label='Monodomain-simulated ECG', color='red', linestyle='-', linewidth=2.5, alpha=0.9)
            #axes[i].plot(aligned_signal2[:, i ], label='Clinical ECG', color='black', linestyle='--', linewidth=2.5, alpha=0.9)
            #axes[i].set_title(f'{deriv_names[i]} | Corr: {cross_corrs[i]:.2f}',fontsize=22)
            axes[i].plot(aligned_signal1_total[:, i+1 ], label=f'{deriv_names[i]}', color='red', linestyle='-', linewidth=2.5, alpha=0.9)
            
            if i>=6:
                           
                axes[i].plot(aligned_signal2_total[:, i+1], label=f'CC = {cross_corrs[i-6]:.2f}', color='black', linestyle='--', linewidth=2.5, alpha=0.9)
            else:
                axes[i].plot(aligned_signal2_total[:, i+1 ], color='black', linestyle='--', linewidth=2.5, alpha=0.9)
            
            
            axes[i].legend(frameon=False,prop={"size":22},loc='upper left')
            # Mostrar el eje Y solo en la primera columna (i = 0, 2, 4)
            if i % 2 == 0:
                axes[i].set_ylabel('Voltage (mV)',fontsize=20)
                axes[i].tick_params(labelsize=20)
                axes[i].set_yticks([-3,-2, -1, 0, 1, 2,3])
            else:
                axes[i].tick_params(labelleft=False)
                
            if i > 9:
            # Eje X para todos
                axes[i].set_xlabel('Time (ms)',fontsize=20)
                axes[i].tick_params(labelsize=20)
            else:
                axes[i].tick_params(labelbottom=False)
                
            axes[i].set_ylim(-3, 3)
                
    
            
        # Mostrar la leyenda solo en el último gráfico
       # axes[-1].legend(fontsize='large')



        
        
        # Crear el título según los parámetros disponibles
       
        
        full_title=f"Mean correlation: {mean_cross_corr:.2f}"
        
       
        plt.suptitle(f' ECG({full_title})', fontsize=26)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        plt.show()
        
        plt.close()
        
    return aligned_signal1, aligned_signal2, cross_corrs, mean_cross_corr



###################################################################################################
####
####                    GENERACIÓN DE GRÁFICAS
####
###################################################################################################

           
def graficar_simple(ECG_calculated, node_GlobalID, preexcitation_time, ruta_carpeta, guardar='yes'):
    ECG_calculated = np.array(ECG_calculated)

    derivation_names = [
        "I", "II", "III",  # Derivaciones estándar
        "aVR", "aVL", "aVF",  # Derivaciones aumentadas
        "V1", "V2", "V3", "V4", "V5", "V6"  # Derivaciones precordiales
    ]

    time = ECG_calculated[:, 0]

    fig, axes = plt.subplots(6, 2, figsize=(10, 12), sharex=True)
    titulo = f"ECG // nodoID:{node_GlobalID} // preexcitationTime: {preexcitation_time}"
    fig.suptitle(titulo, fontsize=16)

    for i in range(12):
        row, col = divmod(i, 2)
        axes[row, col].plot(time, ECG_calculated[:, i+1],
                            label="ECG", linestyle='--', color='g')
        axes[row, col].set_title(f'Derivación {derivation_names[i]}')
        axes[row, col].grid()

    axes[5, 1].legend(loc="upper right")

    fig.supxlabel("Tiempo (s)")
    fig.supylabel("Amplitud")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # ✅ Guardar antes de mostrar o cerrar
    if guardar == 'yes':
        os.makedirs(ruta_carpeta, exist_ok=True)
        nombre_archivo = f"ECG_node{node_GlobalID}_preex{preexcitation_time}.png"
        ruta_guardado = os.path.join(ruta_carpeta, nombre_archivo)
        fig.savefig(ruta_guardado)

    # plt.show()
    plt.close(fig)   


         

def graficar_by_preexcitationTime(resultados, guardar='yes', ruta_guardado='.'):
    derivation_names = [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6"
    ]
    
    agrupado = defaultdict(list)
    for carpeta, nodo_archivo, preexcitacion, ecg, ecg_calculated_normalized in resultados:
        agrupado[preexcitacion].append(ecg)
    
    for preexc_time, ecgs in agrupado.items():
        ecgs_array = np.array(ecgs)  # (N, 151, 12)
        N = ecgs_array.shape[0]
        
        time = np.linspace(0, 1, ecgs_array.shape[1])
        
        fig, axes = plt.subplots(6, 2, figsize=(12, 14), sharex=True)
        fig.suptitle(f"Señales ECG Simuladas - Preexcitación {preexc_time}", fontsize=16)
        
        for i in range(12):
            row, col = divmod(i, 2)
            for sim in range(N):
                axes[row, col].plot(time, ecgs_array[sim, :, i+1], color='black', alpha=0.3)
            axes[row, col].set_title(f'Derivación {derivation_names[i]}')
            axes[row, col].grid()
        
        fig.supxlabel("Tiempo (s)")
        fig.supylabel("Amplitud")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if guardar == 'yes':
            os.makedirs(ruta_guardado, exist_ok=True)
            ruta = os.path.join(ruta_guardado, f"ECG_preexc_{preexc_time}.png")
            fig.savefig(ruta)
            

        plt.show()
        plt.close(fig)
        
        
def graficar_max_correlation(resultados, target_ecg_normalized_path, guardar='no'):
    import matplotlib.pyplot as plt
    import numpy as np

    derivation_names = [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6"
    ]
    
    all_signals = []
    all_signals_normalized = []
    resultados_corr = {}

    normalized_target_signal = np.loadtxt(target_ecg_normalized_path, delimiter=',')

    for carpeta, nodo_archivo, preexcitacion, ecg, ecg_calculated_normalized in resultados:
        all_signals.append(ecg)
        all_signals_normalized.append(ecg_calculated_normalized)

        ecg_calculated_normalized = ecg_calculated_normalized[0:111, :]

        aligned_signal1, aligned_signal2, cross_corrs, mean_cross_corr = align_and_cross_correlate(
            signal1=ecg_calculated_normalized,
            signal2=normalized_target_signal,
            reference_lead=0,
            reference='signal2'
        )

        resultados_corr[(nodo_archivo, preexcitacion)] = (mean_cross_corr, ecg_calculated_normalized)

    max_key = max(resultados_corr, key=lambda k: resultados_corr[k][0])
    max_corr, mejor_ecg = resultados_corr[max_key]
    nodo_max, preex_max = max_key

    print(f"Máxima correlación: {max_corr:.4f} para nodo {nodo_max} con preex {preex_max}")

    time = normalized_target_signal[:, 0]

    fig, axes = plt.subplots(6, 2, figsize=(10, 12), sharex=True)
    fig.suptitle(f"ECG con Máxima Correlación (Nodo: {nodo_max}, Preex: {preex_max}, Corr: {max_corr:.4f})", fontsize=16)

    for i in range(12):
        row, col = divmod(i, 2)
        axes[row, col].plot(time, normalized_target_signal[:, i+1], label="Objetivo", linestyle='dotted', color='red')
        axes[row, col].plot(time, mejor_ecg[:, i+1], label="Mejor ECG", linestyle='--', color='green')
        axes[row, col].set_title(f'Derivación {derivation_names[i]}')
        axes[row, col].grid()

    axes[5, 1].legend(loc="upper right")
    fig.supxlabel("Tiempo (s)")
    fig.supylabel("Amplitud")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if guardar == 'yes':
        fig.savefig("ECG_max_corr.png")
        print("Figura guardada como: ECG_max_corr.png")

    plt.show()
    plt.close(fig)


    
    
  

# USO:
directorio_raiz = 'F:\LL_new'
resultados = procesar_carpetas(directorio_raiz,guardar_ecg='yes')
recopilar_pngs(directorio_raiz)

# graficar_by_preexcitationTime(resultados,guardar='yes')

#target_ecg_normalized_path = 'E:/SIMULACIONES/NEW_simulation_ctrl/CINC_2025/auxiliar_codes/mesh_ID6/ID6_ECG_filtrado.csv'  
#graficar_max_correlation(resultados,target_ecg_normalized_path,guardar='yes')

# borrar_pngs(directorio_raiz, incluir_subcarpetas=True)