import matplotlib
matplotlib.use('Agg')  # Usar backend sin GUI para entornos sin interfaz gráfica
import matplotlib.pyplot as plt
import numpy as np
from modules.simulation_utils import limpiar_directorio

######## FUNCIONES DE VISUALIZACIÓN
           
def graficar_best(all_simulated_signals, ECG_best_normalized, normalized_target_signal,best_error, final=None):
    """
    Genera y guarda gráficas comparativas de las señales ECG simuladas, la mejor señal encontrada,
    y la señal objetivo.
    
    Parámetros:
    ----------
    all_simulated_signals : list
        Lista de señales ECG simuladas
    ECG_best_normalized : numpy.ndarray
        Mejor señal ECG normalizada encontrada hasta el momento
    normalized_target_signal : numpy.ndarray
        Señal ECG objetivo normalizada a comparar
    final : str, optional
        Si es 'yes', guarda las gráficas con nombres específicos para resultados finales
    """
    ECG_best_normalized = np.array(ECG_best_normalized)
    normalized_target_signal = np.array(normalized_target_signal)
    
    # Definir nombres de derivaciones para las etiquetas de las gráficas
    derivation_names = [
        "I", "II", "III",  # Derivaciones estándar
        "aVR", "aVL", "aVF",  # Derivaciones aumentadas
        "V1", "V2", "V3", "V4", "V5", "V6"  # Derivaciones precordiales
    ]
    
    # Convertir a un array tridimensional 
    all_simulated_signals_array = np.array(all_simulated_signals)  # (N, 151, 12)
    
    # Obtener número de simulaciones
    N = all_simulated_signals_array.shape[0]
    
    # Extraer datos de tiempo (igual para todas las simulaciones)
    time=all_simulated_signals_array[0, :, 0]
    time_target = normalized_target_signal[:, 0]
    
    # Crear figura para todas las señales simuladas
    fig, axes = plt.subplots(6, 2, figsize=(10, 12), sharex=True)
    fig.suptitle("Señales ECG Simuladas en sus Posiciones", fontsize=16)
    
    # Graficar cada derivación
    for i in range(12):
        row, col = divmod(i, 2)  # Calcular posición en la cuadrícula 6x2
        
        # Graficar todas las simulaciones en gris transparente
        for sim in range(N):
            axes[row, col].plot(time, all_simulated_signals_array[sim, :, i + 1], 
                               color='black', alpha=0.3)  
        
        # Graficar la mejor señal encontrada
        axes[row, col].plot(time, ECG_best_normalized[:, i+1], 
                           label="Mejor señal", linestyle='--', color='g')
    
        # Graficar la señal objetivo
        axes[row, col].plot(time_target, normalized_target_signal[:, i+1], 
                           label="Señal objetivo", linestyle='dotted', color='r')
        
        # Configurar título y estilo de la gráfica
        axes[row, col].set_title(f'Derivación {derivation_names[i]}')
        axes[row, col].grid()
    
    # Añadir leyenda en la última gráfica
    axes[5, 1].legend(loc="upper right")
    
    # Configurar etiquetas de ejes y ajustar diseño
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.legend()
    
    plt.show()
    
    # Guardar la primera gráfica con nombre apropiado según el parámetro 'final'
    if final == 'yes':
        plt.savefig("ECG_best_final.png")
    else:
        plt.savefig("ECG_best.png")
        
    # Cerrar la figura para liberar memoria
    plt.close()
    
    # Crear segunda figura solo con la mejor señal y la objetivo (versión limpia)
    fig, axes = plt.subplots(6, 2, figsize=(10, 12), sharex=True)
    fig.suptitle(f"Señales ECG\nBest Error: {best_error:.6f}", fontsize=16)
    
    # Graficar cada derivación sin el ruido de fondo de todas las simulaciones
    for i in range(12):
        row, col = divmod(i, 2)
        
        # Graficar solo la mejor señal y la objetivo
        axes[row, col].plot(time, ECG_best_normalized[:, i+1], 
                           label="Mejor señal", linestyle='--', color='g')
        axes[row, col].plot(time_target, normalized_target_signal[:, i+1], 
                           label="Señal objetivo", linestyle='dotted', color='r')
        
        axes[row, col].set_title(f'Derivación {derivation_names[i]}')
        axes[row, col].grid()
    
    # Añadir leyenda
    axes[5, 1].legend(loc="upper right")
    
    # Configurar etiquetas y diseño
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.legend()
    
    plt.show()
    
    # Guardar la segunda gráfica
    if final == 'yes':
        plt.savefig("ECG_best_clean_final.png")
    else:
        plt.savefig("ECG_best_clean.png")
    
    # Eliminar archivos temporales
    limpiar_directorio()


