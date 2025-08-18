%%function ECG = torsoECG_2_pseudoECGformat(torso_nodes_file, V_mat, factor_escala,nombre_ecg_aiso)
    torso_nodes_file='./torso_nodes.txt';
    file=dir('torso_propagation*');
    factor_escala=1;
    nombre_ecg_aiso='';

    data_torso = load(file(1).name);
    factor_escala = factor_escala;
    
    % Abrir el fichero para lectura
    fileID = fopen(torso_nodes_file, 'r');
    
    % Verificar si el archivo se abrió correctamente
    if fileID == -1
        error('No se puede abrir el archivo.');
    end
    
    % Leer el contenido del archivo
    fileContent = textscan(fileID, '%s', 'Delimiter', '\n');
    fclose(fileID);
    
    % Convertir el contenido a una celda de cadenas
    lines = fileContent{1};
    
    % Estructura para almacenar las variables
    nodes = struct();
    
    % Procesar cada línea para extraer nombres de variables y valores
    for i = 1:length(lines)
        line = strtrim(lines{i});  % Eliminar espacios en blanco al inicio y final
        if ~isempty(line)
            % Dividir la línea en nombre y valor
            parts = strsplit(line, '=');
            varName = strtrim(parts{1});  % Nombre de la variable
            valueStr = strtrim(parts{2});  % Valor como cadena
    
            % Eliminar el punto y coma final
            valueStr = strrep(valueStr, ';', '');
    
            % Convertir el valor a numérico
            value = str2double(valueStr);
    
            % Asignar el valor a la estructura
            nodes.(varName) = value;
        end
    end
    
    T = data_torso.time;
    V = data_torso.V;
    ecgNodes = data_torso.ecgNodes;
    
    %create an empty array to store the potentials of the leads
    ECG_torso = [];
    
    %nodes +1
    nodes.nodeLA = nodes.nodeLA + 1;
    nodes.nodeRA = nodes.nodeRA + 1;
    nodes.nodeLL = nodes.nodeLL + 1;
    nodes.nodeV1 = nodes.nodeV1 + 1;
    nodes.nodeV2 = nodes.nodeV2 + 1;
    nodes.nodeV3 = nodes.nodeV3 + 1;
    nodes.nodeV4 = nodes.nodeV4 + 1;
    nodes.nodeV5 = nodes.nodeV5 + 1;
    nodes.nodeV6 = nodes.nodeV6 + 1;
    
    V_V1 = V(ecgNodes == nodes.nodeV1, :) * factor_escala;
    V_V2 = V(ecgNodes == nodes.nodeV2, :) * factor_escala;
    V_V3 = V(ecgNodes == nodes.nodeV3, :) * factor_escala;
    V_V4 = V(ecgNodes == nodes.nodeV4, :) * factor_escala;
    V_V5 = V(ecgNodes == nodes.nodeV5, :) * factor_escala;
    V_V6 = V(ecgNodes == nodes.nodeV6, :) * factor_escala;
    V_LA = V(ecgNodes == nodes.nodeLA, :) * factor_escala;
    V_RA = V(ecgNodes == nodes.nodeRA, :) * factor_escala;
    V_LL = V(ecgNodes == nodes.nodeLL, :) * factor_escala;
    
    V_total = [T, V_V1', V_V2', V_V3', V_V4', V_V5', V_V6', V_LA', V_RA', V_LL'];
    
    ECG = V_total;
    % Crear el nombre del archivo usando strcat y num2str para convertir factor_escala a cadena
    nombre_archivo = strcat('ecg_aiso_torso_fct_', num2str(factor_escala), '.dat');

    % Ahora puedes usar writematrix con el nombre del archivo construido
	dlmwrite(nombre_archivo, ECG, 'delimiter', ',');
%%end