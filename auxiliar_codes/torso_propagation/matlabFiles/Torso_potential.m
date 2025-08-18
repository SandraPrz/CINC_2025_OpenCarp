%% Torso potential
%%% by Edison
%%% It creates .ENS from torso mesh & surface potentials, to create
%%% animations of potential propagation on the torso with paraview 

function Torso_potential(input_VTKtorso,input_potentials,out_folder_path)


disp(input_VTKtorso)
disp(input_potentials)
disp(out_folder_path)

vtk = fopen(input_VTKtorso,'r');        % Apertura de archivo VTK de superficie del torso      
while 1
    tline=fgetl(vtk);                               % Lectura de línea de texto 
    if strfind(tline,'POINTS')                      % Verifica si alcanzó la palabra POINTS
        auxnod = regexp(tline,'\d*','Match');
        nnd  = eval(cell2mat(auxnod));              % Determinación número de nodos
        node = fscanf(vtk,'%f %f %f',[3, nnd])';    % Extracción de coordenadas de nodos
    end
    if strfind(tline,'POLYGONS')                    % Verifica si alcanzó la palabra POLYGONS
        auxelm = regexp(tline,'\d*','Match');               
        nelm = eval(cell2mat(auxelm(1)));           % Determinación número de elementos
        elm  = fscanf(vtk,'%i %i %i %i',[4, nelm])';% Extracción de id de nodos que forman cada elemento
        elm  = elm(:,2:4)+1; 
        break    
    end
end

%%
if ~exist(out_folder_path, 'dir')
    mkdir(out_folder_path)  % Crea carpeta para guardado de imagenes del ECG
end

geo=fopen(strcat(out_folder_path,'/','Torso_ENS_0.geo'),'w');
fprintf(geo,'Ensight Model Geometry File\n');
fprintf(geo,'\t\t\t\t\t\t\t\t\t\t SIMULATOR OUTPUT\n');
fprintf(geo,'node id given\n');
fprintf(geo,'element id given\n');
fprintf(geo,'part\n');
fprintf(geo,'%10i\n',1);
fprintf(geo,'Model, Geometry %10i\n',1);
fprintf(geo,'coordinates\n');
fprintf(geo,'%10i\n',nnd);
fprintf(geo,'%10i\n',(1:nnd)');
fprintf(geo,'%10i\n',node(:));
fprintf(geo,'tria3\n');
fprintf(geo,'%10i\n',nelm);
fprintf(geo,'%10i\n',(1:nelm)');
fprintf(geo,'%10i %10i %10i\n',elm');
fclose(geo);



load(input_potentials);         % Apertura de archivo de potenciales del torso
ensFiles = length(time); 
L = numel(num2str(ensFiles));       % Número de archivos de potencial
id = strcat('Torso_ENS_Vn_', repmat('0', 1, 8-L), repmat('*', 1, L), '.ens');

VnCase=fopen(strcat(out_folder_path,'/','Torso_ENS_Vn.case'),'w');
fprintf(VnCase,'FORMAT\n');
fprintf(VnCase,'type: \t\t\t\t\t\t\t ensight gold\n\n');
fprintf(VnCase,'GEOMETRY\n');
fprintf(VnCase,'model: \t\t\t\t\t\t\t Torso_ENS_0.geo\n\n');
fprintf(VnCase,'VARIABLE\n');
fprintf(VnCase,'scalar per node: \t\t\t\t Potential2 \t\t\t\t %s\n\n',id);
fprintf(VnCase,'TIME\n');
fprintf(VnCase,'time set: %16i\n',1);
fprintf(VnCase,'number of steps: %11i\n',ensFiles);
fprintf(VnCase,'filename start number: %3i\n',1);
fprintf(VnCase,'filename increment: %6i\n',1);
fprintf(VnCase,'time values:\n');
fprintf(VnCase,'%13.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f\n',time');
fclose(VnCase);

for i = 1:ensFiles
    nfile = sprintf('%08d',i);  % Genera número de archivo .ens
    ens=fopen(strcat(out_folder_path,'/','Torso_ENS_Vn_',nfile,'.ens'),'w');
    fprintf(ens,'Ensight Model Post Process\n');
    fprintf(ens,'part\n');
    fprintf(ens,'%10i\n',1);
    fprintf(ens,'coordinates');
    fprintf(ens,'\n%1.6E',V(:,i));
    fclose(ens);
end
end