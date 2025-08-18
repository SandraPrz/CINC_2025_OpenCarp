# Created by: Jorge Sanchez
import os

def stimBlock(mesh_ids, stimTimes, folder_path, num_stim=0, filename="node"):
    """! Get the stimulation nodes from the mesh and write the stimulation command
    @param mesh_ids The global mesh ids
    @param stimTimes The time of stimulation of the points to stimulate
    @param num_stim The number of stimulations to do
    @param folder_path The path to folder output
    
    @return num_stim The number of stimulations to do
    @return stims_list The list of stimulations to do
    """
    
    stim_dir = os.path.join(folder_path, "stim") 
    os.makedirs(stim_dir, exist_ok=True)
    
    # Group LATs and write cmd
    stims_list = []
    i=0

    for idx, stimTime in zip(mesh_ids, stimTimes):

        vtx_path = os.path.join(stim_dir, f"{filename}{i}.vtx")
        
        #with open('./stim/' + filename + str(i) + '.vtx', "w") as f:
        with open(vtx_path, "w") as f:
            f.write("{}\n".format(1))
            f.write("intra\n") 
            f.write("{}\n".format(str(idx)))
        # Write cmd
        stimfactor = 1
        
        
        
        stims_list += ['-stim[' + str(num_stim) + '].crct.type',           0,
                        '-stim[' + str(num_stim) + '].pulse.strength',     stimfactor,
                        '-stim[' + str(num_stim) + '].ptcl.duration',      2.0,
                        '-stim[' + str(num_stim) + '].ptcl.start',         stimTime,
                        '-stim[' + str(num_stim) + '].ptcl.npls',          1,
                        #'-stim[' + str(num_stim) + '].elec.vtx_file',      filename + str(i) + '.vtx']                        
                        '-stim[' + str(num_stim) + '].elec.vtx_file',      vtx_path
                        ]
        num_stim += 1
        i += 1
            
    return num_stim, stims_list