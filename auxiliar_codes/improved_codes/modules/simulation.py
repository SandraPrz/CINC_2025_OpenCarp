from carputils import settings
from carputils import tools
import os 

def simulation_eik(job, args, num_stim, stims_list):
    
    velocity=args.myocardial_CV
    # Generate and return base name
    geom_base = args.geom[:-4]
    if geom_base.endswith('_uvc'):
        meshname = geom_base[:-4]

    # Get basic command line, including solver options
    cmd = tools.carp_cmd()

    cmd += tools.gen_physics_opts(EikonalTags=[1,2])

    cmd += ['-num_imp_regions', 1,
            '-imp_region[0].num_IDs', 2,
            '-imp_region[0].ID[0]', 1,
            '-imp_region[0].ID[1]', 2]
    
    if args.model == "OHara":
        cmd += ['-imp_region[0].im', "OHara",
                '-imp_region[0].dream.Idiff.model',   0,
                '-imp_region[0].dream.Idiff.A_F',     0.91,
                '-imp_region[0].dream.Idiff.tau_F',   0.25,
                '-imp_region[0].dream.Idiff.V_th',    -60,
                '-dt', 10]  # OHara: 10 us
                
    elif args.model == "MitchellSchaeffer":
        cmd += ['-imp_region[0].im', "MitchellSchaeffer",
                '-imp_region[0].dream.Idiff.model',   0,
                '-imp_region[0].dream.Idiff.A_F',     1.0,
                '-imp_region[0].dream.Idiff.tau_F',   0.5,
                '-imp_region[0].dream.Idiff.V_th',    0.5,
                '-dt', 100]  # Mitchell: 100 us
                
                
    cmd += ['-num_gregions',           2,
            '-gregion[0].name',        "epi-mid",
            '-gregion[0].num_IDs',      1,
            '-gregion[0].ID[0]',        1,
            '-gregion[0].g_il',         0.8,
            '-gregion[0].g_it',         0.8/3,
            '-gregion[0].g_in',         0.8/3,
            '-gregion[0].g_el',         0.8,
            '-gregion[0].g_et',         0.8/3,
            '-gregion[0].g_en',         0.8/3,
            '-gregion[0].dream.vel_n',   velocity*0.42,#aniso original /2
            '-gregion[0].dream.vel_t',   velocity*0.42,
            '-gregion[0].dream.vel_l',   velocity,

            '-gregion[1].name',        "endo",
            '-gregion[1].num_IDs',      1,
            '-gregion[1].ID[0]',        2,
            '-gregion[1].g_il',         0.8,
            '-gregion[1].g_it',         0.8/3,
            '-gregion[1].g_in',         0.8/3,
            '-gregion[1].g_el',         0.8,
            '-gregion[1].g_et',         0.8/3,
            '-gregion[1].g_en',         0.8/3,
            '-gregion[1].dream.vel_n',   velocity*3.5,
            '-gregion[1].dream.vel_t',   velocity*3.5,
            '-gregion[1].dream.vel_l',   velocity*3.5]#capa rapida esta a *3

#     cmd += ['-num_stim',            num_stim]
#     cmd += stims_list

#    cmd += ['-num_stim',                 1,
#            '-stim[0].crct.type',         0,
#            '-stim[0].pulse.strength',    20.0,
#            '-stim[0].ptcl.duration',     2.0,
#            '-stim[0].ptcl.start',        0.0,
#            '-stim[0].ptcl.npls',         1,
##            '-stim[0].elec.vtx_file',     "node.vtx"]
#            '-stim[0].elec.geom_type',    1,
#            '-stim[0].elec.radius',       600,
#            '-stim[0].elec.p0[0]',        63552,
#            '-stim[0].elec.p0[1]',        118789,
#            '-stim[0].elec.p0[2]',        1071120]
    
    cmd += ['-num_stim', num_stim]

    cmd = cmd + stims_list

    cmd += ['-dream.solve',        2,
            '-dream.fim.max_iter', 500,
            '-dream.fim.tol',      0.001]

    cmd += ['-simID',        job.ID,
            '-meshname',     args.simulation_files,
            '-tend',         args.duration,
            '-output_level', 0,
            '-spacedt',      1,
            '-timedt',       100]
    
    # Run simulation 
    job.carp(cmd)
