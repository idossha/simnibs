"""
Example to run TESoptimize with an 4x1 center-surround TES montage to 
optimize the field intensity in the ROI

Copyright (c) 2024 SimNIBS developers. Licensed under the GPL v3.
"""
from simnibs import opt_struct


''' Initialize structure '''
opt = opt_struct.TesFlexOptimization()
opt.subpath = "m2m_ernie"                                            # path of m2m folder containing the headmodel
opt.output_folder = "tes_optimze_4x1tes_intensity"

''' Set up goal function '''
opt.goal = "mean"                                                    # maximize the mean of "magn" in the ROI ("magn" defined by e_postproc)
opt.e_postproc = "magn"                                              # postprocessing of e-fields ("magn": magnitude, 
                                                                     # "normal": normal component, "tangential": tangential component)
''' Define electrodes and array layout '''
electrode_layout = opt.add_electrode_layout("CircularArray")                # Nx1 center surround montage
electrode_layout.radius_inner = 10                                          # radius of inner electrode
electrode_layout.radius_outer = 10                                          # radius of outer electrodes
electrode_layout.distance_bounds = [25, 100]                                # distance bounds between inner and outer electrodes
electrode_layout.n_outer = 4                                                # number of outer electrodes
electrode_layout.dirichlet_correction = False                               # set to True when all outer electrodes are connected to the same channel (slower)
electrode_layout.current = [0.002, -0.002/4, -0.002/4, -0.002/4, -0.002/4]  # initial currents

''' Define ROI '''
roi = opt.add_roi()
roi.method = "surface"
roi.surface_type = "central"                                         # define ROI on central GM surfaces
roi.roi_sphere_center_space = "subject"
roi.roi_sphere_center = [-41.0, -13.0,  66.0]                        # center of spherical ROI in subject space (in mm)
roi.roi_sphere_radius = 20                                           # radius of spherical ROI (in mm)
# for visual control of ROI:
#roi.subpath = opt.subpath
#roi.write_visualization('','roi.msh')

''' Run optimization '''
opt.run()
