"""
Example to run TESoptimize with a standard TES montage to optimize the strength 
of the normal component of the electric field in the ROI.

Copyright (c) 2024 SimNIBS developers. Licensed under the GPL v3.
"""
from simnibs import opt_struct


''' Initialize structure '''
opt = opt_struct.TesFlexOptimization()
opt.subpath = 'm2m_ernie'                               # path of m2m folder containing the headmodel
opt.output_folder = "tes_optimize_tes_Enormal_intensity"

''' Set up goal function '''
opt.goal = "mean"                                       # maximize the mean of "normal" in the ROI ("normal" defined by e_postproc)
opt.e_postproc = "normal"                               # postprocessing of e-fields ("magn": magnitude,
                                                        # "normal": normal component, "tangential": tangential component)
''' Define electrodes and array layout '''
electrode_layout = opt.add_electrode_layout("ElectrodeArrayPair")   # Pair of TES electrode arrays (here: 1 electrode per array)
electrode_layout.length_x = [70]                                    # x-dimension of electrodes
electrode_layout.length_y = [50]                                    # y-dimension of electrodes
electrode_layout.dirichlet_correction_detailed = False              # account for inhomogenous current distribution at electrode-skin interface (slow)
electrode_layout.current = [0.002, -0.002]                          # electrode currents

''' Define ROI '''
roi = opt.add_roi()
roi.method = "surface"
roi.surface_type = "central"                            # define ROI on central GM surfaces
roi.roi_sphere_center_space = "subject"                 
roi.roi_sphere_center = [-41, -13,  66]                 # center of spherical ROI in subject space (in mm)
roi.roi_sphere_radius = 20                              # radius of spherical ROI (in mm)
# uncomment for visual control of ROI:
#roi.subpath = opt.subpath
#roi.write_visualization('','roi.msh')

''' Run optimization '''
opt.run()
