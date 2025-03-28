%
%  This example wil go throgh simulations and calculate
%  the average of the electric field magnitude in MNI space
%
% It is a follow-up to the "run_simulations_group.m" example
%    
% Copyright (c) 2019 SimNIBS developers. Licensed under the GPL v3.

subjects = {'sub01', 'sub09', 'sub10', 'sub12', 'sub15'};
results_folder = 'mni_volumes';
field_name = 'magnE';

mni_image_suffix = ['_TDCS_1_scalar_MNI_' field_name '.nii.gz'];

template_image = nifti_load(fullfile('bipolar', subjects{1}, results_folder, [subjects{1} mni_image_suffix]));
field_avg = zeros(size(template_image.vol));

for i = 1:length(subjects)
    sub = subjects{i};
    % load the nifti image
    img = nifti_load(fullfile(pwd, 'bipolar', sub, results_folder, [sub mni_image_suffix]));
    % accumulate values
    field_avg = field_avg + img.vol;
end

% calculate mean
field_avg = field_avg/length(subjects);

% save
avg_image = template_image;
avg_image.vol = field_avg;
nifti_save(avg_image, ['mean_' field_name '.nii.gz']);