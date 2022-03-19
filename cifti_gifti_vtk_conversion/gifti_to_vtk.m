% load matlab_GIFTI toolbox into MATLAB
addpath matlab_GIfTI

% get all subjects
all_sub_dir = '/data_qnap/yifeis/HCP_7T/';
all_subjects = dir(all_sub_dir);
all_subjects = {all_subjects([all_subjects.isdir]).name};
all_subjects = all_subjects(~ismember(all_subjects ,{'.','..'}));
N = length(all_subjects);
fprintf('There are %d subjects.\n', N);

% each subject
for i = 9:10
    disp(' ');
    sub = all_subjects{i};
    % load right and left hemisphere surfaces
    surfaceMidL = gifti(strcat('/data_qnap/yifeis/new/', sub, '/MNINonLinear/', sub, '.L.midthickness_MSMAll.164k_fs_LR.surf.gii'));
    surfaceMidR = gifti(strcat('/data_qnap/yifeis/new/', sub, '/MNINonLinear/', sub, '.R.midthickness_MSMAll.164k_fs_LR.surf.gii'));

    % get gifti data
    sub_dir = strcat(all_sub_dir, sub);
    fprintf('The %s direcotry: %s \n', sub, sub_dir);
    all_files = dir(sub_dir);
    all_files = {all_files.name};
    all_files = all_files(~ismember(all_files ,{'.','..'}));
    n = length(all_files);
    fprintf('The %s  has %d files.\n', sub, n);

    % each gifti file
    for x =1:n
        % load the gifti data
        f = all_files{x};
        if contains(f, '.func.gii') & contains(f, 'rest') % 'ses_left/right.func.gii'
            f_dir = strcat(sub_dir, '/', f);
            disp(f_dir);
            fmri_data = gifti(f_dir);

            % whether the data is left or right
            if contains(f, 'right')
                surf = surfaceMidR;
            elseif contains(f, 'left')
                surf = surfaceMidL;
            end
            % copy vertices and faces from surface file to fMRI data
            fmri_data.vertices = surf.vertices;
            fmri_data.faces = surf.faces;

            % save to vtk
            saveas(fmri_data, strcat(sub_dir, '/', f(1:end-9), '.vtk'), 'legacy-ascii')
            disp(strcat(sub_dir, '/', f(1:end-9), '.vtk'));

%             % save as mat
%             save(strcat(sub_dir, '/', f(1:end-9), '.mat'),'fmri_data');
%             disp(strcat(sub_dir, '/', f(1:end-9), '.mat'));
        end
    end
end
