// load matlab_GIFTI toolbox into MATLAB


// get all subjects
all_sub_dir = '/data_qnap/yifeis/HCP_7T/';
all_subjects = dir(all_sub_dir);
all_subjects = {all_subjects([all_subjects.isdir]).name};
all_subjects = all_subjects(~ismember(all_subjects ,{'.','..'}));
N = length(all_subjects);
fprintf('There are %d subjects.\n', N);

// each subject
for i = 1:N
    sub = all_subjects{i};
    // load right and left hemisphere surfaces
    surfaceMidL = gifti(strcat('/data_qnap/yifeis/new/', sub, '/MNINonLinear/fsaverage_LR32k/', sub, '.L.midthickness.32k_fs_LR.surf.gii'));
    surfaceMidR = gifti(strcat('/data_qnap/yifeis/new/', sub, '/MNINonLinear/fsaverage_LR32k/', sub, '.R.midthickness.32k_fs_LR.surf.gii'));

    // get gifti data
    sub_dir = strcat(all_dir, sub);
    fprintf('The %s direcotry: %s \n', sub, sub_dir);
    all_files = dir(sub_dir);
    all_files = {all_files.name};
    all_files = all_files(~ismember(all_files ,{'.','..'}));
    n = length(all_files);
    fprintf('The %s  has %d files.\n', sub, n);
    // each gifti file
    for x =1:n
        // load the gifti data
        f = all_files{x}; // 'ses_left/right.func.gii'
        f_dir = strcat(sub_dir, '/', f)
        fmri_data = gifti(f_dir);

        // whether the data is left or right
        if contains(f, 'right')
            surf = surfaceMidR
        else contains(f, 'left')
            surf = surfaceMidL

        // copy vertices and faces from surface file to fMRI data
        fmri_data.vertices = surf.vertices;
        fmri_data.faces = surf.surface;

        // save to vtk
        saveas(fmri_data, strcat(sub_dir, '/', f(1:end-9), '.vtk'), 'legacy -ascii')

    end
end
