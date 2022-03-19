% load matlab_GIFTI toolbox into MATLAB
addpath matlab_GIfTI

% load left and right hemisphere surfaces
surfaceMidL = gifti('/data_qnap/yifeis/new/100610/MNINonLinear/100610.L.midthickness_MSMAll.164k_fs_LR.surf.gii');
surfaceMidR = gifti('/data_qnap/yifeis/new/100610/MNINonLinear/100610.R.midthickness_MSMAll.164k_fs_LR.surf.gii');

% load left and right
fmriL = gifti('/data_qnap/yifeis/HCP_7T/100610/retbar1_left.func.gii');
fmriR = gifti('/data_qnap/yifeis/HCP_7T/100610/retbar1_right.func.gii');

% copy vertices and faces from surface files to the timeseries data
fmriL.vertices = surfaceMidL.vertices;
fmriL.faces = surfaceMidL.faces;
fmriR.vertices = surfaceMidR.vertices;
fmriR.faces = surfaceMidR.faces;

% write to vtk
saveas(fmriL, '/data_qnap/yifeis/HCP_7T/100610/retbar1_left.vtk', 'legacy-ascii');
saveas(fmriR, '/data_qnap/yifeis/HCP_7T/100610/retbar1_right.vtk', 'legacy-ascii');

% save as mat
save('/data_qnap/yifeis/HCP_7T/100610/retbar1_left.mat','fmriL');
save('/data_qnap/yifeis/HCP_7T/100610/retbar1_right.mat','fmriR');