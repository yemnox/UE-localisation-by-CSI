function x = dm_load(scen, name, t, r)
% Load one DeepMIMO v4 matrix, e.g. dm_load(scen,'power',5,0)
%   -> o1_3p5/power_t005_tx000_r000.mat
% Works whatever the variable inside the .mat is called.
f = fullfile(scen, sprintf('%s_t%03d_tx000_r%03d.mat', name, t, r));
s = load(f);
c = struct2cell(s);
x = double(c{1});
end
