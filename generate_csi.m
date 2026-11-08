%% generate_csi.m - build CSI tensors (antennas x subcarriers) from DeepMIMO v4 paths
% Run from: C:\Users\LENOVO\Desktop\RT5\5G\UE localisation
% Output  : csi_o1.mat  with H (N x 32 x 32 complex), pos (N x 3), los (N x 1)
clear; clc; close all;
scen = fullfile(pwd, 'o1_3p5');

%% ---------------- CONFIG (adjust after running explore_o1.m) ----------------
TX        = 6;                       % base station tx set (t005); pick from the map
GRID      = 0;                       % user grid r000
BOX       = [-inf inf 400 800];     % [xmin xmax ymin ymax] crop in metres
MAX_USERS = 30000;                   % random subsample of the valid users
N_ANT     = 32;                      % BS ULA elements, half-wavelength spacing
N_SUB     = 32;                      % OFDM subcarriers
BW        = 50e6;                    % bandwidth (Hz)
N_PATHS   = 20;                      % strongest paths kept per user
rng(0);                              % reproducible subsampling
%% ---------------------------------------------------------------------------

fprintf('Loading tx set t%03d, grid r%03d ...\n', TX, GRID);
pos = dm_load(scen, 'rx_pos', TX, GRID);
P   = dm_load(scen, 'power',  TX, GRID);

% users that receive at least one path and fall inside the crop box
valid = any(~isnan(P), 2) & ...
        pos(:,1) >= BOX(1) & pos(:,1) <= BOX(2) & ...
        pos(:,2) >= BOX(3) & pos(:,2) <= BOX(4);
idx = find(valid);
fprintf('Valid users: %d\n', numel(idx));
if numel(idx) > MAX_USERS
    idx = sort(idx(randperm(numel(idx), MAX_USERS)));
end
N = numel(idx);
fprintf('Users kept : %d\n', N);

% load the remaining path parameters for the kept users only
P   = P(idx,:);
ph  = dm_load(scen, 'phase',  TX, GRID);  ph  = ph(idx,:);
tau = dm_load(scen, 'delay',  TX, GRID);  tau = tau(idx,:);
az  = dm_load(scen, 'aod_az', TX, GRID);  az  = az(idx,:);
el  = dm_load(scen, 'aod_el', TX, GRID);  el  = el(idx,:);
itr = dm_load(scen, 'inter',  TX, GRID);  itr = itr(idx,:);
pos = pos(idx,:);

% keep the N_PATHS strongest paths per user (missing paths are NaN)
[~, ord] = sort(P, 2, 'descend', 'MissingPlacement', 'last');
ord = ord(:, 1:min(N_PATHS, size(P,2)));
li  = sub2ind(size(P), repmat((1:N)', 1, size(ord,2)), ord);
P = P(li); ph = ph(li); tau = tau(li); az = az(li); el = el(li); itr = itr(li);

% ---- unit handling (explore_o1.m prints the raw ranges)
if max(P(:), [], 'omitnan') < 0
    amp = sqrt(10.^(P/10));          % power stored in dB
else
    amp = sqrt(P);                   % power stored linear
end
if max(abs(ph(:)), [], 'omitnan') > 2*pi + 0.1
    ph = deg2rad(ph);                % phase stored in degrees
end
if median(el(:), 'omitnan') > 45
    sin_th = sind(el);               % el is a zenith angle (90 = horizon)
else
    sin_th = cosd(el);               % el is an elevation angle (0 = horizon)
end
dir_y = sin_th .* sind(az);          % projection on the ULA axis

% ---- channel: H(m,k) = sum_l g_l * exp(j*pi*m*dir_l) * exp(-j*2*pi*f_k*tau_l)
m = (0:N_ANT-1)';
f = (0:N_SUB-1) * BW / N_SUB;
H = complex(zeros(N, N_ANT, N_SUB, 'single'));
for u = 1:N
    ok = ~isnan(P(u,:));
    g  = amp(u,ok) .* exp(1j*ph(u,ok));            % 1 x L
    A  = exp(1j*pi * m * dir_y(u,ok));             % N_ANT x L
    D  = exp(-1j*2*pi * tau(u,ok).' * f);          % L x N_SUB
    H(u,:,:) = reshape(single((A .* g) * D), [1 N_ANT N_SUB]);
    if mod(u, 5000) == 0, fprintf('  %d / %d\n', u, N); end
end

% LoS flag (interaction code 0 = direct path), useful for LoS/NLoS error analysis
los    = any(itr == 0, 2);
pos    = single(pos);
bs_pos = dm_load(scen, 'tx_pos', TX, GRID); bs_pos = bs_pos(1,:);
cfg    = struct('scenario','o1_3p5', 'tx_set',TX, 'grid',GRID, 'n_ant',N_ANT, ...
                'n_sub',N_SUB, 'bw',BW, 'n_paths',N_PATHS, 'fc',3.5e9);

fprintf('H: %s   pos: %s   LoS users: %.1f%%\n', mat2str(size(H)), mat2str(size(pos)), 100*mean(los));
save('csi_o1.mat', 'H', 'pos', 'los', 'bs_pos', 'cfg', '-v7');
fprintf('Saved csi_o1.mat\n');

%% ---- sanity figures (reusable in the article / slides)
rx_dB = 10*log10(squeeze(sum(sum(abs(H).^2, 2), 3)));
figure('Name','Received power map');
scatter(pos(:,1), pos(:,2), 2, rx_dB, 'filled'); hold on;
plot(bs_pos(1), bs_pos(2), 'r^', 'MarkerFaceColor', 'r', 'MarkerSize', 10);
axis equal; colorbar; xlabel('x (m)'); ylabel('y (m)');
title(sprintf('Received power (dB), BS t%03d', TX));

figure('Name','CSI image of one user');
subplot(1,2,1); imagesc(abs(squeeze(H(1,:,:))));   colorbar; title('|H|');
xlabel('subcarrier'); ylabel('antenna');
subplot(1,2,2); imagesc(angle(squeeze(H(1,:,:)))); colorbar; title('angle(H)');
xlabel('subcarrier'); ylabel('antenna');
