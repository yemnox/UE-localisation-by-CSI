%% explore_o1.m - look at the O1 3.5 GHz v4 scenario before generating CSI
clear; clc; close all;
scen    = fullfile(pwd, 'o1_3p5');
tx_sets = 3:20;                       % the 18 base stations (t003..t020)

% --- base station positions
bs = zeros(numel(tx_sets), 3);
for i = 1:numel(tx_sets)
    p = dm_load(scen, 'tx_pos', tx_sets(i), 0);
    bs(i,:) = p(1,:);
end

% --- the three user grids (r000, r001, r002)
g0 = dm_load(scen, 'rx_pos', 3, 0);
g1 = dm_load(scen, 'rx_pos', 3, 1);
g2 = dm_load(scen, 'rx_pos', 3, 2);
fprintf('Users per grid: r000=%d  r001=%d  r002=%d\n', size(g0,1), size(g1,1), size(g2,1));

% --- units check on one BS / grid 0 (tells generate_csi.m how to read them)
P  = dm_load(scen, 'power',  3, 0);
ph = dm_load(scen, 'phase',  3, 0);
d  = dm_load(scen, 'delay',  3, 0);
el = dm_load(scen, 'aod_el', 3, 0);
az = dm_load(scen, 'aod_az', 3, 0);
fprintf('Max paths stored : %d\n', size(P,2));
fprintf('power  : %8.1f .. %8.1f   (negative => dB)\n', min(P(:),[],'omitnan'),  max(P(:),[],'omitnan'));
fprintf('phase  : %8.1f .. %8.1f   (|x|>2pi => degrees)\n', min(ph(:),[],'omitnan'), max(ph(:),[],'omitnan'));
fprintf('delay  : %8.2e .. %8.2e   (seconds)\n', min(d(:),[],'omitnan'),  max(d(:),[],'omitnan'));
fprintf('aod_az : %8.1f .. %8.1f\n', min(az(:),[],'omitnan'), max(az(:),[],'omitnan'));
fprintf('aod_el : median %.1f    (~90 => zenith angle, ~0 => elevation)\n', median(el(:),'omitnan'));
fprintf('users with no path to BS t003: %d\n', sum(all(isnan(P),2)));

% --- map
figure('Name','O1 3.5 GHz layout');
k = 1:20:size(g0,1); scatter(g0(k,1), g0(k,2), 1, 'b.'); hold on;
k = 1:20:size(g1,1); scatter(g1(k,1), g1(k,2), 1, 'g.');
k = 1:20:size(g2,1); scatter(g2(k,1), g2(k,2), 1, 'm.');
plot(bs(:,1), bs(:,2), 'r^', 'MarkerFaceColor', 'r');
for i = 1:numel(tx_sets)
    text(bs(i,1), bs(i,2), sprintf('  t%03d', tx_sets(i)), 'Color', 'r', 'FontSize', 8);
end
axis equal; grid on; xlabel('x (m)'); ylabel('y (m)');
legend('grid r000','grid r001','grid r002','BS','Location','best');
title('O1 3.5 GHz: user grids and base stations');
