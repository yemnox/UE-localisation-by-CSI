%% plot_rays.m - draw the ray-traced paths between BS t006 and two users
% Loads the interaction points of every path (inter_pos). This file is large
% (~400 MB on disk), so the first load takes a while and needs a few GB of RAM.
clear; close all;
root = fileparts(fileparts(mfilename('fullpath')));   % repo root
scen = fullfile(root, 'data', 'o1_3p5');
TX = 6; GRID = 0;

rx  = dm_load(scen, 'rx_pos', TX, GRID);
bs  = dm_load(scen, 'tx_pos', TX, GRID); bs = bs(1,:);
P   = dm_load(scen, 'power',  TX, GRID);
itr = dm_load(scen, 'inter',  TX, GRID);

d = hypot(rx(:,1) - bs(1), rx(:,2) - bs(2));
ok = any(~isnan(P), 2) & rx(:,2) > 400 & rx(:,2) < 800;
cand = find(ok);
[~, i1] = min(abs(d(cand) - 30));  [~, i2] = min(abs(d(cand) - 150));
users = cand([i1 i2]);

fprintf('Loading inter_pos (slow) ...\n');
ip = dm_load(scen, 'inter_pos', TX, GRID);
fprintf('inter_pos size: %s  (expected users x paths x interactions x 3)\n', mat2str(size(ip)));
paths = cell(1, 2);
for k = 1:2
    paths{k} = reshape(ip(users(k), :, :, :), size(ip, 2), size(ip, 3), 3);
end
clear ip;

% user grid outline (top view background)
g = rx(1:50:end, :);
figure('Name','Ray tracing','Position',[100 100 1100 520]);
for k = 1:2
    u = users(k);
    subplot(1,2,k); hold on; grid on;
    scatter3(g(:,1), g(:,2), g(:,3), 1, [.85 .85 .85], '.');
    pw = P(u,:); valid = find(~isnan(pw));
    cmap = parula(256);
    lo = min(pw(valid)); hi = max(pw(valid));
    for l = valid
        pts = squeeze(paths{k}(l, :, :));         % interactions x 3
        if isvector(pts), pts = reshape(pts, [], 3); end
        pts = pts(all(~isnan(pts), 2) & any(pts ~= 0, 2), :);
        line_pts = [bs; pts; rx(u,:)];
        c = cmap(1 + round(255 * (pw(l) - lo) / max(hi - lo, eps)), :);
        lw = 0.5 + 2.5 * (pw(l) - lo) / max(hi - lo, eps);
        plot3(line_pts(:,1), line_pts(:,2), line_pts(:,3), '-', 'Color', c, 'LineWidth', lw);
        if itr(u,l) == 0
            plot3(line_pts(:,1), line_pts(:,2), line_pts(:,3), 'r--', 'LineWidth', 1.5);
        end
    end
    plot3(bs(1), bs(2), bs(3), 'r^', 'MarkerFaceColor', 'r', 'MarkerSize', 10);
    plot3(rx(u,1), rx(u,2), rx(u,3), 'ko', 'MarkerFaceColor', 'k');
    title(sprintf('User %.0f m from BS: %d paths (red dashed = LoS)', d(u), numel(valid)));
    xlabel('x (m)'); ylabel('y (m)'); zlabel('z (m)');
    colormap(parula); cb = colorbar; cb.Label.String = 'relative path power';
    view(35, 30); axis equal;
end
exportgraphics(gcf, fullfile(root, 'figs', 'rays.png'), 'Resolution', 200);
