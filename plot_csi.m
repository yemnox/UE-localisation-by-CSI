%% plot_csi.m - CSI "images" of a near user and a far user, square pixels
% Run from the "UE localisation" folder after generate_csi.m
load('csi_o1.mat', 'H', 'pos', 'bs_pos');
dist = hypot(double(pos(:,1)) - bs_pos(1), double(pos(:,2)) - bs_pos(2));
[~, near] = min(abs(dist - 20));      % user ~20 m from the BS
[~, far]  = min(abs(dist - 150));     % user ~150 m from the BS

figure('Name','CSI images','Position',[100 100 900 650]);
users = [near far];
for r = 1:2
    h = squeeze(H(users(r),:,:));
    subplot(2,2,2*r-1); imagesc(abs(h));  axis image; colorbar;
    title(sprintf('|H|, user at %.0f m', dist(users(r))));
    xlabel('subcarrier'); ylabel('antenna');
    subplot(2,2,2*r);   imagesc(real(h)); axis image; colorbar;
    title(sprintf('Re(H), user at %.0f m', dist(users(r))));
    xlabel('subcarrier'); ylabel('antenna');
end
colormap(parula);
exportgraphics(gcf, 'csi_images.png', 'Resolution', 200);
