f_star_std = sqrt(f_star_variance);

hold('off');

fill([x_star; flipud(x_star)], ...
     [       f_star_mean + 2 * f_star_std; ...
      flipud(f_star_mean - 2 * f_star_std)], ...
     [0.9, 0.9, 1], ...
     'edgecolor', 'none');

hold('on');

plot(x, y, 'k.');
plot(x_star, y_star, 'r.');
plot(x_star, f_star_mean, '-', ...
     'color', [0, 0, 0.8]);

axis([-30, 30, -5, 5]);
set(gca, 'tickdir', 'out', ...
         'box',     'off');
