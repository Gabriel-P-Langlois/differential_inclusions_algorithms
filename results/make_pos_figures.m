function make_pos_figures(mat_file)
%MAKE_POS_FIGURES  Time against problem size for the synthetic limiter
%   benchmark.
%
%   make_pos_figures() reads the newest .mat in results/pos_synthetic_runs/
%   and writes
%
%       results/figures/pos_synthetic_time.pdf
%
%   make_pos_figures(mat_file) reads that file instead.
%
%   INPUT
%       mat_file   -   optional path of the .mat to read
%
%   One log-log curve per solver, Algorithm 1 (di_pos_w) and DR: the mean
%   time of one solve against the number n of points.  The dashed line has
%   slope one through the first point of Algorithm 1.

    script_dir = fileparts(mfilename('fullpath'));
    run_dir    = fullfile(script_dir, 'pos_synthetic_runs');
    out_dir    = fullfile(script_dir, 'figures');
    if ~isfolder(out_dir), mkdir(out_dir); end

    if nargin < 1 || isempty(mat_file)
        files = dir(fullfile(run_dir, 'pos_synthetic_*.mat'));
        if isempty(files)
            error('make_pos_figures: no .mat in %s.', run_dir);
        end
        [~, newest] = max([files.datenum]);
        mat_file    = fullfile(run_dir, files(newest).name);
    end

    S   = load(mat_file);
    res = S.res;
    fprintf('make_pos_figures: %s\n', mat_file)

    names   = {'Algorithm 1', 'DR'};
    arms    = {'dipos', 'dr'};
    markers = {'o', 'v'};

    % Drawn at the size it has in the paper, width 0.52\linewidth on a
    % 6.5 in text block, so that the fonts keep their point sizes.
    fig = figure('Visible', 'off', 'Units', 'inches', ...
                 'Position', [0, 0, 3.4, 2.7]);
    ax  = axes(fig, 'FontSize', 8);
    hold(ax, 'on')
    leg = {};
    for j = 1 : numel(arms)
        t  = res.(['time_' arms{j}]);
        ok = ~isnan(t);
        if ~any(ok), continue, end
        loglog(ax, res.N(ok), t(ok), ['-' markers{j}], 'LineWidth', 1, ...
               'MarkerSize', 4)
        leg{end+1} = names{j};   %#ok<AGROW>
    end

    % Reference slope one through the first point of Algorithm 1.
    t1 = res.time_dipos;
    ok = find(~isnan(t1), 1);
    if ~isempty(ok)
        Nl = res.N(~isnan(t1));
        loglog(ax, Nl, t1(ok) * Nl / res.N(ok), 'k--', 'LineWidth', 0.8)
        leg{end+1} = 'slope 1';
    end

    set(ax, 'XScale', 'log', 'YScale', 'log')
    grid(ax, 'on')
    xlabel(ax, 'n')
    ylabel(ax, 'mean time of one solve (s)')
    legend(ax, leg, 'Location', 'northwest', 'FontSize', 7)
    hold(ax, 'off')

    out_file = fullfile(out_dir, 'pos_synthetic_time.pdf');
    exportgraphics(fig, out_file, 'ContentType', 'vector')
    close(fig)
    fprintf('  wrote %s\n', out_file)
end
