function make_pde_oc_figures(example, mat_file)
%MAKE_PDE_OC_FIGURES  Time against problem size for a PDE-OC benchmark run.
%
%   make_pde_oc_figures(example) reads the newest .mat of that example in
%   results/pde_oc_runs/ and writes, for every (beta, bounds) pair,
%
%       results/figures/pde_oc_<example>_pair<i>_time.pdf
%
%   make_pde_oc_figures(example, mat_file) reads that file instead.
%
%   INPUT
%       example    -   '3D' or '2D'
%       mat_file   -   optional path of the .mat to read
%
%   One log-log curve per solver, against N.  A solver that did not run on a
%   mesh leaves a gap.  The dashed line has slope one through the first
%   point of Algorithm 2.

    if nargin < 1, example = '3D'; end

    script_dir = fileparts(mfilename('fullpath'));
    run_dir    = fullfile(script_dir, 'pde_oc_runs');
    out_dir    = fullfile(script_dir, 'figures');
    if ~isfolder(out_dir), mkdir(out_dir); end

    if nargin < 2 || isempty(mat_file)
        files = dir(fullfile(run_dir, sprintf('pde_oc_%s_*.mat', example)));
        if isempty(files)
            error('make_pde_oc_figures: no .mat of example %s in %s.', example, run_dir);
        end
        [~, newest] = max([files.datenum]);
        mat_file    = fullfile(run_dir, files(newest).name);
    end

    S = load(mat_file);
    fprintf('make_pde_oc_figures: %s\n', mat_file)

    names   = {'Algorithm 2', 'MOSEK', 'Gurobi', 'HiGHS'};
    pfx     = {'nsred', 'mosek', 'gurobi', 'highs'};
    markers = {'o', 's', 'd', '^'};

    for ip = 1 : numel(S.res_all)
        res  = S.res_all{ip};

        % Drawn at the size it has in the paper, a subfigure of width
        % 0.48\linewidth on a 6.5 in text block, so that the fonts keep their
        % point sizes.
        fig = figure('Visible', 'off', 'Units', 'inches', ...
                     'Position', [0, 0, 3.1, 2.5]);
        ax  = axes(fig, 'FontSize', 8);
        hold(ax, 'on')
        leg = {};
        for j = 1 : numel(pfx)
            t  = res.(['t_' pfx{j}]);
            ok = ~isnan(t);
            if ~any(ok), continue, end
            loglog(ax, res.N(ok), t(ok), ['-' markers{j}], 'LineWidth', 1, ...
                   'MarkerSize', 4)
            leg{end+1} = names{j};   %#ok<AGROW>
        end

        % Reference slope one through the first point of Algorithm 2.
        t1 = res.t_nsred;
        ok = find(~isnan(t1), 1);
        if ~isempty(ok)
            Nl = res.N(~isnan(t1));
            loglog(ax, Nl, t1(ok) * Nl / res.N(ok), 'k--', 'LineWidth', 0.8)
            leg{end+1} = 'slope 1';   %#ok<AGROW>
        end

        set(ax, 'XScale', 'log', 'YScale', 'log')
        % Room above the largest time and below the smallest, so that their
        % markers clear the edges of the axes.
        t_max = max(cellfun(@(p) max(res.(['t_' p])), pfx));
        t_min = min(cellfun(@(p) min(res.(['t_' p])), pfx));
        ylim(ax, [10^floor(log10(t_min) - 0.3), 10^ceil(log10(t_max) + 0.3)])
        grid(ax, 'on')
        xlabel(ax, 'N')
        ylabel(ax, 'time (s)')
        legend(ax, leg, 'Location', 'northwest', 'FontSize', 7)
        hold(ax, 'off')

        out_file = fullfile(out_dir, ...
            sprintf('pde_oc_%s_pair%d_time.pdf', example, ip));
        exportgraphics(fig, out_file, 'ContentType', 'vector')
        close(fig)
        fprintf('  wrote %s\n', out_file)
    end
end
