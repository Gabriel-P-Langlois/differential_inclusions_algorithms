function make_pos_tables()
%MAKE_POS_TABLES  Write the LaTeX table bodies of the positivity benchmarks.
%
%   Reads what the two positivity drivers save and writes
%
%       results/tables/pos_dg.tex          from results/pos_PDE_runs/*.mat
%       results/tables/pos_synth_time.tex  from results/pos_synthetic_runs/*.mat
%       results/tables/pos_synth_acc.tex   from the same file
%
%   The files hold table rows only, ampersand-separated and ending in \\,
%   to be pasted into the tables of the paper.
%
%   The DG table (Section 6.1.2) has one row per (k, n) and solver, with the
%   total limiter iterations, the stage-1 time (the per-step means, summed)
%   with its ratio to the best stage-1 time on the same (k, n), the wall
%   time with one stage-1 call per step (R.wall_one), the mass drift and the
%   errors at the final time.  The synthetic tables (Section 6.1.1) have one
%   row per solver, with one column per Delta x.  The solvers are
%   Algorithm 1 (di_pos_w) and DR.

    script_dir = fileparts(mfilename('fullpath'));
    out_dir    = fullfile(script_dir, 'tables');
    if ~isfolder(out_dir), mkdir(out_dir); end

    dg_table(script_dir, out_dir);
    synth_tables(script_dir, out_dir);
end


function dg_table(script_dir, out_dir)
%DG_TABLE  One row per march saved by benchmark_pos_PDE.m.
    run_dir = fullfile(script_dir, 'pos_PDE_runs');
    files   = dir(fullfile(run_dir, 'k*_n*_*.mat'));
    if isempty(files)
        fprintf('make_pos_tables: no march in %s, skipping the DG table\n', run_dir);
        return
    end

    names = struct('dipos', 'Algorithm~\ref{alg:quad_prog}', 'dr', 'DR');

    Rs  = cell(numel(files), 1);
    tl  = zeros(numel(files), 1);
    key = zeros(numel(files), 3);
    for i = 1 : numel(files)
        S = load(fullfile(run_dir, files(i).name));
        Rs{i} = S.R;
        tl(i) = sum(Rs{i}.tlim);
        key(i, :) = [Rs{i}.k, Rs{i}.n, arm_order(Rs{i}.arm)];
    end

    rows = cell(numel(files), 1);
    for i = 1 : numel(files)
        R = Rs{i};
        if isfield(names, R.arm), solver = names.(R.arm); else, solver = R.arm; end
        best = min(tl(key(:, 1) == R.k & key(:, 2) == R.n));
        wall = R.wall;
        if isfield(R, 'wall_one'), wall = R.wall_one; end
        rows{i} = sprintf('%d & %d & %s & %s & %d & %s (%.1f) & %s & %s & %s & %s \\\\\n', ...
            R.k, R.n, fmt_g(R.tau), solver, sum(R.work), ...
            fmt_time(tl(i)), tl(i) / best, fmt_time(wall), fmt_e(R.drift), ...
            fmt_e(R.eL2), fmt_e(R.eLinf));
    end

    [~, order] = sortrows(key);
    f = fopen(fullfile(out_dir, 'pos_dg.tex'), 'w');
    fprintf(f, '%% DG marches from pos_PDE_runs/\n');
    for i = order(:)'
        fprintf(f, '%s', rows{i});
    end
    fclose(f);
    fprintf('make_pos_tables: wrote %s (%d rows)\n', ...
            fullfile(out_dir, 'pos_dg.tex'), numel(files));
end


function synth_tables(script_dir, out_dir)
%SYNTH_TABLES  Time and accuracy of the synthetic benchmark, one column per dx.
    run_dir = fullfile(script_dir, 'pos_synthetic_runs');
    files   = dir(fullfile(run_dir, '*.mat'));
    if isempty(files)
        fprintf(['make_pos_tables: benchmark_pos_synthetic.m has saved nothing yet, ' ...
                 'skipping its tables\n']);
        return
    end
    [~, newest] = max([files.datenum]);
    S = load(fullfile(run_dir, files(newest).name));
    res = S.res;

    arms  = {'dipos', 'dr'};
    names = {'Algorithm~\ref{alg:quad_prog}', 'DR'};

    f = fopen(fullfile(out_dir, 'pos_synth_time.tex'), 'w');
    fprintf(f, '%% synthetic benchmark from %s\n', files(newest).name);
    T = cell2mat(cellfun(@(a) res.(['time_' a]), arms, 'UniformOutput', false));
    for j = 1 : numel(arms)
        fprintf(f, '%s', names{j});
        for idx = 1 : numel(res.dx)
            t = T(idx, j);
            col = T(idx, :);
            if isnan(t)
                fprintf(f, ' & ---');
            else
                fprintf(f, ' & %s (%.1f)', fmt_time(t), t / min(col(~isnan(col))));
            end
        end
        fprintf(f, ' \\\\\n');
    end
    fclose(f);

    f = fopen(fullfile(out_dir, 'pos_synth_acc.tex'), 'w');
    fprintf(f, '%% synthetic benchmark from %s\n', files(newest).name);
    labels = {'error', 'conservation'};
    measures = {'err_', 'cons_'};
    for k = 1 : numel(measures)
        m = measures{k};
        for j = 1 : numel(arms)
            if j == 1, lab = labels{k}; else, lab = ''; end
            fprintf(f, '%s & %s', lab, names{j});
            v = res.([m arms{j}]);
            for idx = 1 : numel(res.dx)
                fprintf(f, ' & %s', fmt_e(v(idx)));
            end
            fprintf(f, ' \\\\\n');
        end
    end
    fclose(f);
    fprintf('make_pos_tables: wrote the synthetic tables to %s\n', out_dir);
end


function o = arm_order(arm)
%ARM_ORDER  Sort key that puts Algorithm 1 first.
    switch arm
        case 'dipos',  o = 1;
        case 'dr',     o = 2;
        otherwise,     o = 3;
    end
end

function s = fmt_g(v)
%FMT_G  Compact general format for a parameter such as tau.
    s = sprintf('%g', v);
end

function s = fmt_time(t)
%FMT_TIME  Seconds, three significant digits and an exponent, as
%   $1.39\cdot10^{-3}$; --- for NaN.
    if isnan(t), s = '---';  return, end
    if t == 0,   s = '$0$';  return, end
    e = floor(log10(t));
    m = t / 10^e;
    if m >= 9.995            % rounding two decimals carries into the exponent
        m = m / 10;
        e = e + 1;
    end
    s = sprintf('$%.2f\\cdot10^{%d}$', m, e);
end

function s = fmt_e(v)
%FMT_E  One digit and an exponent, as $1.2\cdot10^{-8}$; --- for NaN.
    if isnan(v), s = '---';  return, end
    if v == 0,   s = '$0$';  return, end
    e = floor(log10(abs(v)));
    m = v / 10^e;
    if abs(m) >= 9.95        % rounding one decimal carries into the exponent
        m = m / 10;
        e = e + 1;
    end
    s = sprintf('$%.1f\\cdot10^{%d}$', m, e);
end
