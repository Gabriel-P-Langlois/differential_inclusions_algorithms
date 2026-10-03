function make_pde_oc_tables(example, mat_file)
%MAKE_PDE_OC_TABLES  Write the LaTeX table bodies of a PDE-OC benchmark run.
%
%   make_pde_oc_tables(example) reads the newest .mat of that example in
%   results/pde_oc_runs/ and writes, for every (beta, bounds) pair,
%
%       results/tables/pde_oc_<example>_pair<i>_time.tex
%       results/tables/pde_oc_<example>_pair<i>_acc.tex
%
%   make_pde_oc_tables(example, mat_file) reads that file instead.
%
%   INPUT
%       example    -   '3D' or '2D'
%       mat_file   -   optional path of the .mat to read
%
%   The files hold table rows only, ampersand-separated and ending in \\,
%   to be pasted into the tables of the paper.  A solver that did not run on a mesh
%   prints ---.
%
%   Time table:      one row per mesh, "t (r)" per solver, where r is the
%                    time divided by the best time on that mesh (the
%                    shifted geometric mean of one instance).
%   Accuracy table:  one row per mesh and solver, with the error against the
%                    reference, the primal gap, and the RELATIVE residuals
%                    r_p, r_d, r_g.  The data carry a factor beta*h^d, so the
%                    absolute residuals compare neither across meshes nor
%                    across solvers.

    if nargin < 1, example = '3D'; end

    script_dir = fileparts(mfilename('fullpath'));
    run_dir    = fullfile(script_dir, 'pde_oc_runs');
    out_dir    = fullfile(script_dir, 'tables');
    if ~isfolder(out_dir), mkdir(out_dir); end

    if nargin < 2 || isempty(mat_file)
        files = dir(fullfile(run_dir, sprintf('pde_oc_%s_*.mat', example)));
        if isempty(files)
            error('make_pde_oc_tables: no .mat of example %s in %s.', example, run_dir);
        end
        [~, newest] = max([files.datenum]);
        mat_file    = fullfile(run_dir, files(newest).name);
    end

    S = load(mat_file);
    fprintf('make_pde_oc_tables: %s\n', mat_file)
    [~, nm, ext] = fileparts(mat_file);
    src_name = [nm ext];   % recorded in each table without the local path

    names = {'Algorithm~\ref{alg:quad_prog_2}', 'MOSEK', 'Gurobi', 'HiGHS'};
    pfx   = {'nsred', 'mosek', 'gurobi', 'highs'};

    for ip = 1 : numel(S.res_all)
        res  = S.res_all{ip};
        beta = S.params.pairs(ip, 1);
        u_a  = S.params.pairs(ip, 2);
        u_b  = S.params.pairs(ip, 3);
        nn   = numel(res.n);

        %% Times, with the ratio to the best solver on each mesh
        f = fopen(fullfile(out_dir, ...
            sprintf('pde_oc_%s_pair%d_time.tex', example, ip)), 'w');
        fprintf(f, '%% %s, beta = %g, %g <= u <= %g; from %s\n', ...
                example, beta, u_a, u_b, src_name);
        for idx = 1 : nn
            t = cellfun(@(p) res.(['t_' p])(idx), pfx);
            best = min(t(~isnan(t)));
            fprintf(f, '%s', mesh_label(example, S.params, res, idx));
            for j = 1 : numel(pfx)
                if isnan(t(j))
                    fprintf(f, ' & ---');
                else
                    fprintf(f, ' & %s (%.1f)', fmt_time(t(j)), t(j) / best);
                end
            end
            fprintf(f, ' \\\\\n');
        end
        fclose(f);

        %% Accuracy
        f = fopen(fullfile(out_dir, ...
            sprintf('pde_oc_%s_pair%d_acc.tex', example, ip)), 'w');
        fprintf(f, '%% %s, beta = %g, %g <= u <= %g; from %s\n', ...
                example, beta, u_a, u_b, src_name);
        for idx = 1 : nn
            for j = 1 : numel(pfx)
                p = pfx{j};
                if j == 1
                    fprintf(f, '%s & %s', mesh_label(example, S.params, res, idx), names{j});
                else
                    % Blank cells under the mesh label (h, N) in both examples.
                    blanks = ' & ';
                    fprintf(f, '%s & %s', blanks, names{j});
                end
                if isnan(res.(['t_' p])(idx))
                    fprintf(f, ' & --- & --- & --- & --- & --- \\\\\n');
                else
                    fprintf(f, ' & %s & %s & %s & %s & %s \\\\\n', ...
                        fmt_e(res.(['err_u_' p])(idx)), ...
                        fmt_e(res.(['fgap_ref_' p])(idx)), ...
                        fmt_e(res.(['r_p_rel_' p])(idx)), ...
                        fmt_e(res.(['r_d_rel_' p])(idx)), ...
                        fmt_e(res.(['r_g_rel_' p])(idx)));
                end
            end
        end
        fclose(f);

        %% Combined time and accuracy, in the layout of the tables of the paper
        f = fopen(fullfile(out_dir, ...
            sprintf('pde_oc_%s_pair%d_paper.tex', example, ip)), 'w');
        fprintf(f, '%% %s, beta = %g, %g <= u <= %g; from %s\n', ...
                example, beta, u_a, u_b, src_name);
        for idx = 1 : nn
            first = true;
            for j = 1 : numel(pfx)
                p = pfx{j};
                if isnan(res.(['t_' p])(idx))
                    % A solver that did not run on this mesh has no row.
                    continue
                end
                if first
                    fprintf(f, '$2^{-%d}$ & %s', mesh_exponent(example, S.params, res, idx), names{j});
                    first = false;
                else
                    fprintf(f, ' & %s', names{j});
                end
                fprintf(f, ' & %s & %s & %s & %s & %s \\\\\n', ...
                    fmt_time_sci(res.(['t_' p])(idx)), ...
                    fmt_e3(res.(['fgap_ref_' p])(idx)), ...
                    fmt_e3(res.(['r_p_rel_' p])(idx)), ...
                    fmt_e3(res.(['r_d_rel_' p])(idx)), ...
                    fmt_e3(res.(['r_g_rel_' p])(idx)));
            end
        end
        fclose(f);
    end

    fprintf('make_pde_oc_tables: wrote %d table(s) to %s\n', 3*numel(S.res_all), out_dir)
end


function s = mesh_label(example, params, res, idx)
%MESH_LABEL  First column of a row: the mesh and its size.
    s = sprintf('$2^{-%d}$ & %s', mesh_exponent(example, params, res, idx), ...
                fmt_int(res.N(idx)));
end

function k = mesh_exponent(example, params, res, idx)
%MESH_EXPONENT  The k of the mesh h = 2^{-k}.
    if strcmp(example, '3D')
        k = round(log2(res.n(idx) + 1));
    else
        % Nt is the same on every mesh, so it belongs in the caption, not in
        % a column.
        k = round(-log2(params.h_vals(idx)));
    end
end

function s = fmt_int(v)
%FMT_INT  An integer in math mode, with a thin space every three digits.
    d = sprintf('%d', round(v));
    s = d(1 : 1 + mod(numel(d) - 1, 3));
    for i = numel(s) + 1 : 3 : numel(d)
        s = [s '\,' d(i : i+2)];   %#ok<AGROW>
    end
    s = ['$' s '$'];
end

function s = fmt_time(t)
%FMT_TIME  Seconds, with three significant digits.
    if t >= 100,     s = sprintf('%.0f', t);
    elseif t >= 10,  s = sprintf('%.1f', t);
    elseif t >= 1,   s = sprintf('%.2f', t);
    else,            s = sprintf('%.3f', t);
    end
end

function s = fmt_e(v)
%FMT_E  One digit and an exponent, as $1.2\cdot10^{-8}$; --- for NaN.
    if isnan(v)
        s = '---';
        return
    end
    if v == 0
        s = '$0$';
        return
    end
    e = floor(log10(abs(v)));
    m = v / 10^e;
    if abs(m) >= 9.95        % rounding one decimal carries into the exponent
        m = m / 10;
        e = e + 1;
    end
    s = sprintf('$%.1f\\cdot10^{%d}$', m, e);
end

function s = fmt_time_sci(t)
%FMT_TIME_SCI  Seconds in scientific form, with three significant digits.
    s = [fmt_e3(t) '\,s'];
end

function s = fmt_e3(v)
%FMT_E3  Two digits and an exponent, as $1.23\cdot10^{-8}$; --- for NaN.
    if isnan(v)
        s = '---';
        return
    end
    if v == 0
        s = '$0$';
        return
    end
    e = floor(log10(abs(v)));
    m = v / 10^e;
    if abs(m) >= 9.995       % rounding two decimals carries into the exponent
        m = m / 10;
        e = e + 1;
    end
    s = sprintf('$%.2f\\cdot10^{%d}$', m, e);
end
