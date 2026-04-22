const DUALRANK_ROLES = ("RankX", "RankY")

function dualrank_root(repo_root::String)
    return normpath(joinpath(repo_root, "logs", "stage2_step2a", "local", "windowed_dualrank"))
end

function dualrank_role_dir(repo_root::String, role::AbstractString)
    return normpath(joinpath(dualrank_root(repo_root), String(role)))
end

function dualrank_log_dir(repo_root::String, role::AbstractString)
    return joinpath(dualrank_role_dir(repo_root, role), "window_per_shard")
end

function dualrank_visualization_dir(repo_root::String, role::AbstractString)
    return joinpath(dualrank_role_dir(repo_root, role), "Visualization")
end

function dualrank_results_dir(repo_root::String, role::AbstractString)
    role_dir = dualrank_role_dir(repo_root, role)
    upper = joinpath(role_dir, "Results")
    lower = joinpath(role_dir, "results")
    if isdir(upper) || !isdir(lower)
        return upper
    end
    return lower
end

function dualrank_info_path(repo_root::String, role::AbstractString)
    return joinpath(dualrank_role_dir(repo_root, role), "run_info.txt")
end

function read_dualrank_info(repo_root::String, role::AbstractString)
    info = Dict{String, String}()
    path = dualrank_info_path(repo_root, role)
    if !isfile(path)
        return info
    end
    for line in eachline(path)
        m = match(r"^([^=]+)=(.*)$", strip(line))
        m === nothing && continue
        info[strip(m.captures[1])] = strip(m.captures[2])
    end
    return info
end

function dualrank_actual_rank(repo_root::String, role::AbstractString)
    info = read_dualrank_info(repo_root, role)
    raw = get(info, "CandidateRank", "")
    isempty(raw) && return nothing
    val = tryparse(Int, raw)
    return val
end

function dualrank_available_roles(repo_root::String)
    roles = String[]
    for role in DUALRANK_ROLES
        role_dir = dualrank_role_dir(repo_root, role)
        if isdir(role_dir)
            push!(roles, role)
        end
    end
    return roles
end

function dualrank_requested(args)
    isempty(args) && return false
    flag = lowercase(strip(String(first(args))))
    return flag == "dualrank" || flag == "--dualrank"
end

function should_use_dualrank_defaults(repo_root::String, args)
    return dualrank_requested(args) && !isempty(dualrank_available_roles(repo_root))
end

function windowed_shard_log_series(log_dir::String, shard::Integer)
    paths = String[]
    if isdir(log_dir)
        base_paths = String[]
        resume_paths = Pair{Int, String}[]
        resume_pattern = Regex("_resume(\\d+)_p$(shard)\\.txt\$")
        base_pattern = Regex("_p$(shard)\\.txt\$")
        for path in readdir(log_dir; join=true)
            name = basename(path)
            m_resume = match(resume_pattern, name)
            if m_resume !== nothing
                idx = tryparse(Int, m_resume.captures[1])
                idx === nothing || push!(resume_paths, idx => path)
            elseif match(base_pattern, name) !== nothing
                push!(base_paths, path)
            end
        end
        sort!(base_paths)
        sort!(resume_paths; by=first)
        append!(paths, base_paths)
        append!(paths, last.(resume_paths))
    end
    if isempty(paths)
        push!(paths, joinpath(log_dir, "log2_03_step2a_stage2light_windowed_local_p$(shard).txt"))
    end
    return paths
end

function windowed_default_log_series(log_dir::String; shard_count::Int=3)
    return [windowed_shard_log_series(log_dir, shard) for shard in 1:shard_count]
end

function print_dualrank_banner(repo_root::String, role::AbstractString)
    rank = dualrank_actual_rank(repo_root, role)
    if rank === nothing
        println("Dual-rank mode: ", role)
    else
        println("Dual-rank mode: ", role, " (candidate rank ", rank, ")")
    end
end
