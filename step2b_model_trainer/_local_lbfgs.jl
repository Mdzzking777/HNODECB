using LinearAlgebra
using ComponentArrays
using Zygote

function _component_loss_and_grad(loss_fn, θ)
  template = deepcopy(θ)
  f0, back = Zygote.pullback(loss_fn, θ)
  if !isfinite(f0)
    g = fill(NaN, length(Vector(θ)))
    return f0, _rebuild_component(g, template)
  end

  gθ = back(one(f0))[1]
  if gθ === nothing
    return f0, _rebuild_component(zeros(length(Vector(θ))), template)
  elseif gθ isa ComponentVector
    return f0, gθ
  elseif gθ isa AbstractVector
    return f0, _rebuild_component(vec(gθ), template)
  else
    return f0, ComponentVector(gθ)
  end
end

function _rebuild_component(vec::AbstractVector, template)
  return ComponentVector(vec, getaxes(template))
end

function _allfinite(vec::AbstractVector)
  return all(isfinite, vec)
end

function _lbfgs_field_or(x, key::Symbol, default)
  if x isa NamedTuple
    return hasproperty(x, key) ? getproperty(x, key) : default
  elseif x isa AbstractDict
    return haskey(x, key) ? x[key] : default
  else
    return hasproperty(x, key) ? getproperty(x, key) : default
  end
end

_lbfgs_copy_hist(hist) = [copy(Vector{Float64}(v)) for v in hist]

function _lbfgs_direction(g::Vector{Float64}, s_hist, y_hist, rho_hist)
  isempty(s_hist) && return -copy(g)

  q = copy(g)
  alpha = zeros(Float64, length(s_hist))
  for i in length(s_hist):-1:1
    alpha[i] = rho_hist[i] * dot(s_hist[i], q)
    q .-= alpha[i] .* y_hist[i]
  end

  sty = dot(s_hist[end], y_hist[end])
  yty = dot(y_hist[end], y_hist[end])
  gamma = (isfinite(sty) && isfinite(yty) && yty > 0.0) ? (sty / yty) : 1.0
  r = gamma .* q

  for i in 1:length(s_hist)
    beta = rho_hist[i] * dot(y_hist[i], r)
    r .+= s_hist[i] .* (alpha[i] - beta)
  end
  return -r
end

function run_local_lbfgs(loss_fn, θ0;
  callback::Function,
  maxiters::Int,
  history_size::Int=10,
  allow_f_increases::Bool=true,
  backtrack::Float64=0.5,
  min_alpha::Float64=1e-8,
  armijo_c1::Float64=1e-4,
  resume_state=nothing,
  checkpoint_every::Int=0,
  checkpoint_fn=nothing)

  template = deepcopy(θ0)
  x = copy(Vector(θ0))
  θ = _rebuild_component(x, template)
  f, gθ = _component_loss_and_grad(loss_fn, θ)
  g = copy(Vector(gθ))

  s_hist = Vector{Vector{Float64}}()
  y_hist = Vector{Vector{Float64}}()
  rho_hist = Float64[]
  start_iter = 1

  if resume_state !== nothing
    saved_x = _lbfgs_field_or(resume_state, :x, nothing)
    if saved_x !== nothing && length(saved_x) == length(x)
      x = copy(Vector{Float64}(saved_x))
      θ = _rebuild_component(x, template)
      f_saved = _lbfgs_field_or(resume_state, :f, NaN)
      g_saved = _lbfgs_field_or(resume_state, :g, nothing)
      if g_saved !== nothing && length(g_saved) == length(g) && isfinite(f_saved) && _allfinite(g_saved)
        f = Float64(f_saved)
        g = copy(Vector{Float64}(g_saved))
      else
        f, gθ = _component_loss_and_grad(loss_fn, θ)
        g = copy(Vector(gθ))
      end
      s_hist = _lbfgs_copy_hist(_lbfgs_field_or(resume_state, :s_hist, Vector{Vector{Float64}}()))
      y_hist = _lbfgs_copy_hist(_lbfgs_field_or(resume_state, :y_hist, Vector{Vector{Float64}}()))
      rho_hist = copy(Vector{Float64}(_lbfgs_field_or(resume_state, :rho_hist, Float64[])))
      hist_len = min(length(s_hist), length(y_hist), length(rho_hist))
      if hist_len != length(s_hist) || hist_len != length(y_hist) || hist_len != length(rho_hist)
        resize!(s_hist, hist_len)
        resize!(y_hist, hist_len)
        resize!(rho_hist, hist_len)
      end
      start_iter = max(1, Int(_lbfgs_field_or(resume_state, :epoch, 0)) + 1)
    end
  end

  for iter in start_iter:maxiters
    direction = _lbfgs_direction(g, s_hist, y_hist, rho_hist)
    if !_allfinite(direction) || dot(direction, g) >= 0.0
      direction = -copy(g)
    end

    alpha = 1.0
    descent = dot(g, direction)
    accepted = false
    x_trial = copy(x)
    θ_trial = θ
    f_trial = f

    while alpha >= min_alpha
      x_trial = x .+ alpha .* direction
      θ_trial = _rebuild_component(x_trial, template)
      f_trial = loss_fn(θ_trial)
      if !isfinite(f_trial)
        alpha *= backtrack
        continue
      end
      if allow_f_increases || f_trial <= f + armijo_c1 * alpha * descent
        accepted = true
        break
      end
      alpha *= backtrack
    end

    accepted || break

    f_new, gθ_new = _component_loss_and_grad(loss_fn, θ_trial)
    g_new = copy(Vector(gθ_new))
    if !isfinite(f_new) || !_allfinite(g_new)
      break
    end

    s = x_trial .- x
    y = g_new .- g
    sty = dot(s, y)
    if isfinite(sty) && sty > 1e-12
      push!(s_hist, s)
      push!(y_hist, y)
      push!(rho_hist, 1.0 / sty)
      if length(s_hist) > history_size
        popfirst!(s_hist)
        popfirst!(y_hist)
        popfirst!(rho_hist)
      end
    end

    old_f = f
    x = x_trial
    θ = θ_trial
    f = f_new
    g = g_new

    step_norm = norm(s)
    dloss = f_new - old_f
    curv = dot(s, s) <= 0.0 || !isfinite(sty) ? NaN : sty / dot(s, s)
    g_dot_step = dot(g_new, s)
    stats = (grad=g_new, step=s, step_norm=step_norm, dloss=dloss, sTy=sty, curv=curv, g_dot_step=g_dot_step, iter=iter, alpha=alpha)

    should_stop = applicable(callback, θ, f, stats) ? callback(θ, f, stats) : callback(θ, f)
    if checkpoint_every > 0 && checkpoint_fn !== nothing &&
       ((iter % checkpoint_every) == 0 || iter == maxiters || should_stop)
      checkpoint_fn((
        epoch=iter,
        x=copy(x),
        f=f,
        g=copy(g),
        s_hist=_lbfgs_copy_hist(s_hist),
        y_hist=_lbfgs_copy_hist(y_hist),
        rho_hist=copy(rho_hist)
      ))
    end
    should_stop && break
  end

  return θ
end
