"""
    ground_truth_function(du, u, p, t)

Derivative function for the AFM DMT-KV model.
State u = [x1, x2, x3] = [tip displacement, tip velocity, sample motion].
Parameter order: k, wd, m, c, Fd, R, dist, Fad, Estar, ks, cs.
"""
function softplus(x, eps)
    z = x / eps
    if z > 50.0
        return x
    elseif z < -50.0
        return 0.0
    end
    return eps * log1p(exp(z))
end
function softplus(x)
    if x > 50.0
        return x
    elseif x < -50.0
        return exp(x)
    end
    return log1p(exp(x))
end
contact_weight(s, eps) = 1.0 - exp(-softplus(-s, eps) / eps)

function make_contact_weight_lookup(times, s_values)
    tvals = collect(Float64, times)
    svals = collect(Float64, s_values)
    n = length(tvals)
    n == length(svals) || error("times and s_values must have the same length.")
    n > 0 || error("times and s_values must be non-empty.")

    function weight_at_time(t)
        if n == 1
            s_t = svals[1]
        elseif t <= tvals[1]
            s_t = svals[1]
        elseif t >= tvals[end]
            s_t = svals[end]
        else
            idx_hi = searchsortedfirst(tvals, t)
            if idx_hi <= 1
                s_t = svals[1]
            elseif tvals[idx_hi] == t
                s_t = svals[idx_hi]
            else
                idx_lo = idx_hi - 1
                t_lo = tvals[idx_lo]
                t_hi = tvals[idx_hi]
                alpha = (t - t_lo) / (t_hi - t_lo)
                s_t = svals[idx_lo] + alpha * (svals[idx_hi] - svals[idx_lo])
            end
        end
        return contact_weight(s_t, adhesion_transition)
    end

    return weight_at_time
end

function ground_truth_function(du, u, p, t)
    k, wd, m, c, Fd, R, dist, Fad, Estar, ks, cs = p

    # Separation
    s = dist + u[1] - u[3]
    delta = softplus(-s, adhesion_transition)
    delta = ifelse(delta > 0.0, delta, 0.0)
    w = contact_weight(s, adhesion_transition)

    F_hertz = (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5)
    Fad_eff = Fad * w

    # Tip kinematics
    du[1] = u[2]
    du[2] = (Fd * cos(wd * t) - k * u[1] - c * u[2] - Fad_eff + F_hertz) / m
    du[3] = (Fad_eff - F_hertz - ks * u[3]) / cs
end

"""
    get_uode_model_function(appr_neural_network, state, original_parameters_opt)

Returns the model derivative function with a neural-network residual added to x2dot.
The mechanistic parameters are scaled by the original parameters.
"""
function get_uode_model_function(appr_neural_network, state, original_parameters_opt)
    f(du, u, p, t) =
        let appr_neural_network = appr_neural_network, st = state, original_parameters_opt = original_parameters_opt

            ode_par = p.ode_par .* original_parameters_opt
            k, wd, m, c, Fd, R, dist, Fad, Estar, ks, cs = ode_par

            # Separation
            s = dist + u[1] - u[3]
            delta = softplus(-s, adhesion_transition)
            delta = ifelse(delta > 0.0, delta, 0.0)
            w = contact_weight(s, adhesion_transition)

            F_hertz = (4.0 / 3.0) * Estar * sqrt(R) * (delta^1.5)
            Fad_eff = Fad * w

            # Tip kinematics
            @inbounds du[1] = u[2]
            @inbounds du[2] = (Fd * cos(wd * t) - k * u[1] - c * u[2] - Fad_eff + F_hertz) / m
            @inbounds du[3] = (Fad_eff - F_hertz - ks * u[3]) / cs

            # NN residual on x2dot (assumes a scalar output)
            û = appr_neural_network(u, p.p_net, st)[1]
            @inbounds du[2] += û[1]
        end
end

"""
    get_uode_model_function_hertz_nn(appr_neural_network, state)

Returns the model derivative function with the net contact-force term
`F_hertz - Fad_eff` replaced by a neural network.
The network input is `[x1, x2, x3]` and the output is gated by a supplied
true contact-weight lookup when available, otherwise by the model-predicted
soft contact gate.
"""
function get_uode_model_function_hertz_nn(appr_neural_network, state, true_contact_weight_at_time=nothing)
    f(du, u, p, t) =
        let appr_neural_network = appr_neural_network,
            st = state,
            true_contact_weight_at_time = true_contact_weight_at_time

            ode_par = p.ode_par
            k, wd, m, c, Fd, R, dist, Fad, Estar, ks, cs = ode_par

            # Separation
            s = dist + u[1] - u[3]
            delta = softplus(-s, adhesion_transition)
            delta = ifelse(delta > 0.0, delta, 0.0)
            w_pred = contact_weight(s, adhesion_transition)

            # NN-based net-contact replacement (scalar output)
            nn_in = collect(promote(u[1], u[2], u[3]))
            û = appr_neural_network(nn_in, p.p_net, st)[1]
            w_gate = true_contact_weight_at_time === nothing ? w_pred : true_contact_weight_at_time(t)
            Fad_eff = Fad * w_pred
            F_contact = û[1] * w_gate

            # Tip kinematics
            @inbounds du[1] = u[2]
            @inbounds du[2] = (Fd * cos(wd * t) - k * u[1] - c * u[2] + F_contact) / m
            @inbounds du[3] = (-F_contact - ks * u[3]) / cs
        end
end
