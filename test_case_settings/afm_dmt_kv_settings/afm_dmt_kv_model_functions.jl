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
contact_weight(s, eps) = 1.0 - exp(-softplus(-s, eps) / eps)

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
    du[2] = (Fd * cos(wd * t) - k * u[1] - c * u[2] + Fad_eff - F_hertz) / m
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
            @inbounds du[2] = (Fd * cos(wd * t) - k * u[1] - c * u[2] + Fad_eff - F_hertz) / m
            @inbounds du[3] = (Fad_eff - F_hertz - ks * u[3]) / cs

            # NN residual on x2dot (assumes a scalar output)
            û = appr_neural_network(u, p.p_net, st)[1]
            @inbounds du[2] += û[1]
        end
end
