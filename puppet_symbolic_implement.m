clear all;
close all;
clc;



n = 1; % Pendulum casea
n_p = 1; % number of attachment points
n_s = 1; % number of strings
m = 1.0;
M = eye(3) * m; % Mass matrix
g = [0 0 -9.81];
T = 40; % "The number of time steps was 40 in all cases"
plannningHorizon = 1.0; % [s]
h = plannningHorizon / T; % step size

s_0 = [0.5]; % lengthes of the strings
k = 1e4; % spring constant
epsilon = 0.001;
s_0 = 0.5;
stringsPairs = reshape([1 1], [2, n_s]);
% linkPairs = reshape([1 2 2 3], [2,2]);



syms _psy(x);

%_x = _sym('x%d', [3*n T]); 
%_p = _sym('p%d', [3*n_p T]);
%_x = _x(:);
%_p = _p(:);

x_i_sym = sym('x%d', [3*n 1]);
p_i_sym = sym('p%d', [3"n_p 1]);
x_im1_sym = sym('x%d_m1', [3*n 1]);
x_im2_sym = sym('x%d_m2', [3*n 1]);

% compute the time-discretized acceleration
xdot_i = (x_i_sym - x_im1_sym) / h;
xdot2_i = (x_i - 2 * x_im1_sym + x_im2_sym) / (h^2);

% internal potential deformation energy (strings and trusses)
W = compute_potential_energy(x_i_sym, p_i_sym);

% calculate x_i with implicit Euler time stepping scheme
U_i = (h^2 / 2) * xdot2_i.' * M * xdot2_i + W + g.' * M * x_i_sym;

% compute gradient and hessian
gradU = jacobian(U_i, x_i_sym);
hessU = jacobian(gradU, x_i_sym);


% Initialize
x_0 = 

% forward simulation x(p)
function x = forward_sim(p, x_0, xdot_0)
    x_i = x_0;
    x_im1 = x_0;
    xdot_i = xdot_0;
    xdot_im1 = xdot_0;

    x = x_0;
    for i = 1:T
        p_i = p(i:i+3*n_p-1);
        grad_i = subs(gradU, p_i_sym, p_i);
        hess_i = subs(hessU, p_i_sym, p_i);

        % compute x_i with Newton's method
        x_opt = NewtonsMethod(grad_i, hess_i, x_i);

        x_im1 = x_i;
        x_i = x_opt;
        xdot_im1 = xdot_i;
        xdot_i = (x_i - x_im1) / h;
        x = [x; x_i]
    end
end

function x_opt = NewtonsMethod(grad_i, hess_i, x_init,err=1e-8,max_iteration=50)
    % for regularization
    r = 1e-8;
    I = eye(size(x_init,1));

    x_k = x_init;

    cnt = 0;
    while(1)
        cnt = cnt + 1;
        grad_k = subs(grad_i, x_i_sym, x_k);
        hess_k = subs(hessU, x_i_sym, x_k);
        dx = -inv(hess_i + r*I) * grad_i;
        x_k = x_k + dx;
        if norm(dx) < err
            x_opt = x_k;
            break
        elseif i == max_iteration
            x_opt = x_k;
            break
        end
    end
end


function dO_dp = compute_dOdp(x, p)
end

% compute Hessian
function H = compute_H(dO_dp, x, p)
end

% compute update direction
function d = compute_d(H, dO_dp)
end

% backtracking line search
function alfa = backtracking(p, d)
    p = p + alfa * d;
end

function p_new = update_p(alfa, d)
end

% model of string



function W = compute_potential_energy(x_i, p_i)
    % W_puppet: the energy stored in the stiff springs (puppet linkages)
    if n > 1
        % TODO: compute W_puppet
    elseif n == 1
        W_puppet = 0;
    else
        disp("ERROR: n is invalid value")
    end

    % W_string: the energy stored in the strings
    % psy: model of a string
    _psy(x) = piecewise(x>0, (1/2)*x^2 + (epsilon/2)*x + epsilon^2 / 6,...
                0>x>-epsilon, (1/(6*epsilon))*x^3 + (1/2)*x^2 + (epsilon/2)*x + epsilon^2/6, ...
                0);

    W_string = 0;
    for i = 1:n_s
        W_string = W_string + k * psy(norm(x_i - p_i) - s_0(i));
    end

    W = W_puppet + W_string
end