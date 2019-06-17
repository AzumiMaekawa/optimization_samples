% this is a example of quadcopter trajectory generation
% based on the paper "Optimization-based iterative learning
% for precise quadrocopter trajectory tracking"

close all;
clear;
clc;

% parameters setting
l = 0.17; % m
m = 0.468; % kg
I_xx = 0.0023; % kg m^2
f_max = 4.5; % m/s^2
f_min = 0.4; % m/s^2
f_dot_max = 27; % m/s^3
phi_dot_max = 22; % rad/s
phi_dot2_max = 150; % rad/s^2
g = 9.8; % m/s^2

N_p = 30; % number of points to specify the yz-plane path
N_lambda = 30 % support points to specify the lambda

p = zeros(N_p);



function sigma_t = lambda(t, t_f, Sigma)
    t_array = zeros(N_lambda);
    for k = 1:N_lambda
        t_array(k) = (k-1) / (N_lambda - 1) * t_f;
    sigma_array = [0 Sigma 1];
    f = fit(t_array, sigma_array, 'poly9');
    sigma_t = f(t);
    end
end


% objective function
% in this time we minimize the end time of the trajectory
function J = evalfunc(x)
    J = x(end); % x(end) = t_f
end

function c = inequality(x)
end

function ceq = equality(x)
    % collocation constraints
    % trapezoid method: (1/2)h(f_(k+1) + f_k) = x_(k+1) - x_k
    t_f = x(end);
    h = t_f / (N_p - 1);

    % state
    y = x(1:N_p);
    y_dot = x(N_p+1:N_p*2);
    z = x(N_p*2+1:N_p*3);
    z_dot = x(N_p*3+1:N_p*4);
    phi = x(N_p*4+1:N_p*5);

    % control
    f_coll = x(N_p*5+1:N_p*6);
    omega_x = x(N_p*6+1:N_p*7);

    y_dot2 = -f_coll .* sin(phi);
    z_dot2 = f_coll .* cos(phi) - g;
    phi_dot = omega_x;

    col_cons = zeros(5); % y, y_dot, z, z_dot, phi
    col_cons(1) = (1/2.) * h * (y_dot(2:N_p) + y_dot(1:N_p-1)) - (y(2:N_p) - y(1:N_p-1));
    col_cons(2) = (1/2.) * h * (y_dot2(2:N_p) + y_dot2(1:N_p-1)) - (y_dot(2:N_p) - y_dot(1:N_p-1));
    col_cons(3) = (1/2.) * h * (z_dot(2:N_p) + z_dot(1:N_p-1)) - (z(2:N_p) - z(1:N_p-1));
    col_cons(4) = (1/2.) * h * (z_dot2(2:N_p) + z_dot2(1:N_p-1)) - (z_dot(2:N_p) - z_dot(1:N_p-1));
    col_cons(5) = (1/2.) * h * (phi_dot(2:N_p) + phi_dot(1:N_p-1)) - (phi(2:N_p) - phi(1:N_p-1));

    for i = 1:5
        ceq = [ceq; col_cons(i)]
    end

    % boundary constraints
    bc_y = [y_dot(1) y_dot(N_p) y_dot2(1) y_dot2(N_p)].';
    bc_z = [z_dot(1) z_dot(N_p) z_dot2(1) z_dot2(N_p)].';
    bc_phi = [phi(1) phi(N_p) phi_dot(1) phi_dot(N_p) phi_dot2(1) phi_dot2(N_p)].';
    bc_u = [f_coll(1)-g f_coll(N_p)-g omega_x(1) omega_x(N_p)].';

    lambda_dot = diff(lambda);
    lambda_dot2 = diff(lambda_dot);
    lambda_dot3 = diff(lambda_dot2);
    lambda_dot4 = diff(lambda_dot3);
    bc_lambda = [lambda_dot(1) lambda_dot(N_lambda) lambda_dot2(1) lambda_dot2(N_lambda)...
                 lambda_dot3(1) lambda_dot3(N_lambda) lambda_dot4(1) lambda_dot4(N_lambda)].';



end

function [c, ceq] = mycon(x)
    c = inequality(x)
    ceq = equality(x)
end



% optimization with fmincon
A = [];
b = [];
Aeq = [];
beq = [];

opt_x = fmincon(@evalfucn,x0,A,b,Aeq,beq,@mycon)