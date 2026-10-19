%% 2D Mesh FSM for sin(x) using parameters from the paper (Type-2B FSM)
clear

% ------------------------------------------------------------
% PARAMETERS
% ------------------------------------------------------------
States = 16;   % 4x4 mesh
M = 4; N = 4;
Px2 = 0.4895; % 0.4985 for sin(x)-- 0.48 for cos(x),tanh(x)
Interval = 0.01;
PX = Interval : Interval : 1 - Interval;
numSamples = numel(PX);

% ------------------------------------------------------------
% TARGET FUNCTION (sin(x) Maclaurin expansion, x∈[0,1])
% ------------------------------------------------------------
PTarget = PX - (1/6)*PX.^3 + (1/120)*PX.^5 - (1/5040)*PX.^7; % sin(x)
%PTarget = 1 - (1/2)*PX.^2 + (1/factorial(4))*PX.^4 - (1/factorial(6))*PX.^6 + (1/factorial(8))*PX.^8; % cos(x)
%PTarget = PX - (PX.^3)/3 + 2*(PX.^5)/15 - 17*(PX.^7)/315; % tanh(x)
% optional higher-order terms can be added:
% PTarget = PTarget + (1/factorial(9))*PX.^9 - (1/factorial(11))*PX.^11;

% ------------------------------------------------------------
% 2D Mesh FSM core (identical to authors' code)
% ------------------------------------------------------------
tx = (Px2 / (1 - Px2)) * (PX ./ (1 - PX));
ty = ((1 - Px2) / Px2) * (PX ./ (1 - PX));

denominator = zeros(1, numSamples);
for ki = 0 : M - 1
    for kj = 0 : N - 1
        denominator = denominator + (tx.^ki) .* (ty.^kj);
    end
end

% Build Φ (basis functions)
Phi = zeros(numSamples, States);
for k = 0 : States - 1
    mi = mod(k, N);
    mj = fix(k / N);
    Phi(:, k + 1) = (tx.^mi) .* (ty.^mj) ./ denominator;
end

% ------------------------------------------------------------
% Solve for Z using constrained least squares (bounded [0,1])
% ------------------------------------------------------------
H = Phi' * Phi;
c = -Phi' * PTarget';
H = (H + H') / 2;
Z = quadprog(2*H, 2*c, [], [], [], [], zeros(States,1), ones(States,1));

% ------------------------------------------------------------
% Normalize and compute output
% ------------------------------------------------------------
Z = Z(:)';
PY = Phi * Z';
MSE = mean((PY - PTarget').^2);

% ------------------------------------------------------------
% Display results
% ------------------------------------------------------------
fprintf('==== 2D Mesh FSM sin(x) synthesis (Paper-accurate) ====\n');
fprintf('Px2 = %.4f  |  MxN = %dx%d\n', Px2, M, N);
fprintf('Mean Squared Error = %.3e\n', MSE);
disp('Z (weights):');
disp(Z);

% Compare with authors' reported Z
% Z_author = [0 0.1548 0 0.0176 1 0.1689 1 0.9074 0 1 0 0 1 0.3868 1 0.8524];
% fprintf('||Z - Z_author||2 = %.3e\n', norm(Z - Z_author));

% ------------------------------------------------------------
% Plot results
% ------------------------------------------------------------
figure('Color','white','Position',[100 100 800 550]);
plot(PX, PTarget, 'k', 'LineWidth', 1.8); hold on;
plot(PX, PY, 'r--', 'LineWidth', 1.8);
legend('Target sin(x)', 'FSM Approximation', 'Location', 'North');
xlabel('P_X', 'FontSize', 14);
ylabel('f(P_X)', 'FontSize', 14);
title(sprintf('2D Mesh FSM Approximation (Px2=%.4f, MSE=%.2e)', Px2, MSE));
grid on;
