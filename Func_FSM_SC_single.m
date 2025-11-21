%% ------------------------------------------------------------
%%  SC Implementation of Transcendental Functions using 2D mesh FSM 
%% ------------------------------------------------------------

% written by: Mehran Moghadam @ ECT lab, Case Western Reserve University

clear

M = 3; N = 3; % M,N are FSM mesh dimension(e.g. 3x3 or 4x4, etc.)

PK = 0.4985; % 0.4985-sin(x) 0.48-cos(x) 0.53-tan(x) 0.5-tanh(x) 0.58-atan(x) 0.5 sigmoid(x),Sinc(x),exp(-x),ln(1+x)

%% To use any of the functions simply uncomment the appropriate Pi one for the desired mesh structure

%% sin(x) for 3x3 mesh
Pi = [0.0035    0.4637    0.4531    0.4597    0.4498    0.5970    0.4462    0.6031    0.8385]; %sin(x)3x3

%% cos(x) for 3x3 mesh
%Pi = [1.0000    1.0000    0.8018    1.0000    0.8321    0.9158    0.8641    0.9325    0.5437];%cos(x)3x3

%% tan(x) for 3x3 mesh
%Pi = [0.0430    0.2461    0.5494    0.2805    0.5412    1    0.5322    1    1];%tan(x)3x3

%% tanh(x) for 3x3 mesh
%Pi = [0.0024    0.4735    0.4172    0.4735    0.4172    0.6055    0.4172    0.6055    0.7451];

%% tanh(x) for 4x4 mesh
%Pi = [0    0.5063    0.2874    0.5417    0.5063    0.2874    0.5417    0.4779    0.2874    0.5417    0.4779    0.5854    0.5417    0.4779    0.5854    0.7469];

%% atan(x) for 3x3 mesh
%Pi = [0.0014    0.4407    0.3655    0.4853    0.4177    0.6926    0.4485    0.6478    0.7267];%atan(x)3x3

%% sigmoid(x) for 3x3 mesh
%Pi = [0.5011    0.6130    0.6193    0.6130    0.6193    0.6434    0.6193    0.6434    0.7302];
%% sigmoid(x) for 4x4 mesh
%Pi = [0.5000    0.6256    0.5800    0.6281    0.6256    0.5800    0.6281    0.6535    0.5800    0.6281    0.6535    0.6318    0.6281    0.6535    0.6318    0.7313];

%% Sinc(x) for 3x3 mesh
%Pi = [1    1    0.9439    1    0.9439    0.9782    0.9439    0.9782    0.8428];%Sinc(x)3x3

%% exp(-x) for 3x3 mesh
%Pi = [0.9947    0.5579    0.6434    0.5579    0.6434    0.5271    0.6434    0.5271    0.3460];
%% exp(-x) for 4x4 mesh
%Pi = [0.9997    0.5057    0.8026    0.5334    0.5057    0.8026    0.5334    0.5483    0.8026    0.5334    0.5483    0.5658    0.5334    0.5483    0.5658    0.3420];

%% ln(1+x) for 3x3 mesh
%Pi = [0.0081    0.4149    0.4270    0.4149    0.4270    0.3869    0.4270    0.3869    0.7741];
%% ln(1+x) for 4x4 mesh
%Pi = [0.0011    0.4818    0.2601    0.3672    0.4818    0.2601    0.3672    0.6341    0.2601    0.3673    0.6341    0.3089    0.3672    0.6341    0.3089    0.7819];

%% Input of the function in binary format in [0,1] interval
Px_input = rand;

BS_length = 256; % SC bit-stream length

func_output = Func_2DmeshFSM_SC(Px_input, PK, Pi, M, N, BS_length); % Call the function with defined parameters


function [mean_out] = Func_2DmeshFSM_SC(Px_input, PK, Pi, M, N, BS_length)
% Simulate the 2-D mesh FSM stochastically.

%mean_out = zeros(size(Px_input));
%var_out = zeros(size(Px_input));

seed_mat = generateBinaryValues(BS_length);
[~,lfsr1] = LFSR3(seed_mat(randi(BS_length),:),BS_length/2,BS_length);
% [~,lfsr2] = LFSR3_2(seed_mat(randi(BS_length),:),BS_length/2,BS_length);
lfsr1 = lfsr1/BS_length;
% lfsr2 = lfsr2/BS_length;

sobol = net(sobolset(2), BS_length);

%for idx = 1:length(Px_input)
     
    %Px = Px_input(idx);        % input probability for this run

    % initialize state (row,col) at (0,0)
    row = 0; col = 0;

    % counters for output ones
    out_ones = 0;
    
    rand_bit = zeros(1,BS_length);
    out_bit = zeros(1,BS_length);
    in_bit = zeros(1,BS_length);
    %in_bit = [1 0 1 0 1 0 0 0 1 0 1 0 0 1 0 0];
    %rand_bit = [0 1 0 0 1 0 1 1 1 1 1 0 0 0 1 1];

    for t = 1:BS_length
        % stochastic input bit (Bernoulli with p=Px)
        %in_bit = rand < Px;
        in_bit(t) = sobol(t,1) < Px_input;
        %in_bit = lfsr1(t) < Px;

        % random bit to decide movement direction
        %rand_bit(t) = rand < PK;   % PK decision
        rand_bit(t) = lfsr1(t) < PK;   % PK decision
        %rand_bit(t) = sobol(t,3) < PK;

        % update state according to mesh FSM transitions
        if in_bit(t)
            if rand_bit(t)
                col = min(col+1,N-1); % move right
            else
                row = min(row+1,M-1); % move up
            end
        else
            if rand_bit(t)
                col = max(col-1,0); % move left
            else
                row = max(row-1,0); % move down
            end
        end

        % output bit from this state:
        % convert Pi(row*N+col+1) to a stochastic bit using random compare
        p_out = Pi(row*N + col + 1);    % per-state output probability
        %out_bit = rand < p_out;
        %out_bit(t) = sobol(t,2) < p_out;
        out_bit(t) = t/BS_length < p_out;
        %out_bit = lfsr2(t) < p_out;

        % accumulate output ones
        out_ones = out_ones + out_bit(t);
    end

    mean_out = out_ones/BS_length; 
    %mean_out(idx) = out_ones/BS_length; % mean output over the run
    %mean_out(idx) = 2*(out_ones/BS_length) - 1;
    %var_out(idx) = (out_ones^2)/BS_length - mean_out(idx)^2;
%end
end