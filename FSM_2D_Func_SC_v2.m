clear
tic
% Mesh dimensions:
M = 4; N = 4; 
PK = 0.4985; % 0.4985-sin(x) 0.48-cos(x) 0.53-tan(x) 0.5-tanh(x) 0.58-atan(x) 0.5 sigmoid(x),Sinc(x),exp(-x),ln(1+x)

inj_rate = 20;
 
%Pi = [0.0035    0.4637    0.4531    0.4597    0.4498    0.5970    0.4462    0.6031    0.8385]; %sin(x)3x3
%Pi = [0 0.5035 0.3089 0.5270 0.5033 0.3081 0.5277 0.5511 0.3074 0.5285 0.5515 0.5651 0.5292 0.5519 0.5654 0.8418]; %sin(x)4x4
%Pi = [1.0000    1.0000    0.8018    1.0000    0.8321    0.9158    0.8641    0.9325    0.5437];%cos(x)3x3
%Pi = [1.0000 0.9983 0.8285 0.9731 0.9989 0.8434 0.9765 0.6909 0.8560 0.9799 0.7067 0.9596 0.9832 0.7234 0.9673 0.5401];%cos(x)4x4
%Pi = [1 0.9983 0.8285 0.9731 0.9989 0.8434 0.9765 0.6909 0.8560 0.9799 0.7067 0.9596 0.9832 0.7234 0.9673 0.5401]; %cos(x)

%Pi = [0.0430    0.2461    0.5494    0.2805    0.5412    1    0.5322    1    1];%tan(x)3x3
%Pi = [0    0.4845    0.5966    0    0.4883    0.5880    0    1    0.5778    0    1    1    0    1    1    1]; %tan(x)4x4
%Pi = [0 0.5066 0.2950 0.5293 0.5032 0.2871 0.5343 0.4934 0.2786 0.5413 0.4852 0.5801 0.5525 0.4667 0.5906 0.7470]; % tanh(x)
%Pi = [0.0024    0.4735    0.4172    0.4735    0.4172    0.6055    0.4172    0.6055    0.7451];% tanh(x)3x3
%Pi = [0    0.5063    0.2874    0.5417    0.5063    0.2874    0.5417    0.4779    0.2874    0.5417    0.4779    0.5854    0.5417    0.4779    0.5854    0.7469];%tanh(x)4x4
%Pi = [0.0014    0.4407    0.3655    0.4853    0.4177    0.6926    0.4485    0.6478    0.7267];% atan(x)3x3
%Pi = [0    0.4786    0.2156    0.6199    0.5096    0.2338    0.5923    0.3880    0.2719    0.5597    0.4260    0.6928    0.5404    0.4507    0.6449    0.7265];%atan(x)4x4
%Pi = [0.5011    0.6130    0.6193    0.6130    0.6193    0.6434    0.6193    0.6434    0.7302];%sigmoid(x)3x3
%Pi = [0.5000    0.6256    0.5800    0.6281    0.6256    0.5800    0.6281    0.6535    0.5800    0.6281    0.6535    0.6318    0.6281    0.6535    0.6318    0.7313];%sigmoid(x)4x4
%Pi = [1    1    0.9439    1    0.9439    0.9782    0.9439    0.9782    0.8428];%Sinc(x)3x3
%Pi = [1    0.9996    0.9469    0.9951    0.9996    0.9469    0.9951    0.8981    0.9469    0.9951    0.8981    0.9927    0.9951    0.8981    0.9927    0.8414];%Sinc(x)4x4
%Pi = [0.9947    0.5579    0.6434    0.5579    0.6434    0.5271    0.6434    0.5271    0.3460];%exp(-x)3x3
%Pi = [0.9997    0.5057    0.8026    0.5334    0.5057    0.8026    0.5334    0.5483    0.8026    0.5334    0.5483    0.5658    0.5334    0.5483    0.5658    0.3420];%exp(-x)4x4
%Pi = [0.0081    0.4149    0.4270    0.4149    0.4270    0.3869    0.4270    0.3869    0.7741];%ln(1+x)3x3
Pi = [0.0011    0.4818    0.2601    0.3672    0.4818    0.2601    0.3672    0.6341    0.2601    0.3673    0.6341    0.3089    0.3672    0.6341    0.3089    0.7819];%ln(1+x)4x4

BS_Len = 1024; % SC bit-stream length
% input probabilities to test
Interval = 1/100; % 0.01
%Px_values = Interval : Interval : 1 - Interval;

%Px_values = rand(1,100);
%Px_values = rand;
%Px_values = 0.0:0.05:1.0;
% x_bp = -1:0.05:1;
% Px_values = (x_bp + 1)/2;

% run stochastic simulation with 100000 cycles per Px



iterations = 1e3;
AE = zeros(1, iterations); % Preallocate mean_out for efficiency
AE_noisy_input = zeros(1, iterations);
AE_noisy_output = zeros(1, iterations);
AE_noisy_input_output = zeros(1, iterations);
AE_noisy_state = zeros(1, iterations);
AE_noisy_all = zeros(1, iterations);

for iter = 1:iterations
    Px_values = rand;
    [mean_out, mean_out_noisy_input, mean_out_noisy_output, mean_out_noisy_input_output, mean_out_noisy_state, mean_out_noisy_all] = Func2DmeshFSM_stochastic(Px_values,PK,Pi,M,N,BS_Len, inj_rate);
%end
    % compare to target sin(x)
    %PTarget = Px_values - (1/6) * Px_values .^ 3 + (1/120) * Px_values .^ 5 - (1/5040)*Px_values.^7; % sin(x) truncated
    %PTarget = 1 - (1/2)*Px_values.^2 + (1/factorial(4))*Px_values.^4 - (1/factorial(6))*Px_values.^6 + (1/factorial(8))*Px_values.^8; % cos(x)
    %PTarget = (Px_values) + (Px_values.^3)/3 + (2/15)*(Px_values.^5) + (17/315)*(Px_values.^7); % tan(x)
    %PTarget = Px_values - (Px_values.^3)/3 + 2*(Px_values.^5)/15 - 17*(Px_values.^7)/315; % tanh(x)
    %PTarget = Px_values - (Px_values.^3)/3 + (Px_values.^5)/5 - (Px_values.^7)/7; % arctan(x)
    %PTarget = 1/2 + Px_values/4 - (Px_values.^3)/48 + (Px_values.^5)/480; % sigmoid(x)
    %PTarget = 1 - (Px_values.^2)/factorial(3) + (Px_values.^4)/factorial(5) - (Px_values.^6)/factorial(7); % Sinc(x)
    %PTarget = 1- Px_values + (Px_values.^2)/factorial(2) - (Px_values.^3)/factorial(3) + (Px_values.^5)/factorial(5); % exp(-x)
    PTarget = Px_values - (Px_values.^2)/2 + (Px_values.^3)/3 - (Px_values.^4)/4 + (Px_values.^5)/5; % ln(1+x)
    %if PTarget > 1 && mean_out > 0.9
     %   AE(iter) = abs(min(PTarget,1) - mean_out);
    %else
        AE(iter) = abs(PTarget - mean_out);
        AE_noisy_input(iter) = abs(PTarget - mean_out_noisy_input);
        AE_noisy_output(iter) = abs(PTarget - mean_out_noisy_output);
        AE_noisy_input_output(iter) = abs(PTarget - mean_out_noisy_input_output);
        AE_noisy_state(iter) = abs(PTarget - mean_out_noisy_state);
        AE_noisy_all(iter) = abs(PTarget - mean_out_noisy_all);
    %end
end



MAE_percentage = mean(AE)*100;
MAE_noisy_input_percentage = mean(AE_noisy_input)*100;
MAE_noisy_output_percentage = mean(AE_noisy_output)*100;
MAE_noisy_input_output_percentage = mean(AE_noisy_input_output)*100;
MAE_noisy_state_percentage = mean(AE_noisy_state)*100;
MAE_noisy_all_percentage = mean(AE_noisy_all)*100;

fprintf('\n MAE (%%): %.2f, std: %.3f \n\n',MAE_percentage, std(AE))
fprintf('\n MAE (%%) with noise rate of %d%% in the input is: %.2f, std: %.3f \n\n',inj_rate, MAE_noisy_input_percentage, std(AE_noisy_input))
fprintf('\n MAE (%%) with noise rate of %d%% in the output is: %.2f, std: %.3f \n\n',inj_rate, MAE_noisy_output_percentage, std(AE_noisy_output))
%fprintf('\n MAE (%%) with noise rate of %d%% in input-Output is: %.2f, std: %.3f \n\n',inj_rate, MAE_noisy_input_output_percentage, std(AE_noisy_input_output))
fprintf('\n MAE (%%) with noise rate of %d%% in State is: %.2f, std: %.3f \n\n',inj_rate, MAE_noisy_state_percentage, std(AE_noisy_state))
fprintf('\n MAE (%%) with noise rate of %d%% in All is: %.2f, std: %.3f \n\n',inj_rate, MAE_noisy_all_percentage, std(AE_noisy_all))

% figure; 
% plot(Px_values, PTarget, 'k','LineWidth',1.5); hold on;
% plot(Px_values, mean_out,'r--o','LineWidth',1.5);
% legend('Target sin(x)','Stochastic FSM');
% xlabel('Px'); ylabel('f(Px)');
%MAE_percentage = mean(abs(PTarget - mean_out),'all') * 100

toc

function [mean_out, mean_out_noisy_input, mean_out_noisy_output, mean_out_noisy_input_output, mean_out_noisy_state, mean_out_noisy_all] = Func2DmeshFSM_stochastic(Px_values, PK, Pi, M, N, numCycles, inj_rate)
% Simulate the 2-D mesh FSM stochastically.
% Px_values : vector of input probabilities to test (e.g. 0:0.1:1)
% PK        : mesh parameter (e.g. 0.4985)
% Pi        : per-state output probabilities (vector length M*N)
% M,N       : mesh dimensions (e.g. 4,4)
% numCycles : how many stochastic bits per Px to run

%S = M*N;                   % total states
%WIDTH = 8;                 % quantization bits for output compare (optional)

%mean_out = zeros(size(Px_values));
%var_out = zeros(size(Px_values));

% seed_mat = generateBinaryValues(numCycles);
% [~,lfsr1] = LFSR3(seed_mat(randi(numCycles),:),numCycles/2,numCycles);
% [~,lfsr2] = LFSR3_2(seed_mat(randi(numCycles),:),numCycles/2,numCycles);
% lfsr1 = lfsr1/numCycles;
% lfsr2 = lfsr2/numCycles;

sobol = net(sobolset(10), numCycles);

precision = log2(numCycles);
vd = zeros(numCycles, precision); % Initialize the van der Corput matrix
for i=1:precision
    vd(:,i) = vdcorput(numCycles-1,2^i);
end


% TRNG = fileread('randomorg.txt');
% TRNG = TRNG - '0';
% rr1 = randi([1, floor(length(TRNG))-numCycles]);  
% TRNG_mat_meh1 = TRNG(rr1:rr1 + numCycles-1);    % First TRNG
% bs_trng1 = TRNG_mat_meh1;
% rr2 = randi([1, floor(length(TRNG))-numCycles]);  
% TRNG_mat_meh2 = TRNG(rr2:rr2+numCycles-1);      % Second TRNG
% bs_trng2 = TRNG_mat_meh2;

%for idx = 1:length(Px_values)
    %Px = Px_values(idx);        % input probability for this run
    Px = Px_values;

    % initialize state (row,col) at (0,0)
    row = 0; col = 0;
    row_n = 0; col_n = 0;   % rows and columns for noisy input
    row_n_s = 0; col_n_s = 0; % rows & columns for noisy states
    row_n_i_s = 0; col_n_i_s = 0; % rows & columns for noisy input and states

    % counters for output ones
    %out_ones = 0;
    
    %rand_bit = zeros(1,numCycles);
    out_bit = zeros(1,numCycles);
    out_bit_noisy = zeros(1,numCycles);
    out_bit_noisy_s = zeros(1,numCycles);
    out_bit_noisy_i_s = zeros(1,numCycles);
    %in_bit = zeros(1,numCycles);
    %in_bit = [1 0 1 0 1 0 0 0 1 0 1 0 0 1 0 0];
    %rand_bit = [0 1 0 0 1 0 1 1 1 1 1 0 0 0 1 1];

    % for sin->(4,1), cos->(3,2),(3,1)
    
    %% stochastic input bit (Bernoulli with p=Px)
    in_bit = Px > sobol(:,4);
    in_bit_noisy = noise_injection(in_bit, inj_rate);
    %% random bit to decide movement direction
    rand_bit = PK > sobol(:,8);
    rand_bit_noisy = noise_injection(rand_bit, inj_rate);

    for t = 1:numCycles
        % stochastic input bit (Bernoulli with p=Px)
        %in_bit(t) = rand < Px;
        %in_bit(t) = sobol(t,4) < Px;   % Sobol
        %in_bit(t) = vd(t,6) < Px;      % VDC-2^n
        %in_bit = lfsr1(t) < Px;        % LFSR
        %in_bit(t) = bs_trng1(t);        % TRNG

        % random bit to decide movement direction
        %rand_bit(t) = rand < PK;   % PK decision
        %rand_bit(t) = sobol(t,8) < PK;  % Sobol
        %rand_bit(t) = vd(t,1) < PK;    % VDC-2^n
        %rand_bit(t) = bs_trng2(t);      % TRNG

        % update state according to mesh FSM transitions
        if in_bit(t)
            if rand_bit(t)
                col = min(col+1,N-1); % move right
                col_n_s = min(col_n_s+1,N-1); % move right
            else
                row = min(row+1,M-1); % move up
                row_n_s = min(row_n_s+1,M-1); % move up
            end
        else
            if rand_bit(t)
                col = max(col-1,0); % move left
                col_n_s = max(col_n_s-1,0); % move left
            else
                row = max(row-1,0); % move down
                row_n_s = max(row_n_s-1,0); % move down
            end
        end
        
        %noisy
        if in_bit_noisy(t)
            if rand_bit_noisy(t)
                col_n = min(col_n+1,N-1); % move right
                col_n_i_s = min(col_n_i_s+1,N-1); % move right
            else
                row_n = min(row_n+1,M-1); % move up
                row_n_i_s = min(row_n_i_s+1,M-1); % move up
            end
        else
            if rand_bit_noisy(t)
                col_n = max(col_n-1,0); % move left
                col_n_i_s = max(col_n_i_s-1,0); % move left
            else
                row_n = max(row_n-1,0); % move down
                row_n_i_s = max(row_n_i_s-1,0); % move down
            end
        end

        % Injecting noise only to states, so the FSM may traverse different
        % row/column than the desired one
        %row_n_s = row;
        %col_n_s = col;
        if rand < (inj_rate / 100)
            row_n_s = randi([0, M-1]); % Random new row
            col_n_s = randi([0, N-1]); % Random new col
        end

        % Injecting noise to both input and states
        if rand < (inj_rate / 100)
            row_n_i_s = randi([0, M-1]); % Random new row
            col_n_i_s = randi([0, N-1]); % Random new col
        end

        % output bit from this state:
        % convert Pi(row*N+col+1) to a stochastic bit using random compare
        p_out = Pi(row*N + col + 1);    % per-state output probability
        p_out_noisy = Pi(row_n*N + col_n + 1); 
        p_out_noisy_state = Pi(row_n_s*N + col_n_s + 1);
        p_out_noisy_input_state = Pi(row_n_i_s*N + col_n_i_s + 1);

        %out_bit(t) = rand < p_out;
        out_bit(t) = sobol(t,2) < p_out;
        out_bit_noisy(t) = sobol(t,2) < p_out_noisy;
        out_bit_noisy_s(t) = sobol(t,2) < p_out_noisy_state;
        out_bit_noisy_i_s(t) = sobol(t,2) < p_out_noisy_input_state;
        %out_bit(t) = vd(t,3) < p_out;
        %out_bit = lfsr2(t) < p_out;
        % out_bit_noisy(t) = out_bit(t);
        % flip_mask = rand(size(out_bit_noisy)) < inj_rate/100;
        % out_bit_noisy(flip_mask) = ~out_bit_noisy(flip_mask);

        % accumulate output ones
        %out_ones = out_ones + out_bit(t);
    end

    %out_bit = p_out > sobol(:,2);

    out_bit_noisy_output = noise_injection(out_bit, inj_rate);
    out_bit_noisy_input_output = noise_injection(out_bit_noisy, inj_rate);
    out_bit_noisy_all = noise_injection(out_bit_noisy_i_s, inj_rate);

    %mean_out(idx) = out_ones/numCycles; % mean output over the run
    %mean_out = out_ones / numCycles;
    mean_out = mean(out_bit);
    mean_out_noisy_input = mean(out_bit_noisy);
    mean_out_noisy_output = mean(out_bit_noisy_output);
    mean_out_noisy_input_output = mean(out_bit_noisy_input_output);
    mean_out_noisy_state = mean(out_bit_noisy_s);
    mean_out_noisy_all = mean(out_bit_noisy_all);

    %mean_out(idx) = 2*(out_ones/numCycles) - 1;
    %var_out(idx) = (out_ones^2)/numCycles - mean_out(idx)^2;
%end
end



function bs_noisy = noise_injection(bitstream, injection_pct)
if injection_pct < 0, injection_pct = 0; end
if injection_pct > 100, injection_pct = 100; end
p_flip = injection_pct / 100;
bs = bitstream;
% accept binary in {0,1} or bipolar in {-1,1}
is_bipolar = all(ismember(unique(bs(:)), [-1,1]));
if is_bipolar
    % map to 0/1 for flipping convenience
    bs01 = (bs == 1);
else
    bs01 = logical(bs);
end
% create a logical mask of the same size as bs01 where each element is true
% with probability p_flip; these positions will be flipped

%% Uniform distribution
flip_mask = rand(size(bs01)) < p_flip; 

%% Gaussian distribution
% mu = 0.5;       % Center of the distribution
% sigma = 0.15;   % Spread of the distribution
% % Create and truncate the distribution
% d = makedist('Normal', 'mu', mu, 'sigma', sigma);
% d_trunc = truncate(d, 0, 1);
% % Generate a 1000x1 vector
% rand_gauss = random(d_trunc, size(bs01));
% flip_mask = rand_gauss < p_flip; 


bs01(flip_mask) = ~bs01(flip_mask);
if is_bipolar
    bs_noisy = double(bs01)*2 - 1; % map back to -1/+1
else
    bs_noisy = double(bs01);
end
end


