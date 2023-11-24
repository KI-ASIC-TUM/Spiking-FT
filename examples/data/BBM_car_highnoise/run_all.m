%{
% BlackBox Simulator
% Infineon Technologies AG (2020)
% - Confidential - 
% provided to TUM for the KI-ASIC project
% Last Modification: Hille Julian (2020-07-16)
%}
rng(0)
% add necessary folders
addpath('lib')
addpath('src')
addpath('data')

% =========================================================================
% Define Parameters
% =========================================================================

mmic.bandwidth = 607.7e6;         % chirp bandwidth   [Hz]
mmic.f0 = 76e9;                 % initial frequency [Hz]
mmic.c0 = 299792458;                  % speed of ligth [m/s]

mmic.Nrange = 512;             % number of range samples
mmic.Nramps = 1;              % number of ramps
%mmic.fcHPF = 200e3;             % corner Frequency HPF [Hz]
mmic.fcHPF = 200e3;             % corner Frequency HPF [Hz]
mmic.fcHPF2 = 0;                % corner Frequency HPF2 [Hz] (0 to deactivate)

% - Define Ramp - %
mmic.Tstart = 4e-6;            % ramp start time [s]
mmic.Tramp  = 2*20.48e-6+0e-6;            % ramp payload duration [s]
mmic.Trep   = mmic.Tstart + mmic.Tramp + 12e-6;            % ramp duration (complete) [s]

% channel setup
mmic.Nrx = 1;                   % number of RX channels
mmic.Ntx = 1;                   % number of TX channels

n_sample_bandwidth = 725
eff_b = mmic.Nrange*mmic.bandwidth/(50e6*mmic.Tramp/2+1)
%eff_b = mmic.Nrange*mmic.bandwidth/(n_sample_bandwidth)
d_res = mmic.c0/(2*eff_b)
d_max = mmic.Nrange/2*d_res

% model level = 0: incl (constant fc) HPF, no AFE/ADC noise & saturation
% model level = 1: incl. HPF, incl. AFE noise, ADC noise
%                  To set AFE gain, use G_afe
% model level = 2: LPF prototype filter represented, 
%                  fc variation supported, 
%                  exact floating point reprresentation of DFE, 
%                  no internal aliasing of DFE
%                  CURRENTLY ONLY decimations 2, 4, 8 supported
%                  ADC nonlinearity (SFDR)
%                  AFE noise vs. gain setting
%                  To set AFE gain, use N_Gafe
mmic.model_level = 1;
mmic.Gafe = 20;               % AFE gain [dB]
mmic.N_Gafe = 6;                % select gain configuration 
                                % 6 => 20 dB, 7 => 26 dB 8 => 32 dB

% System settings
mmic.tia_clipping = 0;          % activate TIA clipping
mmic.afenoise = 0;              % activate AFE noise 
mmic.adcnoise = 0;              % activate ADC noise
mmic.phasenoise = 0;            % activate Phase noise
mmic.thermalnoise = 0;          % activate Thermal noise
mmic.pll = 0;                   % activate PLL
mmic.verbose = 0;               % activate verbose 
mmic.adc.sat_on = 0;            % activate adc saturation
mmic.adc.nonlin_on = 0;         % activate ADC non-linearity

% Visualization
mmic.plot.range_doppler = 0;    % Plot range doppler (simulation only)

n_distances = 200;
create_dataset(15, 0.9, './data/BBM_car_lownoise/', n_distances, d_max, mmic)
create_dataset(0, 0.5, './data/BBM_pedestrian_lownoise/', n_distances, d_max, mmic)

% Switch noise ON
mmic.afenoise = 1;              % activate AFE noise 
mmic.adcnoise = 1;              % activate ADC noise
mmic.phasenoise = 1;            % activate Phase noise
mmic.thermalnoise = 1;          % activate Thermal noise
mmic.adc.sat_on = 1;            % activate adc saturation
mmic.adc.nonlin_on = 1;         % activate ADC non-linearity


create_dataset(15, 0.9, './data/BBM_car_highnoise/', n_distances, d_max, mmic)
create_dataset(5, 0.5, './data/BBM_pedestrian_highnoise/', n_distances, d_max, mmic)

function create_dataset(rcs, dmax_ratio, path, n_distances, d_max, mmic)
    target = [];
    target_idx = 0
    data = [];
    targets = [];
    for d = 2:n_distances+2
        
        k = 1;
        distance = dmax_ratio*d_max/n_distances*d;
        target(k).name = ['target_' num2str(target_idx)];
        target(k).r0 = [distance, 0, 0];      % range (x,y,z) [m]
        target(k).v  = [0.0, 0, 0];       % velocity (x,y,z) [m/s]
        target(k).rcs = rcs;              % Radar Cross Section  [dBsm]
        target(k).given_level = [];        % if provided, Radar equation for level 
                                        % calculation is bypassed [dBm @RX]
        target(k).phase = 0.0;             % phase [rad]
    
        target_idx = target_idx + 1
    
        % run BlackBox Model and return output samples
        [samples, rtrx, t] = BBM_run0(mmic,target);
    
    
        % Copy all samples
        sig_matrix = samples.rxmat;
        data = cat(4,data, [sig_matrix]);
        targets = cat(3, targets, [target]);
    end
    % Define Folder where to dump the files
    %DUMP_PATH = ['./data/BBM/']
    DUMP_PATH = [path]
    if ~exist(DUMP_PATH,'dir')
           mkdir(DUMP_PATH)
    end
    size(data)
    f_name = [DUMP_PATH 'data.mat'];
    save(f_name,'data');
    size(targets)
    f_name = [DUMP_PATH 'targets.mat'];
    save(f_name,'targets');
    f_name = [DUMP_PATH 'config.mat'];
    save(f_name,'mmic');
    f_name = [DUMP_PATH 'rtrx_config.mat'];
    save(f_name,'rtrx');
    disp(rtrx)
    disp('Files saved')
    copyfile('run_all.m', DUMP_PATH) 
end
