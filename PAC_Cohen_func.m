function [PAC, dPAC] = PAC_Cohen_func(data, LB_lf, LB_hf, HB_lf, HB_hf)

srate = 1000;
% -- band-pass filter
thetaband           = [LB_lf LB_hf];
gammaband           = [HB_lf HB_hf];

edge_clip = 300;

gamma_filt_order    = round(1.5*(srate/gammaband(1)));
gamma_ffrequencies1 = [mean(gammaband)-(gammaband(2)-gammaband(1)) mean(gammaband)+(gammaband(2)-gammaband(1))]/(srate/2);

gamma_filterweights = fir1(gamma_filt_order,gamma_ffrequencies1); 

theta_filt_order    = round(1.5*(srate/thetaband(1)));
theta_ffrequencies1  = [mean(thetaband)-(thetaband(2)-thetaband(1)) mean(thetaband)+(thetaband(2)-thetaband(1))]/(srate/2);

theta_filterweights = fir1(theta_filt_order,theta_ffrequencies1);


gammafilt = filtfilt(gamma_filterweights,1,data);
thetafilt = filtfilt(theta_filterweights,1,data);

%%

% -- compute theta phase angles and gamma power
% -- first and last 300 pnts are removed to account for edge artifacts
thetaphase = angle(hilbert(thetafilt(edge_clip+1:end-edge_clip)));
gammapower = abs(hilbert(gammafilt(edge_clip+1:end-edge_clip)));

PAC  = abs(mean(exp(1i*thetaphase) .* gammapower));
dPAC  = abs(mean( (exp(1i*thetaphase) - mean(exp(1i*thetaphase))) .* gammapower));
end
