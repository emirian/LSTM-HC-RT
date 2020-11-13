% http://www.mikexcohen.com/lecturelets/eulersconvolution/eulersconvolution.html
function [as] = Mean_Phase_Amp_Extraction_func2(epoch, frex)
data = epoch;
edge_clip = 400;
%% create a Morlet wavelet

srate = 1000; % in hz
time  = -1:1/srate:1; % best practice is to have time=0 at the center of the wavelet
%frex  = 6.5; % frequency of wavelet, in Hz

% create complex sine wave
sine_wave = exp( 1i*2*pi*frex.*time );

% create Gaussian window
s = 7 / (2*pi*frex); % this is the standard deviation of the gaussian
%s = 1 / (2*pi*frex); % this is the standard deviation of the gaussian
gaus_win  = exp( (-time.^2) ./ (2*s^2) );

% now create Morlet wavelet
cmw  = sine_wave .* gaus_win;

%% define convolution parameters

nData = length(data);
nKern = length(cmw);
nConv = nData + nKern - 1;
half_wav = floor( length(cmw)/2 )+1;

%% FFTs

% note that the "N" parameter is the length of convolution, NOT the length
% of the original signals! Super-important!


% FFT of wavelet, and amplitude-normalize in the frequency domain
cmwX = fft(cmw,nConv);
cmwX = cmwX ./ max(cmwX);

% FFT of data
dataX = fft(data,nConv);

% now for convolution...
as = ifft( dataX .* cmwX );

% cut 1/2 of the length of the wavelet from the beginning and from the end
as = as(half_wav-1:end-half_wav);

as = as(edge_clip+1:end-edge_clip);

%% different ways of visualizing the outputs of convolution

%Amp = mean(abs(as));
%Ang = abs(mean(exp(1i*angle (as))));

% for i = 1 :nData
%    abs(mean(exp(Ang(i))))
% end



% figure(2), clf
% plot3(time(1:2000),abs(as),angle(as),'k')
% xlabel('Time (ms)'), ylabel('Amplitude'), zlabel('Phase')
% rotate3d
% set(gca,'xlim',[-300 1200])


end
