%% This code creates a memory map of the binary file
close all; clear all

samp_freq = 30000; %sampling frequency (in Hz)

nchan = 6; %this has to be known in advance (if wrong, data will be loaded incorrectly)

[FileName,PathName,FilterIndex] = uigetfile('.dat'); %select file

contFile=fullfile(PathName,FileName); %create full file path with file name

s=dir(contFile);
file_size=s.bytes; %determine file size in byte

samples=file_size/2/nchan;

m=memmapfile(contFile,'Format',{'int16' [nchan samples] 'mapped'}); %create memory map of the file

data = m.Data;

%to create a .mat variable with the matrix containing the data for all
%channels, you need to convert the memory map to double precision use: data_bin = double(data.mapped)
% If your PC does not have enought memory to open the complete file, you can open a specific electrode site(x):
% chan(x,:) = double(data.mapped(x)

data_bin = double(data.mapped); % converte o memory map para double
save('continuous23.mat', 'data_bin', '-v7.3'); % guarda os dados num ficheiro .mat


