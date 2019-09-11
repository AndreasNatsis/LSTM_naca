clearvars -except SD_dat
%%%%%%%%%%% CASE PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
step = 2;                                         % take every second point on the readings
prognose_time =500/step;                          % milliseconds
input_variables = [ 3:22 ];                       % how many variables are taken into account (LSTM)
inputSize = size(input_variables,2);              % how many variables for input (pressure/IMU)
                                                  % 8 pressure / 2*6 IMUs
numHiddenUnits = 30;                              % LSTM neurons per layer
numHiddenUnits1 = 30;                             %
numHiddenUnits2 = 30;                             %                                     
                                                 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

input_data = table2array(SD_dat);
offset_values =[0  mean(input_data(1:30000,2:22))];
for i=2:22
    input_data(:,i)=input_data(:,i)-offset_values(i);
end

input_data(:,1) = round(input_data(:,1)/1000);

colmin = min(input_data);
colmax = max(input_data);
pr_data = rescale(input_data,'InputMin',colmin,'InputMax',colmax);

pr_data(:,1) =input_data(:,1); %% ignore scaling for the first column/time

sd_array =pr_data';



target_encoder = sd_array(2,1:step:end)';
input_data = sd_array(3:22,1:step:end);

SampleRate = 1000/step;
%%Filter podition signal (degrees)
df = designfilt('lowpassfir','FilterOrder',150,'CutoffFrequency',10/SampleRate,'SampleRate',SampleRate);
target_encoder_filtered = filtfilt(df,target_encoder);


for i = prognose_time+1:size(target_encoder_filtered,1)
    target_diff_encoder_filtered(i)= target_encoder_filtered(i)-target_encoder_filtered(i-prognose_time);
end



target_diff_encoder_filtered_rescaled = rescale(target_diff_encoder_filtered);
%data(1,:)=Ang_acceleration;


numTimeStepsTrain = floor(0.60*size(input_data,2));
numTimeStepsValid= floor(0.1*size(input_data,2));
numTimeStepsTest = size(input_data,2)-numTimeStepsTrain-numTimeStepsValid;

XTrain = input_data(:,1:numTimeStepsTrain);

YTrain = target_diff_encoder_filtered_rescaled(1,prognose_time + 1:numTimeStepsTrain+prognose_time);


XValidation = input_data(:,numTimeStepsTrain+1:end-prognose_time-numTimeStepsTest);
YValidation = target_diff_encoder_filtered_rescaled(1,numTimeStepsTrain+prognose_time+1:end-numTimeStepsTest);

XTest = input_data(:,numTimeStepsTrain+numTimeStepsValid+1:end-prognose_time);
YTest = target_diff_encoder_filtered_rescaled(1,numTimeStepsTrain+numTimeStepsValid+prognose_time+1:end);

% mu = mean(XTrain,2);
% sig = std(XTrain,0,2);
% for i=1:size(data,1)
%     XTrain(i,:) = (XTrain(i,:) - mu(i))/sig(i);
%     
%     XTest(i,:) = (XTest(i,:) - mu(i))/sig(i);
%     
%     
% end
% YTrain(1,:) = (YTrain(1,:) - mu(1))/sig(1);
% YTest(1,:) = (YTest(1,:) - mu(1))/sig(1);
%% Define LSTM Network Architecture
% Create an LSTM regression network. Specify the LSTM layer to have 200 hidden units.


numResponses = 1;


layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer(0.3)
    lstmLayer(numHiddenUnits1,'OutputMode','sequence')
    dropoutLayer(0.3)
    lstmLayer(numHiddenUnits2)  %,'OutputMode','last'
    dropoutLayer(0.3)
    fullyConnectedLayer(numResponses)
    regressionLayer];

%% Specify the training options. Set the solver to 'adam' and train for 250 epochs. To prevent the gradients from exploding, set the gradient threshold to 1. Specify the initial learn rate 0.005, and drop the learn rate after 125 epochs by multiplying by a factor of 0.2.

opts = trainingOptions('adam', ...
    'MaxEpochs',1500, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',100, ...
    'LearnRateDropFactor',0.6, ...
    'Verbose',0, ...
    'Plots','training-progress',...
    'ValidationData',{XValidation,YValidation},...
    'ValidationPatience',100,...
    'ValidationFrequency',20);

%% Train LSTM Network
%Train the LSTM network with the specified training options by using trainNetwork.

net = trainNetwork(XTrain,YTrain,layers,opts);




%% unstandardize the predictions and the test data

YPred = predict(net,XTest);


% YPred = YPred*sig(1) + mu(1);
% YTest(1,:) = YTest(1,:)*sig(1) + mu(1);
% 
% YTest = YTest/1000;
% YPred = YPred/1000;
% 
% 
% Mean_sq_err = mean((YPred-YTest).^2)
figure(10)

plot([1:2:2*size(XTest,2)]/(1000),YTest,[1:2:2*size(XTest,2)]/(1000),YPred)

% ax = gca;
% ax.XAxis.Exponent = 3;

YPred2 = predict(net,XTrain);

% 
% figure(10)
% 
% plot([1:size(XTest,2)],YTest,[1:size(XTest,2)],YPred)
% 
% ax = gca;
% ax.XAxis.Exponent = 3;