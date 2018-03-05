%% Make all randomly generated samples reusable
rng('default');
rng(1)

%% Normalize X's and Y's and merge back
normX = normc(DataX); 
normY = normc(DataY)
finaldata = [normX normY];

%% Data Visualization

%Correlation Matrix
CorCoe = corrcoef(normX);

%Correlation Plots for most correlated features
mat1 = [ normX(:,3) normX(:,6) ];
mat2 = [ normX(:,3) normX(:,7) ];
mat3 = [ normX(:,5) normX(:,9) ];
    
corrplot(mat1, 'varNames',{'Finishinng','Composure'});
corrplot(mat2, 'varNames',{'Finishinng','Off the Ball'});
corrplot(mat3, 'varNames',{'Heading','Strenght'});

%Boxplot
boxplot(finaldata(:,[1:9]), 'Label', name)
figure
plot(1:100,DataY,'b')
ylabel('Value in k£ ','FontSize',15)
xlabel('n°/id of the player','FontSize',15)
hold on
plot(1:100,DataX(:,2),'r')
h_legend = legend({'Value at the peak','Initial value'})
set(h_legend, 'FontSize', 16)
hold off

%% Basic statistics
Data = [DataX DataY]
std_bef = std(Data)
std_aft = std(finaldata)
mean_bef = mean(Data)
mean_aft = mean(finaldata);
kurt_bef = kurtosis(Data)
kurt_aft = kurtosis(finaldata)
skew_bef = skewness(Data)
skew_aft = skewness(finaldata)
med_bef = median(Data)
med_aft = median(finaldata)

%% Partition dataset into train and test data

X = finaldata(:,1:9)
Y = finaldata(:,10)
cvpart = cvpartition(Y,'holdout',0.3);
Xtrain = X(training(cvpart),:);
Ytrain = Y(training(cvpart),:);
Xtest = X(test(cvpart),:);
Ytest = Y(test(cvpart),:);

%% Random Forest

% We create a a loop that returns a matrix of MSE according to different combinations
% of MinLeafSize and Numpredictorstosample

MseRandomForest = [,]; % Matrix of MSE with different combinations


for k=1:10;
    for j=1:8;
        
    Mdl = TreeBagger(5,Xtrain,Ytrain,'Method','Regression','MinLeafSize',k,'NumPredictorsToSample',j)
    
    runTest = predict(Mdl,Xtest);
    a = abs(Ytest-runTest).^2;
    MseRandomForest(k,j) = sum(a(:))/numel(Ytest);
    end;

end;

%Returning the lowest MSE / best combination of parameters
[r,c]=find(MseRandomForest==min(min(MseRandomForest)))
MSE_rf_min = min(min(MseRandomForest))

%Graphing the MSE Matrix as a heatmap
X = [1:8];
Y = [1:10];
subplot(1,1,1);
imagesc(flipud(MseRandomForest)), axis equal tight, colorbar;
set(gca, 'YTick', 1:10, 'YTickLabel', 10:-1:1);
title('MSE Matrix Heatmap');
xlabel('Number Predictor to Sample');
ylabel('Minimum Leaf Size');
colormap jet;

%Plotting predictor importance 
%rf = TreeBagger(5,Xtrain,Ytrain,'Method','Regression','MinLeafSize',2,'NumPredictorsToSample',7,...
%figure;
%bar(imp);
%title('Curvature Test');
%ylabel('Predictor importance estimates');
%xlabel('Predictors');
%h = gca;
%h.XTickLabel = rf.PredictorNames;
%h.XTickLabelRotation = 45;
%h.TickLabelInterpreter = 'none';



%% Linear regression

% We try the model using different links

%inverse gaussian link
Beta = glmfit(Xtrain,Ytrain,'inverse gaussian');
prediction = glmval(Beta, Xtest, -2);
a = abs(Ytest-prediction).^2;
MSE_inversegaussian = sum(a(:))/numel(Ytest);

%poisson distribution
Beta2 = glmfit(Xtrain,Ytrain,'poisson');
prediction2 = glmval(Beta2, Xtest, 'log');
a = abs(Ytest-prediction2).^2;
MSE_poisson = sum(a(:))/numel(Ytest);

%Gamma distribution
Beta3 = glmfit(Xtrain,Ytrain,'gamma');
prediction3 = glmval(Beta3, Xtest, 'reciprocal');
a = abs(Ytest-prediction3).^2;
MSE_gamma = sum(a(:))/numel(Ytest);

%Normal distribution
% This link gives us the best model with the lowest Mean Square Error
Beta4 = glmfit(Xtrain,Ytrain,'normal');
prediction4 = glmval(Beta4, Xtest, 'identity');
a = abs(Ytest-prediction4).^2;
MSE_normal = sum(a(:))/numel(Ytest);


