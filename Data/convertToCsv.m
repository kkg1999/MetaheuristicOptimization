function [] = convertToCsv(fileName)
    data=importdata(strcat(fileName,'/',fileName,'_data.mat'));
    train=data.train;
    trainLabel=data.trainLabel;
    test=data.test;
    testLabel=data.testLabel;
    numTrain=size(train,1);
    numTest=size(test,1);
    numFeatures=size(train,2);
    mat(1:numTrain,:)=train;
    mat(numTrain+1:numTrain+numTest,:)=test;
    for loop1=1:numTrain
        mat(loop1,numFeatures+1)=int16(find(trainLabel(loop1,:)==1));
    end
    for loop2=numTrain+1:numTrain+numTest
        mat(loop2,numFeatures+1)=int16(find(testLabel(loop2-numTrain,:)==1));
    end
    csvwrite(strcat(fileName,'/',fileName,'.csv'),mat);
end