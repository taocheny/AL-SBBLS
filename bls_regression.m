clear;
warning off all;
format compact;
load ST.mat; 

assert(isfloat(train_x), 'train_x must be a float');
assert(isfloat(test_x), 'test_x must be a float');



%----C: the regularization parameter for sparse regualarization
s = .8;              %----s: the shrinkage parameter for enhancement nodes
best = 1;
result = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We apply the grid search on the test data set for instance and simplicity
% in this code, however, the reader can easily modify it to perform a grid
% search on validation set by replacing the test set with a validation set.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for NumFea= 1:20             %searching range for feature nodes  per window in feature layer
    for NumWin= 1:20          %searching range for number of windows in feature layer
        for NumEnhan= 2:2:200  %searching range for enhancement nodes
            clc;
            rand('state',1)
%             for i=1:NumWin
%                 WeightFea=2*rand(size(train_x,2)+1,NumFea)-1;
%                   b1=rand(size(train_x,2)+1,N1);  % sometimes use this may lead to better results, but not for sure!
%                 WF{i}=WeightFea;
%             end                                                          %generating weight and bias matrix for each window in feature layer
            %             if NumFea*NumWin>=NumEnh 
            %                 WeightEnhan=orth(2*rand(NumWin*NumFea+1,NumEnhan)-1);
            %             else
            %                 WeightEnhan=orth(2*rand(NumWin*NumFea+1,NumEnhan)'-1)';
            %             end
%             WeightEnhan=2*rand(NumWin*NumFea+1,NumEnhan)-1;
            %             WeightEnhan=rand(NumWin*NumFea+1,NumEnhan);    %You may choose one of the above initializing methods for weights connecting feature layer with enhancement layer
            fprintf(1, 'Fea. No.= %d, Win. No. =%d, Enhan. No. = %d\n', NumFea, NumWin, NumEnhan);
            [NetoutTest, Training_time,Testing_time,train_ERR,test_ERR] = bls_train(train_x,train_y,test_x,test_y,s,NumFea, NumWin, NumEnhan);
            time =Training_time + Testing_time;
            result = [result; NumFea NumWin NumEnhan test_ERR train_ERR]; % recording all the searching reaults
            if best > test_ERR
                best = test_ERR; 
                save('optimalST.mat','test_ERR', 'train_ERR','NumFea', 'NumWin', 'NumEnhan','time','Training_time','Testing_time',"NetoutTest");
            end
            clearvars -except best NumFea NumWin NumEnhan train_x train_y test_x test_y  C s result NetoutTest
        end
    end
    toc
end




 
