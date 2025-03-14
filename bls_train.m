function      [NetoutTest, Training_time,Testing_time,train_ERR,test_ERR] = bls_train(train_x,train_y,test_x,test_y,s,NumFea, NumWin, NumEnhan)

%%%%%%%%%%%%%%Training%%%%%%%%%%%%%%
%%%%%%%%%%%%%%feature nodes%%%%%%%%%%%%%%

train_x = zscore(train_x')'; 
Ns=NumFea*NumWin;
X1 = [train_x .1 * ones(size(train_x,1),1)];
feature_nodes=zeros(size(train_x,1),NumWin*NumFea);
for i=1:NumWin
    Wr=2*rand(size(train_x,2)+1,NumFea)-1;
    A1 = X1 * Wr;
    A1 = mapminmax(A1);
    clear W;
    Ws  =  sparse_bls(A1,X1,1e-3,50)';
    We{i}=Ws;
    F1 = X1 * Ws;
    fprintf(1,'Feature nodes in window %f: Max Val of Output %f Min Val %f\n',i,max(F1(:)),min(F1(:)));
    [F1,ps1]  =  mapminmax(F1',0,1);
    F1 = F1';
    ps(i)=ps1;
    feature_nodes(:,NumFea*(i-1)+1:NumFea*i)=F1;
end

clear X1;
clear F1;
%%%%%%%%%%%%%enhancement nodes%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X2 = [feature_nodes .1 * ones(size(feature_nodes,1),1)];
if Ns>=NumEnhan
    wh=orth(2*rand(Ns+1,NumEnhan)-1);
else
    wh=orth(2*rand(Ns+1,NumEnhan)'-1)'; 
end
enhancement_nodes = X2 *wh;
L2 = max(max(enhancement_nodes));
L2 = s/L2;
fprintf(1,'Enhancement nodes: Max Val of Output %f Min Val %f\n',L2,min(enhancement_nodes(:)));
enhancement_nodes = tansig(enhancement_nodes * L2); 
T3=[feature_nodes enhancement_nodes];
clear X2;clear enhancement_nodes;

tic
a=10^(-6);%初始化参数
b=10^(-6);    
k=size(T3,2);
c=repmat(a,k,1);
d=repmat(a,k,1);
mu_old=zeros(k,1);%初始化W均值与方差
VARw_old=eye(size(T3,2));
sig_old=abs(randn(k,1));%初始化sigma、eta和rho
eta_old=abs(randn(k,1));

for s=1:k
    if sig_old(s,1)==0
        sig_old(s,1)=10^-5;
    end
    if eta_old(s,1)==0
        eta_old(s,1)=10^-5;
    end
end
VAReta_old=ones(k,1);
rho_old=randn(1,1);
t=1;
L_old= norm(train_y-T3*mu_old,2)+(2/rho_old)*norm(diag(eta_old)*mu_old,1);
converged = false;
while  ~converged
    t=t+1;
    mu_new=rho_old*VARw_old*T3'*train_y;
    VARw_new=diag(sig_old)-diag(sig_old)*T3'*(eye(size(train_y,1))/((rho_old^(-1))*eye(size(T3,1))+T3*diag(sig_old)*T3'))*T3*diag(sig_old);
    for i=1:k
        sig_new(i,1)=sqrt(mu_new(i,1)^2+VARw_new(i,i))/sqrt(eta_old(i,1)^2+VAReta_old(i,1))+1/(eta_old(i,1)^2+VAReta_old(i,1));
        eta_new(i,1)=(c(i,1)+1)/(abs(mu_new(i,1))+d(i,1));  
        VAReta_new(i,1)=(c(i,1)+1)/((abs(mu_new(i,1))+d(i,1))^2);
    end
        rho_new=(a+size(train_y,1)/2)/(abs((train_y-T3*mu_new)'*(train_y-T3*mu_new))+trace(T3*VARw_new*T3'));
        L_new= norm(train_y-T3*mu_new,2)+(2/rho_new)*norm(diag(eta_new)*mu_new,1);
        kesai=(abs(L_new-L_old))/L_old;
        %kesai=norm(mu_new-mu_old,2);
    if kesai>0.01 && t<=30
        converged = false;
%         mu_old=mu_new;
        VARw_old=VARw_new;
        rho_old=rho_new;
        sig_old=sig_new;
        eta_old=eta_new;
        L_old=L_new;
    else
        converged = true;
        WeightTop = mu_new;
    end
end
 
Training_time = toc;
disp('Training has been finished!');
disp(['The Total Training Time is : ', num2str(Training_time), ' seconds' ]);
NetoutTrain = T3 * WeightTop;
clear T3;

RMSE =  sqrt(sum((NetoutTrain-train_y).^2)/size(train_y,1));
% MAPE = sum(abs(NetoutTrain-train_y))/mean(train_y)/size(train_y,1);
train_ERR = RMSE;
% train_MAPE = MAPE;
fprintf(1, 'TrainiNumWin RMSE is : %e\n', RMSE);

%%%%%%%%%%%%%%Testing%%%%%%%%%%%%%%
tic;

test_x = zscore(test_x')';
XX1 = [test_x .1 * ones(size(test_x,1),1)];
feature_nodes_test=zeros(size(test_x,1),Ns);
for i=1:NumWin
    Ws=We{i};ps1=ps(i);
    F2 = XX1 * Ws;
    F2  =  mapminmax('apply',F2',ps1)';
    clear Ws; clear ps1;
    feature_nodes_test(:,NumFea*(i-1)+1:NumFea*i)=F2;
end
clear F2;clear XX1;
XX2= [feature_nodes_test .1 * ones(size(feature_nodes_test,1),1)]; 
enhancement_nodes_test = tansig(XX2 * wh * L2);
TT3=[feature_nodes_test enhancement_nodes_test];
clear XX2;clear wh;clear enhancement_nodes_test;

NetoutTest = TT3 * WeightTop;

RMSE = sqrt(sum((NetoutTest-test_y).^2)/size(test_y,1));
%      MSE = sum((x-test_y).^2)/size(test_y,1);
% MAPE = sum(abs(NetoutTest-test_y))/mean(test_y)/size(test_y,1);

clear TT3;
test_ERR = RMSE;
% test_MAPE = MAPE;
%% Calculate the testing accuracy
Testing_time = toc;
disp('Testing has been finished!');
disp(['The Total Testing Time is : ', num2str(Testing_time), ' seconds' ]);
fprintf(1, 'TestiNumWin RMSE is : %e\n', RMSE);

