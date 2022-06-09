function ff=KBreconstucter(G,s,mask,K,alpha,beta,k,wf)
%%k-bandlimited approximate
%对评分s每一列进行重构，ff是重构后的矩阵，把s的未知量预测填充后的矩阵
%此函数适用于s无全行/列列为零的情况，故预处理把s全行/列为零的行/列删去
if nargin<8
    wf=[];
end
ws=numel(find(G.W));
[m,n]=size(mask);
if isempty(wf)
    LL=G.L;
else
    LL=G.U*(repmat(wf,1,G.N).*G.U');
end
klk=alpha*K+beta*K*LL*K;
Uk=G.U(:,1:k);
ff=[];
for j=1:n
    labels=find(mask(:,j));
    if ~isempty(labels)
        yL=s(labels,j);
        KL=K(labels,:);
        a=inv(Uk'*(KL'*KL+klk)*Uk)*(Uk'*KL'*yL); 
        f=K*Uk*a;
    end
    ff=[ff,f];
end