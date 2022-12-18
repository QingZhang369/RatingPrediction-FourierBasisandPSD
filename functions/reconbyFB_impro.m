%%---This program is designed to reconstruct signals
%%---G: a graph
%%---s: A matrix whose eachcolumn is a sampled signal
%%---Mask:A matrix the same size as s. A element is 1 
%%--       means known label, and 0 means unknown label
%%---beta: the regularization parameter
%%---w: the weight function, if default, reconstruct signal by GM
%%---ff: the reconstructed signals which is s matrix the same size as s

function ff = reconbyFB_impro(U,s,mask,beta,w,k)
%UNTITLED2 此处显示有关此函数的摘要
[N,Ns]=size(s);
% E=eye(N);
idx=find(mask);
mc=sum(s(idx))/numel(idx); %calculate the mean
ff=[];

Uk=U(:,1:k);
%KL=Uk'*U*(repmat(w,1,N).*U')*Uk;
KL=diag(w(1:k));
for j=1:Ns
    labels=find(mask(:,j));
    if isempty(labels)
        f=mc*ones(N,1);
    else
        xs=s(labels,j);
        V=Uk(labels,:)'*Uk(labels,:)+beta*KL;
        f0=V\(Uk(labels,:)'*xs);
        f=Uk*f0;
    end
    ff=[ff,f];
end
end


