%%---This program is designed to reconstruct signals
%%---G: a graph
%%---s: A matrix whose eachcolumn is a sampled signal
%%---Mask:A matrix the same size as s. A element is 1 
%%--       means known label, and 0 means unknown label
%%---beta: the regularization parameter
%%---w: the weight function, if default, reconstruct signal by GM
%%---ff: the reconstructed signals which is s matrix the same size as s


function ff = reconstructer(G,s,Mask,beta,w)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
%if the weight function default,set w=G.e
if nargin<5 
    w=[];
end

E=eye(G.N);
[N,Ns]=size(Mask);
idx=find(Mask);
mc=sum(s(idx))/numel(idx); %calculate the mean
ff=[];
%--the maximum value of the weight function is 
%--standardized to $\lambda_{max}$
ws=numel(find(G.W));
if isempty(w)
    KL=G.L/sum(G.e);
else
    w=w/sum(w);
    KL=G.U*(repmat(w,1,G.N).*G.U');
end
for j=1:Ns
    labels=find(Mask(:,j));
    if isempty(labels)
        f=mc*ones(1,N);
    else
        lbs=numel(labels);
        yL=s(labels,j);
        Nlabels=1:N;
        Nlabels(labels)=[];
        p=[labels;Nlabels'];
        LL=KL(p,p);
        El=E(1:lbs,:);
        V=El'*El/lbs+beta*LL;%/ws;
        f(p)=inv(V)*El'*yL/lbs;
    end
    ff=[ff,f'];
end
end



