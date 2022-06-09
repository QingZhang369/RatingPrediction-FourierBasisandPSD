function [G,K,as,mr,mrc]=GKas(s,mask,gama,adjust)
if adjust
    [as,mr,mrc]=AdjustUI(s);
else
    as=s;
end
Wi=Simxy(as,mask,1);
fvector=Feavec(s,mask);
K=KernelGram(fvector,gama);

[w,c]=extremsimilarity(s,mask,3,1,1);
G.W=Wi.*w.*c;
G.d=sum(G.W,2);
G.L=diag(G.d)-G.W;
G.N=size(G.L,1);
[U,V]=eig(G.L);
G.U=U;
G.e=diag(V);
G.lmax=G.e(end);