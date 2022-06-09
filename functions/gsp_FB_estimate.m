function [V,psd,r]=gsp_FB_estimate(S)
%%Estimation for Fourier basis and PSD;
%%S is the covariance matrix of a GWSS stochastic graph signal x
%%Retun: V--Fourier basis,psd--
N=size(S,1);
U = dctmtx(N)';     %%Return discrete cosine transform matrix
B=U(:,2:end);
BSB=B'*S*B;
BSB=round(BSB*100000)/100000;  %%把那些很小的数直接置成0
%BSB=(BSB+BSB')/2;
[X,d] = eig(BSB);   %%X--eigenvectors,d--eigenvalues
d=diag(d);
idx=find(d<0);
d(idx)=-d(idx);
X(:,idx)=-X(:,idx);
[d,idx]=sort(d,'descend');
X=X(:,idx);
V=[U(:,1),B*X];
% psd=[sum(U(:,1).* sum(S.*repmat(U(:,1)',N,1),2));diag(d)];  %%U(:,1)'*S*U(:,1), sum(S.*repmat(U(:,1)',N,1),2))==S*U(:,1)
psd=[U(:,1)'*S*U(:,1);d];
r=norm(psd)/norm(S,'fro');