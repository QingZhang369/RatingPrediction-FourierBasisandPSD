function psd=my_psd_estimate(G,s,Mask)
%%输入图G，评分表s(无评分的补0)相当于算法3.1中的Y，评分表的mask(元素为0或1)；
%%按照算法3.1，输出psd，列向量(分量是分别对频率\lambda0，...\lambdaN-1的psd)
[N,Ns]=size(s);
c=mean(s(Mask));
p=sum(Mask,2)/Ns;
S=cov(s');
S1=diag(S)./p-(1-p)*c^2;        %%Y的协方差矩阵的对角元
S2=(S-diag(diag(S)))./(p*p');   %%Y的协方差矩阵的非对角元

psd=sum((G.U.^2).*repmat(S1,1,N))'+...
          +sum((G.U'*S2).*G.U',2);
psd=abs(psd);
psd=smooth(psd,51);
% psd=smooth(psd,41);
% psd=smooth(psd,31);


