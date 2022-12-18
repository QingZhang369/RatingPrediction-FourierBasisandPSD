%%%%此函数计算一个顶点集合的核克莱姆矩阵
%M----m*n的矩阵,每行对应一个顶点
%type---类型
%    ---type==1:K(i,j)=exp(norm(vi-vj)^2/delta^2)
%gama---参数
%K----m*m的核克莱姆矩阵矩阵

function K=KernelGram(fvector,gama)
[row,col]=size(fvector);
ks1=repmat(fvector,1,1,row);
ks2=permute(ks1,[3,2,1]);     %ks2是什么意思？
temp=sum((ks2-ks1).^2,2);
clear ks1;
clear ks2;
temp=squeeze(temp);
K=exp(-temp/gama);

