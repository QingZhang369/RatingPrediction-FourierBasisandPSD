%%%%此函数计算一个顶点集合的核克莱姆矩阵
%M----m*n的矩阵,每行对应一个顶点
%mask---m*n的0,1矩阵,标识M中有效数据位置，1对应有效数据，(mask=M~=0)
%fvector---m*8的矩阵，每行对应一个顶点的特征向量
function fvector=Feavec(M,mask)
[row,col]=size(M);
fvector=[];
for i=1:row
    mask1=mask.*repmat(mask(i,:),row,1);
    M1=M.*mask1;     %Q:M1是什么？
    rr=sum(M1);
    nr=sum(mask1);
    if nr==0
        fvector=[fvector;[0,0]];
    else
        idx=find(nr); 
        rr=rr(idx);
        nr=nr(idx);
        mr=rr./nr;
        br=M(i,idx)-mr;
        fvector=[fvector;[mean(br),std(br,1)]];%,skewness(br),kurtosis(br)-3
    end
end
temp=[sum(M==ones(size(M)),2),sum(M==2*ones(size(M)),2),...   
    sum(M==3*ones(size(M)),2),sum(M==4*ones(size(M)),2),...
    sum(M==5*ones(size(M)),2)]./repmat(sum(mask,2)+eps,1,5);
temp=[temp,sum(mask,2)/max(sum(mask,2))];
fvector=[fvector,temp];