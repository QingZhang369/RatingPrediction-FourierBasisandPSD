%%%%此函数计算矩阵R的行与行之间的亲密度
%R----m*n的矩阵
%W----m*m的矩阵，W(i,j)表示UI的第i行与第j行的余弦距离

function W=Simxy(R,mask,type)
[U,I]=size(R);
singluarrows=find(sum(abs(R),2)==0);
if ~isempty(singluarrows)
    R(singluarrows,:)=repmat(sum(R)./(sum(mask)+eps),...
        numel(singluarrows),1);
end
if type==1
    temp=sqrt(sum(R.*R,2))+eps;
    temp1=R./repmat(temp,1,I);
    W=(temp1*temp1'+1)/2;
else
    W=corr(R');
    W=(W+1)/2;
end
W=W.*(eye(U)<1);


