%CF(Becker-2011)
% 根据UI的行关系和某列关系最密切的k个评分来估计某个用户对某个项目的评分
% UI----m*n的矩阵
% W----m*m的矩阵，表示UI的行之间的相互关系
% dset---u*2的矩阵，每行表示UI中元素的行列位置
% k-----整数，表示k近邻
% estr----u*1的矩阵，dset中每个对应元素的估计评分

function estr=Estimate(UI,mask,W,dset,k)
[U,I]=size(UI);
Wi=W(dset(:,1),:)';
rs=UI(:,dset(:,2));
rsu=mask(:,dset(:,2));
Wi=Wi.*rsu;
[sWi,idx]=sort(Wi,'descend');
p=Wi>=repmat(sWi(k,:),U,1);
Wi=Wi.*p;
sW=sum(Wi);
idx=find(sW==0);
if ~isempty(idx)
    sW(idx)=1;
end
estr=(sum(Wi.*rs)./sW)';
if ~isempty(idx)
    rjs=sum(mask(dset(idx,1),:),2);
    idx1=find(rjs);
    if  ~isempty(idx1)
        estr(idx(idx1))=sum(UI(dset(idx(idx1),1),:),2)./sum(mask(dset(idx(idx1),1),:),2);
        %sum(UI(:)/sum(mask(:)));
    end
    idx1=find(rjs==0);
    if  ~isempty(idx1)
        estr(idx(idx1))=sum(UI(:))./sum(mask(:));
        %sum(UI(:)/sum(mask(:)));
    end
end

% 原Estimate.m函数中
% 1.UI:Un*In--矩阵
% 2.W:Un*Un--表示UI行与行之间的相互关系
% 3.dset:p*2--表示待预测元素的行列位置
% 4.Wi=W(dset(:,1),:)--表示dset中行与所有行的关系
% 5.rs=UI(:,dset(:,2))---表示UI中对应待预测元素所在列的整列元素

