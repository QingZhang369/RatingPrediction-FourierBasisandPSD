% 根据UI的行关系和某列关系最密切的k个评分来估计某个用户对某个项目的评分
% UI----m*1的列向量，
% W----m*m的矩阵，表示UI的元素之间的相互关系
% dset---u*1的向量，表示UI中元素的位置
% k-----整数，表示k近邻
% estr----u*1的向量，dset中每个对应元素的估计评分

function estr=EstimateVector(UI,mask,W,dset,k)
m=size(UI,1);
Wi=W(dset,:)';   %%m*u,第j列表示dset中用户j与所有用户的关系
rs=repmat(UI,1,numel(dset));
rsu=repmat(mask,1,numel(dset));
Wi=Wi.*rsu;      %%与已知标签的权重
[sWi,idx]=sort(Wi,'descend');  %Wi每列降序排列
p=Wi>=repmat(sWi(k,:),m,1);    
Wi=Wi.*p;                      %每列中权重前k最大的留下，其余为零，即从已知标签中找出与dest相似度最大的k个
sW=sum(Wi);
idx=find(sW==0);
if ~isempty(idx)
    sW(idx)=1;
end
estr=(sum(Wi.*rs)./sW)';       %从该列的已知标签中找出跟第i个元素相似度最大的k个，以这k个的已知标签做加权平均

end