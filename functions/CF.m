% 根据UI的行关系和某列关系最密切的k个评分来估计某个用户对某个项目的评分
% UI----m*n的矩阵
% W----m*m的矩阵，表示UI的行之间的相互关系
% dset---u*2的矩阵，每行表示UI中元素的行列位置
% k-----整数，表示k近邻
% estr----u*1的矩阵，dset中每个对应元素的估计评分

function R=CF(UI,mask)
[r,c]=size(mask);
% R=UI;
% for j=1:c
%     idx=find(mask(:,j)==0);  
%     n=numel(idx);
%     Wi=W(idx,:);
%     maskj=repmat(mask(:,j)',n,1);
%     Wi=Wi.*maskj;
%     [Wi,idx0]=sort(Wi,2,'descend');
%     idx0=idx0(:,1:k);
%     idx0=idx0';
%     Wi=Wi(:,1:k);
%     rr=UI(idx0(:),j);
%     rr=reshape(rr,k,n)'; 
%     R(idx,j)=(sum(Wi(:,1:k).*rr,2)./sum(Wi(:,1:k),2))';
% end
R1=UI;
R2=UI;
mr=sum(UI,2)./sum(mask,2);          %mr是行已知标签的平均(行已知标签和/已知标签个数)，不是行平均(行和/本行总个数)
for j=1:c
    idx=find(mask(:,j)==0);  
    R1(idx,j)=mr(idx);
end
mr=sum(UI)./sum(mask);
for i=1:r
    idx=find(mask(i,:)==0);  
    R2(i,idx)=mr(idx);
end
 R=(R1+R2)/2;

% [U,I]=size(UI);
% Wi=W(dset(:,1),:)';
% rs=UI(:,dset(:,2));
% rsu=mask(:,dset(:,2));
% Wi=Wi.*rsu;
% [sWi,idx]=sort(Wi,'descend');
% p=Wi>=repmat(sWi(k,:),U,1);
% Wi=Wi.*p;
% sW=sum(Wi);
% idx=find(sW==0);
% if ~isempty(idx)
%     sW(idx)=1;
% end
% estr=(sum(Wi.*rs)./sW)';
% if ~isempty(idx)
%     rjs=sum(mask(dset(idx,1),:),2);
%     idx1=find(rjs);
%     if  ~isempty(idx1)
%         estr(idx(idx1))=sum(UI(dset(idx(idx1),1),:),2)./sum(mask(dset(idx(idx1),1),:),2);
%         %sum(UI(:)/sum(mask(:)));
%     end
%     idx1=find(rjs==0);
%     if  ~isempty(idx1)
%         estr(idx(idx1))=sum(UI(:))./sum(mask(:));
%         %sum(UI(:)/sum(mask(:)));
%     end
% end

% 原Estimate.m函数中
% 1.UI:Un*In--矩阵
% 2.W:Un*Un--表示UI行与行之间的相互关系
% 3.dset:p*2--表示待预测元素的行列位置
% 4.Wi=W(dset(:,1),:)--表示dset中行与所有行的关系
% 5.rs=UI(:,dset(:,2))---表示UI中对应待预测元素所在列的整列元素

