% ����UI���й�ϵ��ĳ�й�ϵ�����е�k������������ĳ���û���ĳ����Ŀ������
% UI----m*n�ľ���
% W----m*m�ľ��󣬱�ʾUI����֮����໥��ϵ
% dset---u*2�ľ���ÿ�б�ʾUI��Ԫ�ص�����λ��
% k-----��������ʾk����
% estr----u*1�ľ���dset��ÿ����ӦԪ�صĹ�������

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
mr=sum(UI,2)./sum(mask,2);          %mr������֪��ǩ��ƽ��(����֪��ǩ��/��֪��ǩ����)��������ƽ��(�к�/�����ܸ���)
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

% ԭEstimate.m������
% 1.UI:Un*In--����
% 2.W:Un*Un--��ʾUI������֮����໥��ϵ
% 3.dset:p*2--��ʾ��Ԥ��Ԫ�ص�����λ��
% 4.Wi=W(dset(:,1),:)--��ʾdset�����������еĹ�ϵ
% 5.rs=UI(:,dset(:,2))---��ʾUI�ж�Ӧ��Ԥ��Ԫ�������е�����Ԫ��

