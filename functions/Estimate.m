%CF(Becker-2011)
% ����UI���й�ϵ��ĳ�й�ϵ�����е�k������������ĳ���û���ĳ����Ŀ������
% UI----m*n�ľ���
% W----m*m�ľ��󣬱�ʾUI����֮����໥��ϵ
% dset---u*2�ľ���ÿ�б�ʾUI��Ԫ�ص�����λ��
% k-----��������ʾk����
% estr----u*1�ľ���dset��ÿ����ӦԪ�صĹ�������

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

% ԭEstimate.m������
% 1.UI:Un*In--����
% 2.W:Un*Un--��ʾUI������֮����໥��ϵ
% 3.dset:p*2--��ʾ��Ԥ��Ԫ�ص�����λ��
% 4.Wi=W(dset(:,1),:)--��ʾdset�����������еĹ�ϵ
% 5.rs=UI(:,dset(:,2))---��ʾUI�ж�Ӧ��Ԥ��Ԫ�������е�����Ԫ��

