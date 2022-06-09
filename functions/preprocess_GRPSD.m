function [s,mask,rs,cs,ff]=preprocess(s)
%%Return:s是原s去掉行和为零的行和列和为零的列
%%rs是原s去掉行和为零的行；
%%cs是原s去掉列和为零的列；
%%ff是矩阵，对原s中行和为零或零和为零的位置的处理，其余位置为零；
[Un,In]=size(s);
mask=(s>0);
% mask1=mask;
rs=1:Un;
cs=1:In;
% kr1=find(sum(mask,2)==0);
% kc1=find(sum(mask)==0);
% kr2=find(sum(mask,2)>1 & sum(mask,2)<3);
% kc2=find(sum(mask)>1 &sum(mask)<3);
% mI=sum(s.*mask)./sum(mask);
% mU=sum(s.*mask,2)./sum(mask,2);
% mUI=sum(sum(s.*mask))/sum(sum(mask));
% 
% ff=zeros(Un,In);
% if ~isempty(kr1)
%     ff(kr1,cs)=ff(kr1,cs)+...
%         repmat(mI,numel(kr1),1);
% end
% if ~isempty(kc1)
%     ff(rs,kc1)=ff(rs,kc1)+...
%         repmat(mU,1,numel(kc1));
% end
% if ~isempty(kr1) & ~isempty(kc1)
%     ff(kr,kc1)=mUI;
% end
% if ~isempty(kr2)
%     mm=sum(s(kr2,:).*mask(kr2,:),2)./sum(mask(kr2,:),2);
%     ff(kr2,cs)=ff(kr2,cs)+...
%         repmat(mm,1,In);
%     for i=1:numel(kr2)
%         cc=find(mask(kr2(i),:));
%         ff(kr2(i),cc)=s(kr2(i),cc);
%     end
% end 
% if ~isempty(kc2)
%     mm=sum(s(:,kc2).*mask(:,kc2))./sum(mask(:,kc2));
%     ff(rs,kc2)=ff(rs,kc2)+...
%         repmat(mm,Un,1);
%     for i=1:numel(kc2)
%         rr=find(mask(:,kc2(i)));
%         ff(rr,kc2(i))=s(rr,kc2(i));
%     end
% end 
% 
% kr=[kr1;kr2];
% kc=[kc1,kc2];
% rs(kr)=[];
% cs(kc)=[];
% s=s(rs,:);
% s=s(:,cs);
% mask=mask(rs,:);
% mask=mask(:,cs);


%%%%%%%%%%%%%%%%%%%%%%%%
kr=find(sum(mask,2)==0);     %%寻找矩阵mask的行和为零的行
kc=find(sum(mask)==0);       %%寻找矩阵mask的列和为零的列
rs(kr)=[];
cs(kc)=[];
s=s(rs,:);
s=s(:,cs);
mask=mask(rs,:);
mask=mask(:,cs);

mI=sum(s.*mask)./sum(mask);           %%sum(s.*mask)是行向量，是矩阵的列和，mI是行向量，每个元素为该列的平均
mU=sum(s.*mask,2)./sum(mask,2);       %%sum(s.*mask，2)是列向量，是矩阵的行和，mU是列向量，每个元素是该行的平均
mUI=sum(sum(s.*mask))/sum(sum(mask)); %%sum(sum(s.*mask))是矩阵的所有元素和，mUI是所有元素的平均
ff=zeros(Un,In);
if ~isempty(kr)
    ff(kr,cs)=ff(kr,cs)+...
        repmat(mI,numel(kr),1);    %%1.行和为零的行，元素全部补列平均
end
if ~isempty(kc)
    ff(rs,kc)=ff(rs,kc)+...
        repmat(mU,1,numel(kc));    %%2.列和为零的列，元素全部补行平均
end
if ~isempty(kr) & ~isempty(kc)     %%3.行和为零且列和为零时，交接位置补所有元素的平均，其他的按照1,2赋列/行平均
    ff(kr,kc)=mUI;
end