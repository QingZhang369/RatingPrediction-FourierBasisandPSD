%%%%调整评分表
%UI---初始评分表
%aUI----调整后的评分表
%mUI-----在初始评分表中，所有已打分项目的平均分
%mU----在初始评分表中，每个用户的所有打分的平均分

function [aUI,mU,mUI]=AdjustUI(UI)
[U,I]=size(UI);
nonEl=UI~=0;   %(每个元素是0或1，非零返回1，sum(nonEl(:))表示非零的个数)
mUI=sum(UI(:))/sum(nonEl(:));  
mU=sum((UI-mUI).*nonEl,2)./(sum(nonEl,2)+eps);
aUI=(UI-repmat(mU+mUI,1,I)).*nonEl;