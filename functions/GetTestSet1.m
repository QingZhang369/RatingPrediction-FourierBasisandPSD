%%从评分表中随机抽取K个评分作为测试集
%UI-----初始评分表
%K-----测试集中评分个数
%testset---K*3的矩阵，每一行是一个用户对一个项目的打分，
%          第一列表示项目，第二列表示用户，第三列为对应打分
%rUI----被用作测试的打分不用于训练，在评分表中将对应项置0

function [testset,rUI]=GetTestSet1(UI,K)
testr=[];
testi=[];
[U,I]=size(UI);
[i,j,v]=find(UI);
ijv=[i,j,v];
k=numel(i);
if k>=K
    rUI=UI;
    rk=randperm(k);
    items=rk(1:K);
    testset=ijv(items,:);
    rUI(testset(:,1)+(testset(:,2)-1)*U)=0;  %把作为测试的在评分表中将对应项置0
else
    'K must be less than ratings;'
end
% for k=1:K
%     x=UI(u,:);
%     [temp,ix]=find(x);
%     iix=randperm(length(ix));
%     testr=[testr;x(ix(iix(1:k)))];
%     testi=[testi;ix(iix(1:k))];
%     UI(u,ix(iix(1:k)))=0;
% end