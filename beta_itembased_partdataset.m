%%三个子数据集学习参数分别代表三个大数据集，movielens-100k的u1和netf数据集的项目优先和用户优先各第一个,即 netfset{1},netfset{6}
%%参数k的学习, user_based
clear all;
close all;
addpath('functions','data');
load('data/u1base.mat');
u{1}.base=u1base;
clear u1base;
load('data/u1test.mat');
u{1}.test=u1test;
clear u1test;

load('netfset.mat');   %%此数据集1:5是item-first，6:10是user-first

warning('off');

Un=943;
In=1682;
implement=1;
%0: load results,
%1: run code, running time is about 620s depending on your computer
if implement
    %%movielens-100k
    betas=[10,25,50,100,300,500];
    m=numel(betas);
    param_beta_ml=[];

       mae_it=[];
       rmse_it=[];
        ubase=u{1}.base;
        utest=u{1}.test;
        s=zeros(Un,In);
        s((ubase(:,2)-1)*Un+ubase(:,1))=ubase(:,3);
        
        [Un,In]=size(s);
        [su,si,mu,mi,mui,mask,rs,cs,ff]=preprocess(s);

        %%%%item-based%%%%%%%%%%
        [urs,itms]=size(mask);
        si=CF(si',mask');
        S=cov(si');
        [U,psd]=gsp_FB_estimate(S);   %%U锛?Un*Un
        psd=psd/max(psd);
        
        rr=500;
%       beta=50;
        wf=@(x)exp(-x*rr);%1./(x.^rr+eps);%
        w=wf(psd);
        w=w/max(w);
      
        for i=1:numel(betas)
        beta=betas(i);
        f = reconbyFB_impro(U,si,mask',beta,w,50);
        f=f'+mi;
        ff(rs,cs)=f;
        idx=find(ff>5); %if a rating is larger than 5 then set it to 5
        ff(idx)=5;
        idx=find(ff<1); %if a rating is  less than 1 then set it to 1
        ff(idx)=1;
        err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
        mae_it=[mae_it;mean(err)];
        end
       
      item_ml= mae_it;

%         %%%%%%%Netflix
       %%%%%%%%===item-based=================%%%%%%%%%%%%%

          
   betas_itfirst=betas;
   betas_usfirst=betas_itfirst;


   item_nef=[];  

  for r=1:5:10

      mae_it=[];
      rmse_it=[];
    
        s=netfset{r}.train;
        utest=netfset{r}.test;
        
        [Un,In]=size(s);
        [su,si,mu,mi,mui,mask,rs,cs,ff]=preprocess(s);
                
        [urs,itms]=size(mask);
        si=CF(si',mask');
        S=cov(si');
        [U,psd]=gsp_FB_estimate(S);   %%U锛?Un*Un
        psd=psd/max(psd);
        for i=1:m
        if r<6
            %%%%%%%item-first(r=1-5)%%%%%%%
            rr=500;
            beta=betas_itfirst(i);
            
        else
        %%%%%%%user-first(r=6-10)%%%%%%%
           rr=500;
            beta=betas_usfirst(i);

        end
        wf=@(x)exp(-x*rr);%1./(x.^rr+eps);%
        w=wf(psd);
        w=w/max(w);
        f = reconbyFB_impro(U,si,mask',beta,w,50);
        f=f'+mi;
        ff(rs,cs)=f;
        idx=find(ff>5); %if a rating is larger than 5 then set it to 5
        ff(idx)=5;
        idx=find(ff<1); %if a rating is  less than 1 then set it to 1
        ff(idx)=1;
       
        err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
        mae_it=[mae_it;mean(err)];    %%相同数据集的取不同beta的结果按列存放
        end

        item_nef=[item_nef,mae_it];  %%输出的param_beta_it是m*5的矩阵，每一行是取同beta的对应5个数据集MAE，每一列对应相同数据集取不同beta的MAE

  end
      %user_first=param_beta_it(:,2);       %%执行item-based时
      %item_first=param_beta_it(:,1);

param_beta_it=[item_ml,item_nef];
    
save('data/param_beta_it.mat','param_beta_it');
else
    load('param_beta_it.mat');
end     
      
%画图，分别找到最适合各自的参数
figure(1)
plot(1:m,param_beta_it(:,1),'ro-','LineWidth',2,'MarkerSize',8);
hold on
plot(1:m,param_beta_it(:,3),'b*-','LineWidth',2,'MarkerSize',8);
plot(1:m,param_beta_it(:,2),'k^-','LineWidth',2,'MarkerSize',8);
axis([1 m 0.68 0.76]);
legend({'ML-100k u1',...
    'Netflix-UF1',...
    'Netflix-IF1'},...
    'Location','northwest','NumColumns',1,'FontSize',12);
xlabel('\beta');
ylabel('MAE of item-based')
ax=gca;
xticks(1:m);
ax.XTickLabel = {'10','25','50','100','300','500'}; 
ax.FontName='Times New Roman';
ax.FontSize = 18;


