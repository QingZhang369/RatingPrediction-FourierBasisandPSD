%%三个子数据集学习参数分别代表三个大数据集，movielens-100k的u1和netf数据集的项目优先和用户优先各第一个,即 netfset{1},netfset{6}
%%参数k的学习, item_based
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
    rrs=[50,100,200,300,500,700];
    m=numel(rrs);

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

        for i=1:m
        rr=rrs(i);

        beta=50;
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
        mae_it=[mae_it;mean(err)];  
        end
        item_ml=mae_it;

%         %%%%%%%Netflix
       %%%%%%%%===item-based=================%%%%%%%%%%%%%          
  rrs_itfirst=rrs;
  rrs_usfirst=rrs_itfirst;

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
            beta=100;
            rr=rrs_itfirst(i);
            
        else
        %%%%%%%user-first(r=6-10)%%%%%%%
           beta=500;
           rr=rrs_usfirst(i);

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

        item_nef=[item_nef,mae_it];  %%输出的param_beta_it是m*2的矩阵，每一行是取同beta的对应5个数据集MAE，每一列对应相同数据集取不同beta的MAE
        
    end
  
 param_rr_it=[item_ml,item_nef];
    
save('data/param_rr_it.mat','param_rr_it');
else
    load('param_rr_it.mat');
end   
      

%画图，分别找到最适合各自的参数,fig1是mae；
figure(1)
plot(1:m,param_rr_it(:,1),'ro-','LineWidth',2,'MarkerSize',8);
hold on
plot(1:m,param_rr_it(:,3),'b*-','LineWidth',2,'MarkerSize',8);
plot(1:m,param_rr_it(:,2),'k^-','LineWidth',2,'MarkerSize',12);
axis([1 m 0.68 0.76]);
legend({'ML-100k u1',...
    'Netflix-UF1',...
    'Netflix-IF1'},...
    'Location','northwest','NumColumns',1,'FontSize',12);
xlabel('Weight function Parameter');
ylabel('MAE of item-based')
ax=gca;
xticks(1:m);
ax.XTickLabel = {'50','100','200','300','500','700'}; 
ax.FontName='Times New Roman';
ax.FontSize = 18;
