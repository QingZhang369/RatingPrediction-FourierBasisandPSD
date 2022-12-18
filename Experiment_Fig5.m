%%The influence of sparsity of dataset of prediction accuracy,compare our method,RKHS+kBR,RKHS+PSD, and CF
clear all;
close all;
addpath('functions','data');
load('ratings.mat');
implement=0;
%0: load results,
%1: run code,
if implement
    t0=tic;
    netfpct_us=[];
    netfpct_it=[];
    netfpct_us1=[];
    netfpct_it1=[];
    netfpct_us2=[];
    netfpct_it2=[];
    netfpct_us3=[];
    netfpct_it3=[];
    
    netfpctrmse_us=[];
    netfpctrmse_it=[];
    netfpctrmse_us1=[];
    netfpctrmse_it1=[];
    netfpctrmse_us2=[];
    netfpctrmse_it2=[];
    netfpctrmse_us3=[];
    netfpctrmse_it3=[];
    
    for r=1:10
        %     tic;
        if r<6  %%%%%%%item-first(r=1-5)%%%%%%%
            Un=1000;
            In=1777;
        else   %%%%%%%user-first(r=6-10)%%%%%%%
            Un=1500;
            In=888;
        end
        
        os=full(ratings{r});
        os=os';
        temp=os>0;
        lbs=sum(temp(:));   
        mae_us=[];
        mae_it=[];
        mae_us1=[];
        mae_it1=[];
        mae_us2=[];
        mae_it2=[];
        mae_us3=[];
        mae_it3=[];
        
        rmse_us=[];
        rmse_it=[];
        rmse_us1=[];
        rmse_it1=[];
        rmse_us2=[];
        rmse_it2=[];
        rmse_us3=[];
        rmse_it3=[];
       
        for pct=0.01:0.005:0.05              
            ntest=lbs-fix(Un*In*pct);         
            [utest,s]=GetTestSet1(os,ntest);  
            [su0,si0,mu,mi,mui,mask,rs,cs,ff]=preprocess(s);
            [urs,itms]=size(mask);

            %%%%%%%%===user-based=================%%%%%%%%%%%%%
             su=CF(su0,mask);
             S=cov(su');
             [U,psd]=gsp_FB_estimate(S);   
             psd=psd/max(psd);
            if r<6  %%%%%%%item-first(r=1-5)%%%%%%%
                rr1=500; 
                beta1=300;

                alpha=0.0001;
                beta=0.0025;
                gama=200;
                rr=0.2;
           
            else   %%%%%%%user-first(r=6-10)%%%%%%%
                rr1=500;
                beta1=300;

                alpha=0.0001;
                beta=0.001;
                gama=200;
                rr=0.2;
            end

            %% GFPSD  (Ours method)
            wf=@(x)exp(-x*rr1); 
            w=wf(psd);
            w=w/max(w);
        
            f = reconbyFB_impro(U,su,mask,beta1,w,20);
            f=f+mu;
            ff(rs,cs)=f;
            idx=find(ff>5); %if a rating is larger than 5 then set it to 5
            ff(idx)=5;
            idx=find(ff<1); %if a rating is  less than 1 then set it to 1
            ff(idx)=1;
            err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
            mae_us=[mae_us,mean(err)];
            rmse_us=[rmse_us,sqrt(mean(err.^2))];
            
 %%other methods
            [s0,~,~,~,~]=preprocess_GRPSD(s);
            [G,K,as,mr,mrc]=GKas(s0,mask,gama,true); 

            %% RKHS+kBR (Yang-2020) 
            f=KBreconstucter(G,as,mask,K,alpha,beta,10);  
            f=f+mr+mrc;
            ff(rs,cs)=f;
            idx=find(ff>5); %if a rating is larger than 5 then set it to 5
            ff(idx)=5;
            idx=find(ff<1); %if a rating is  less than 1 then set it to 1
            ff(idx)=1;
            err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
            mae_us1=[mae_us1,mean(err)];
            rmse_us1=[rmse_us1,sqrt(mean(err.^2))];

            %% RKHS+PSD  (Yang-2022)
            psd=my_psd_estimate(G,as,mask);
            psd=psd/max(psd);
            
            p=@(x) exp(-x/rr);
            wf=p(psd);
            wf=G.lmax*wf/max(wf);
            f=KBreconstucter(G,as,mask,K,alpha,beta,10,wf); 
            f=f+mr+mrc; 
            ff(rs,cs)=f;
            idx=find(ff>5); %if a rating is larger than 5 then set it to 5
            ff(idx)=5;
            idx=find(ff<1); %if a rating is  less than 1 then set it to 1
            ff(idx)=1;
            err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
            mae_us2=[mae_us2,mean(err)];
            rmse_us2=[rmse_us2,sqrt(mean(err.^2))];

            %% CF(Yang-2007) 
            [as,mU,mUI]=AdjustUI(s);
            mask1=(s>0);
            W=Simxy(as,mask1,1);
            estr1=Estimate(s,mask1,W,utest(:,1:2),35);
            err=abs(estr1-utest(:,3));
            mae_us3=[mae_us3,mean(err)];
            rmse_us3=[rmse_us3,sqrt(mean(err.^2))];
            
            %% %%%%%%===item-based=================%%%%%%%%%%%%%
             si=CF(si0',mask');
             S=cov(si');
             [U,psd]=gsp_FB_estimate(S);   
             psd=psd/max(psd);
            if r<6
            rr1=500;
            beta1=300;

                alpha=0.0001;
                beta=0.000025;
                gama=200;
                rr=0.5;
                %%%%%%%user-first(r=6-10)%%%%%%%
            else
            rr1=500;
            beta1=300;

                alpha=0.0001;
                beta=0.00025;
                gama=200;
                rr=0.2;
            end

        %% GFPSD  (Ours method)
        wf=@(x)exp(-x*rr1);
        w=wf(psd);
        w=w/max(w);
        f = reconbyFB_impro(U,si,mask',beta1,w,20);
        f=f'+mi;
        ff(rs,cs)=f;
        idx=find(ff>5); %if a rating is larger than 5 then set it to 5
        ff(idx)=5;
        idx=find(ff<1); %if a rating is  less than 1 then set it to 1
        ff(idx)=1;
        
        err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
        mae_it=[mae_it,mean(err)];
        rmse_it=[rmse_it,sqrt(mean(err.^2))];
        
 %%other methods
            [G,K,as,mr,mrc]=GKas(s0',mask',gama,true);
              
            %% RKHS+kBR (Yang-2020)
            f=KBreconstucter(G,as,mask',K,alpha,beta,10);
            f=f+mr+mrc;
            ff(rs,cs)=f';
            idx=find(ff>5); %if a rating is larger than 5 then set it to 5
            ff(idx)=5;
            idx=find(ff<1); %if a rating is  less than 1 then set it to 1
            ff(idx)=1;
            err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
            mae_it1=[mae_it1,mean(err)];
            rmse_it1=[rmse_it1,sqrt(mean(err.^2))];
            
            %% RKHS+PSD  (Yang-2022)
            psd=my_psd_estimate(G,as,mask');
            psd=psd/max(psd);
            p=@(x) exp(-rr*x);
            wf=p(psd);
            wf=G.lmax*wf/max(wf);
            
            f=KBreconstucter(G,as,mask',K,alpha,beta,10,wf);
            f=f+mr+mrc;
            ff(rs,cs)=f';
            idx=find(ff>5); %if a rating is larger than 5 then set it to 5
            ff(idx)=5;
            idx=find(ff<1); %if a rating is  less than 1 then set it to 1
            ff(idx)=1;
            err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
            mae_it2=[mae_it2,mean(err)];
            rmse_it2=[rmse_it2,sqrt(mean(err.^2))];
            
             %% CF
            [as,mU,mUI]=AdjustUI(s);
            mask1=(s>0);
            W=Simxy(as',mask1',1);
            estr2=Estimate(s',mask1',W,utest(:,2:-1:1),10);
            err=abs(estr2-utest(:,3));
            mae_it3=[mae_it3,mean(err)];
            rmse_it3=[rmse_it3,sqrt(mean(err.^2))];
            
            
        end
        %%mae
        netfpct_us=[netfpct_us;mae_us];
        netfpct_it=[netfpct_it;mae_it];
        netfpct_us1=[netfpct_us1;mae_us1];
        netfpct_it1=[netfpct_it1;mae_it1];
        netfpct_us2=[netfpct_us2;mae_us2];
        netfpct_it2=[netfpct_it2;mae_it2];
        netfpct_us3=[netfpct_us3;mae_us3];
        netfpct_it3=[netfpct_it3;mae_it3]; 
        
        netfpctrmse_us=[netfpctrmse_us;rmse_us];
        netfpctrmse_it=[netfpctrmse_it;rmse_it];
        netfpctrmse_us1=[netfpctrmse_us1;rmse_us1];
        netfpctrmse_it1=[netfpctrmse_it1;rmse_it1];
        netfpctrmse_us2=[netfpctrmse_us2;rmse_us2];
        netfpctrmse_it2=[netfpctrmse_it2;rmse_it2];
        netfpctrmse_us3=[netfpctrmse_us3;rmse_us3];
        netfpctrmse_it3=[netfpctrmse_it3;rmse_it3];

    end
    
    save('data/netfpct_us.mat','netfpct_us');
    save('data/netfpct_it.mat','netfpct_it');
    save('data/netfpct_us1.mat','netfpct_us1');
    save('data/netfpct_it1.mat','netfpct_it1');
    save('data/netfpct_us2.mat','netfpct_us2');
    save('data/netfpct_it2.mat','netfpct_it2');    
    save('data/netfpct_us3.mat','netfpct_us3');
    save('data/netfpct_it3.mat','netfpct_it3'); 
    
    save('data/netfpctrmse_us.mat','netfpctrmse_us');
    save('data/netfpctrmse_it.mat','netfpctrmse_it');
    save('data/netfpctrmse_us1.mat','netfpctrmse_us1');
    save('data/netfpctrmse_it1.mat','netfpctrmse_it1');
    save('data/netfpctrmse_us2.mat','netfpctrmse_us2');
    save('data/netfpctrmse_it2.mat','netfpctrmse_it2');
    save('data/netfpctrmse_us3.mat','netfpctrmse_us3');
    save('data/netfpctrmse_it3.mat','netfpctrmse_it3');
else
    load('netfpct_us.mat');
    load('netfpct_it.mat');
    load('netfpct_us1.mat');
    load('netfpct_it1.mat');
    load('netfpct_us2.mat');
    load('netfpct_it2.mat');
    load('netfpct_us3.mat');
    load('netfpct_it3.mat');
    
    load('netfpctrmse_us.mat');
    load('netfpctrmse_it.mat');
    load('netfpctrmse_us1.mat');
    load('netfpctrmse_it1.mat');
    load('netfpctrmse_us2.mat');
    load('netfpctrmse_it2.mat');
    load('netfpctrmse_us3.mat');
    load('netfpctrmse_it3.mat');

end

%plot:MAE(user-first,item-first)

figure(1)
plot(1:9,mean(netfpct_us3(6:10,:)),'o-','LineWidth',2,'MarkerSize',8);
hold on
plot(1:9,mean(netfpct_us1(6:10,:)),'*-','LineWidth',2,'MarkerSize',8);
plot(1:9,mean(netfpct_us2(6:10,:)),'^-','LineWidth',2,'MarkerSize',8);
plot(1:9,mean(netfpct_us(6:10,:)),'^-','LineWidth',2,'MarkerSize',8);
axis([1 9 0.68 0.88]);
legend({'CF',...
    'RKHS+kBR',...
    'RKHS+PSD',...
    'Ours'},...
    'Location','northeast','NumColumns',1,'Fontsize',12);
xlabel('Known entries percentage (%)');
ylabel('MAE of UB+UF')
ax=gca;
xticks(1:9);
ax.XTickLabel = {'1','1.5','2','2.5','3','3.5','4','4.5','5'}; 
ax.FontName='Times New Roman';
ax.FontSize = 18;

figure(2)
plot(1:9,mean(netfpct_us3(1:5,:)),'o-','LineWidth',2,'MarkerSize',8);
hold on
plot(1:9,mean(netfpct_us1(1:5,:)),'*-','LineWidth',2,'MarkerSize',8);
plot(1:9,mean(netfpct_us2(1:5,:)),'^-','LineWidth',2,'MarkerSize',8);
plot(1:9,mean(netfpct_us(1:5,:)),'^-','LineWidth',2,'MarkerSize',8);
axis([1 9 0.68 0.88]);
legend({'CF',...
    'RKHS+kBR',...
    'RKHS+PSD',...
    'Ours'},...
    'Location','northeast','NumColumns',1,'Fontsize',12);
xlabel('Known entries percentage (%)');
ylabel('MAE of UB+IF')
ax=gca;
xticks(1:9);
ax.XTickLabel = {'1','1.5','2','2.5','3','3.5','4','4.5','5'}; 
ax.FontName='Times New Roman';
ax.FontSize = 18;

figure(3)
plot(1:9,mean(netfpct_it3(6:10,:)),'o-','LineWidth',2,'MarkerSize',8);
hold on
plot(1:9,mean(netfpct_it1(6:10,:)),'*-','LineWidth',2,'MarkerSize',8);
plot(1:9,mean(netfpct_it2(6:10,:)),'^-','LineWidth',2,'MarkerSize',8);
plot(1:9,mean(netfpct_us(6:10,:)),'^-','LineWidth',2,'MarkerSize',8);
axis([1 9 0.68 0.88]);
legend({'CF',...
    'RKHS+kBR',...
    'RKHS+PSD',...
    'Ours'},...
    'Location','northeast','NumColumns',1,'Fontsize',12);
xlabel('Known entries percentage (%)');
ylabel('MAE of IB+UF')
ax=gca;
xticks(1:9);
ax.XTickLabel = {'1','1.5','2','2.5','3','3.5','4','4.5','5'}; 
ax.FontName='Times New Roman';
ax.FontSize = 18;

figure(4)
plot(1:9,mean(netfpct_it3(1:5,:)),'o-','LineWidth',2,'MarkerSize',8);
hold on
plot(1:9,mean(netfpct_it1(1:5,:)),'*-','LineWidth',2,'MarkerSize',8);
plot(1:9,mean(netfpct_it2(1:5,:)),'^-','LineWidth',2,'MarkerSize',8);
plot(1:9,mean(netfpct_us(1:5,:)),'^-','LineWidth',2,'MarkerSize',8);
axis([1 9 0.68 0.88]);
legend({'CF',...
    'RKHS+kBR',...
    'RKHS+PSD',...
    'Ours'},...
    'Location','northeast','NumColumns',1,'Fontsize',12);
xlabel('Known entries percentage (%)');
ylabel('MAE of IB+IF')
ax=gca;
xticks(1:9);
ax.XTickLabel = {'1','1.5','2','2.5','3','3.5','4','4.5','5'};  
ax.FontName='Times New Roman';
ax.FontSize = 18;

%RMSE:user-first,item-first

figure(5)
plot(1:9,mean(netfpctrmse_us3(6:10,:)),'o-','LineWidth',2,'MarkerSize',8);
hold on
plot(1:9,mean(netfpctrmse_us1(6:10,:)),'*-','LineWidth',2,'MarkerSize',8);
plot(1:9,mean(netfpctrmse_us2(6:10,:)),'^-','LineWidth',2,'MarkerSize',8);
plot(1:9,mean(netfpctrmse_us(6:10,:)),'^-','LineWidth',2,'MarkerSize',8);
axis([1 9 0.88 1.08]);
legend({'CF',...
    'RKHS+kBR',...
    'RKHS+PSD',...
    'Ours'},...
    'Location','northwest','NumColumns',1,'Fontsize',12);
xlabel('Known entries percentage (%)');
ylabel('RMSE of UB+UF')
ax=gca;
xticks(1:9);
ax.XTickLabel = {'1','1.5','2','2.5','3','3.5','4','4.5','5'}; 
ax.FontName='Times New Roman';
ax.FontSize = 18;

figure(6)
plot(1:9,mean(netfpctrmse_us3(1:5,:)),'o-','LineWidth',2,'MarkerSize',8);
hold on
plot(1:9,mean(netfpctrmse_us1(1:5,:)),'*-','LineWidth',2,'MarkerSize',8);
plot(1:9,mean(netfpctrmse_us2(1:5,:)),'^-','LineWidth',2,'MarkerSize',8);
plot(1:9,mean(netfpctrmse_us(1:5,:)),'^-','LineWidth',2,'MarkerSize',8);
axis([1 9 0.88 1.08]);
legend({'CF',...
    'RKHS+kBR',...
    'RKHS+PSD',...
    'Ours'},...
    'Location','northeast','NumColumns',1,'Fontsize',12);
xlabel('Known entries percentage (%)');
ylabel('RMSE of UB+IF')
ax=gca;
xticks(1:9);
ax.XTickLabel = {'1','1.5','2','2.5','3','3.5','4','4.5','5'}; 
ax.FontName='Times New Roman';
ax.FontSize = 18;

figure(7)
plot(1:9,mean(netfpctrmse_it3(6:10,:)),'o-','LineWidth',2,'MarkerSize',8);
hold on
plot(1:9,mean(netfpctrmse_it1(6:10,:)),'*-','LineWidth',2,'MarkerSize',8);
plot(1:9,mean(netfpctrmse_it2(6:10,:)),'^-','LineWidth',2,'MarkerSize',8);
plot(1:9,mean(netfpctrmse_it(6:10,:)),'^-','LineWidth',2,'MarkerSize',8);
axis([1 9 0.88 1.08]);
legend({'CF',...
    'RKHS+kBR',...
    'RKHS+PSD',...
    'Ours'},...
    'Location','northeast','NumColumns',1,'Fontsize',12);
xlabel('Known entries percentage (%)');
ylabel('RMSE of IB+UF')
ax=gca;
xticks(1:9);
ax.XTickLabel = {'1','1.5','2','2.5','3','3.5','4','4.5','5'}; 
ax.FontName='Times New Roman';
ax.FontSize = 18;

figure(8)
plot(1:9,mean(netfpctrmse_it3(1:5,:)),'o-','LineWidth',2,'MarkerSize',8);
hold on
plot(1:9,mean(netfpctrmse_it1(1:5,:)),'*-','LineWidth',2,'MarkerSize',8);
plot(1:9,mean(netfpctrmse_it2(1:5,:)),'^-','LineWidth',2,'MarkerSize',8);
plot(1:9,mean(netfpctrmse_it(1:5,:)),'^-','LineWidth',2,'MarkerSize',8);
axis([1 9 0.88 1.08]);
legend({'CF',...
    'RKHS+kBR',...
    'RKHS+PSD',...
    'Ours'},...
    'Location','northeast','NumColumns',1,'Fontsize',12);
xlabel('Known entries percentage (%)');
ylabel('RMSE of IB+IF')
ax=gca;
xticks(1:9);
ax.XTickLabel = {'1','1.5','2','2.5','3','3.5','4','4.5','5'};  
ax.FontName='Times New Roman';
ax.FontSize = 18;

