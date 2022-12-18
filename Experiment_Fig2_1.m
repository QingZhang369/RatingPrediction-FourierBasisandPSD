%%dataset: movielens-100k-u1; netf: netfset{1}(item_first),netfset{6}(user_first)
%%learn parameter k, user_based
clear all;
close all;
addpath('functions','data');
load('data/u1base.mat');
u{1}.base=u1base;
clear u1base;
load('data/u1test.mat');
u{1}.test=u1test;
clear u1test;

load('netfset.mat');   

warning('off');

%%movielens-100k
Un=943;
In=1682;
implement=1;
%0: load results,
%1: run code, 
if implement
    ks=[5,20,30,50,100,150];
    m=numel(ks);
    param_k_ml=[];

  
      mae_us=[];
      rmse_us=[];

        ubase=u{1}.base;
        utest=u{1}.test;
        s=zeros(Un,In);
        s((ubase(:,2)-1)*Un+ubase(:,1))=ubase(:,3);
        
        [Un,In]=size(s);
        [su,si,mu,mi,mui,mask,rs,cs,ff]=preprocess(s);
        
        %%%%user-based%%%%%%%%%%     
        [urs,itms]=size(mask);
        su=CF(su,mask);
        S=cov(su');
        [U,psd]=gsp_FB_estimate(S); 
        psd=psd/max(psd);
        
        rr=500;
        beta=50;
        
        wf=@(x)exp(-x*rr); %1./(x.^rr+eps);%
        w=wf(psd);
        w=w/max(w);
        
    for i=1:numel(ks)
        k=ks(i);
        f = reconbyFB_impro(U,su,mask,beta,w,k);
        f=f+mu;
        ff(rs,cs)=f;
        idx=find(ff>5); %if a rating is larger than 5 then set it to 5
        ff(idx)=5;
        idx=find(ff<1); %if a rating is  less than 1 then set it to 1
        ff(idx)=1;
        
        err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3)); 
        mae_us=[mae_us;mean(err)];  
       
    end
        user_ml=mae_us;


%%%%%%%Netflix
          
   ks_itfirst=ks;
   ks_usfirst=ks_itfirst;


user_nef=[];  

  for r=1:5:10

      mae_us=[];
      rmse_us=[];
    
        s=netfset{r}.train;
        utest=netfset{r}.test;
        
        [Un,In]=size(s);
        [su,si,mu,mi,mui,mask,rs,cs,ff]=preprocess(s);
 
        [urs,itms]=size(mask);

        %%%%%%%%===user-based=================%%%%%%%%%%%%%
        su=CF(su,mask);
        S=cov(su');
        [U,psd]=gsp_FB_estimate(S);   
        psd=psd/max(psd);

        for i=1:m
            
        if r<6
            %%%%%%%item-first(r=1-5)%%%%%%%
            rr=500;
            beta=500;
            k=ks_itfirst(i);
            
        else
        %%%%%%%user-first(r=6-10)%%%%%%%
           rr=500;
           beta=300;
           k=ks_usfirst(i);
        end
         
        wf=@(x)exp(-x*rr);%1./(x.^rr+eps);%
        w=wf(psd);
        w=w/max(w);
        
        f = reconbyFB_impro(U,su,mask,beta,w,k);
        f=f+mu;
        ff(rs,cs)=f;
        idx=find(ff>5); %if a rating is larger than 5 then set it to 5
        ff(idx)=5;
        idx=find(ff<1); %if a rating is  less than 1 then set it to 1
        ff(idx)=1;
       
        err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
        mae_us=[mae_us;mean(err)];  
       
        end
        user_nef=[user_nef,mae_us];  
%         

 end        
            
param_k_us=[user_ml,user_nef];
    
save('data/param_k_us.mat','param_k_us');
else
    load('param_k_us.mat');
end  

%plot; 
figure(1)
plot(1:m,param_k_us(:,1),'ro-','LineWidth',2,'MarkerSize',8);
hold on
plot(1:m,param_k_us(:,3),'b*-','LineWidth',2,'MarkerSize',8);
plot(1:m,param_k_us(:,2),'k^-','LineWidth',2,'MarkerSize',8);
axis([1 m 0.68 0.76]);
legend({'ML-100k u1',...
    'Netflix-UF1',...
    'Netflix-IF1'},...
    'Location','northwest','NumColumns',1,'FontSize',12);
xlabel('k');
ylabel('MAE of user-based')
ax=gca;
xticks(1:m);
ax.XTickLabel = {'5','20','30','50','100','150'}; 
ax.FontName='Times New Roman';
ax.FontSize = 18;
