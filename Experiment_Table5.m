%%Netflix dataset,compare the proposed method, RKHS+kBR, and RKHS+PSD
clear all;
close all;
addpath('functions','data');
warning('off');

implement=0;
% 0: load results,
%1: run code, 
if implement
   load('netfset.mat');

    mae_us=zeros(30,1);   
    rmse_us=zeros(30,1);
    mae_it=zeros(30,1);
    rmse_it=zeros(30,1);
    mtu=zeros(30,1);
    mti=zeros(30,1);
    for r=1:10
        s=netfset{r}.train;
        utest=netfset{r}.test;
        
        [Un,In]=size(s);
       
        [su0,si0,mu,mi,mui,mask,rs,cs,ff]=preprocess(s);
%         mtu(r)=mtu(r)+toc(t0);
%         mti(r)=mti(r)+toc(t0);
        %%%%user-based%%%%%%%%%%
        if r<6
%       %%%%%%%item-first(r=1-5)%%%%%%%
%            
            rr=500;
            beta=300;
 
                alpha1=0.0001;
                beta1=0.0025;
                gama1=200;

                rr1=0.2;
        else
        %%%%%%%user-first(r=6-10)%%%%%%%
            rr=500;
            beta=300;

                alpha1=0.0001;
                beta1=0.001;
                gama1=200;

                rr1=0.2;
        end

        %% RKHS+kBR(Yang-2020)
        t0=tic;
        M=s;
        mask1=(M>0);
        [aM,mU,mUI]=AdjustUI(M);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Wi=Simxy(aM,mask1,1);
        fvector=Feavec(M,mask1);
        K=KernelGram(fvector,gama1);
        
        D=diag(sum(Wi,2));
        L=D-Wi;
        L=size(L,1)*L/trace(L);
        [U,V]=eig(L);
        estr=zeros(size(utest,1),1);
        items=unique(utest(:,2));  
        k=5;
 
        %         klk=beta*L+alph*Ki;
        
        klk=alpha1*K+beta1*K*L*K;
        Uk=U(:,1:k);
        for j=1:numel(items)
            labels=find(mask1(:,items(j))); 
            us=find(utest(:,2)==items(j));  
            numlbs=numel(labels);
            if numlbs>0
                yL=M(labels,items(j));
                KL=K(labels,:);
                %%%%%%%==my opion
                a=pinv(Uk'*(KL'*KL+klk)*Uk)*(Uk'*KL'*yL);
                f=K*Uk*a;
                %%%%%%%==my opion
            else
                allus=1:U;
                idx=find(sum(mask1,2)==0);
                if ~isempty(idx)
                    f(idx)=sum(M(:).*mask1(:))/sum(mask1(:));
                    allus(idx)=[];
                end
                f(allus)=sum(M(allus,:).*mask1(allus,:),2)./sum(mask1(allus,:),2);
            end
            estr(us)=f(utest(us,1));
        end
        mtu(r)=mtu(r)+toc(t0);

        err=abs(estr-utest(:,3));
        mae=mean(err);
        err=(estr-utest(:,3)).^2;
        rmse=sqrt(mean(err)); 
        mae_us(r)=mae;
        rmse_us(r)=rmse;

        %% RKHS+PSD%%(Yang-2022)
        [s0,~,~,~,~]=preprocess_GRPSD(s);
        t0=tic;
        [G,K,as,mr,mrc]=GKas(s0,mask,gama1,true);
        
        psdparam.sm=51;     
        psdparam.quantization = false;
        psd=my_psd_estimate(G,as,mask);
        psd=psd/max(psd);
        
        p=@(x)exp(-x/rr1);
        wf=p(psd);
        wf=G.lmax*wf/max(wf);
        
        f=KBreconstucter(G,as,mask,K,alpha1,beta1,10,wf);
        f=f+mr+mrc;             
%         f=f+mu;
        ff(rs,cs)=f;
        idx=find(ff>5); %if a rating is larger than 5 then set it to 5
        ff(idx)=5;
        idx=find(ff<1); %if a rating is  less than 1 then set it to 1
        ff(idx)=1;
        mtu(r+10)=mtu(r+10)+toc(t0);
        
        err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
        mae=mean(err);
        rmse=sqrt(mean(err.^2));
        mae_us(r+10)=mae;
        rmse_us(r+10)=rmse;

        %% GFPSD+appro  (Ours method)
        t0=tic; 
        su=CF(su0,mask);
        S=cov(su');
        [U,psd]=gsp_FB_estimate(S);   
        psd=psd/max(psd);
        wf=@(x)exp(-x*rr); %1./(x.^rr+eps);%
        w=wf(psd);
        w=w/max(w);
        
        f = reconbyFB_impro(U,su,mask,beta,w,20);
        f=f+mu;
        ff(rs,cs)=f;
        idx=find(ff>5); %if a rating is larger than 5 then set it to 5
        ff(idx)=5;
        idx=find(ff<1); %if a rating is  less than 1 then set it to 1
        ff(idx)=1;
        mtu(r+20)=mtu(r+20)+toc(t0);
        
        err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
        mae=mean(err);
        rmse=sqrt(mean(err.^2));
        mae_us(r+20)=mae;
        rmse_us(r+20)=rmse;
     
        %%%%item-based%%%%%%%%%%      
        if r<6
%       %%%%%%%item-first(r=1-5)%%%%%%%            
            rr=500;
            beta=300;

                alpha1=0.005;
                beta1=0.001;
                gama1=200;
               
                alpha2=0.0001;
            beta2=0.000025;
            gama2=200;
            rr2=0.5;
          
        else
        %%%%%%%user-first(r=6-10)%%%%%%%
            rr=500;
            beta=300;
             
                alpha1=0.005;
                beta1=0.001;
                gama1=200;
                rr1=0.2;

                  alpha2=0.0001;
                  beta2=0.00025;
                  gama2=200;
                  rr2=0.2;
        end
       
        %% RKHS+kBR (Yang-2020)
        t0=tic;
        M=s;
        mask1=(M>0);
        [aM,mU,mUI]=AdjustUI(M);
        Wi=Simxy(aM',mask1',1);
        fvector=Feavec(M',mask1');
        K=KernelGram(fvector,gama1);
        %K=KernelGram(M',mask',para,1);
        
        D=diag(sum(Wi,2));
        L=D-Wi;
        L=size(L,1)*L/trace(L);
        [U,V]=eig(L);
        estr=zeros(size(utest,1),1);
        us=unique(utest(:,1));
        k=10;
        klk=alpha1*K+beta1*K*L*K;
        Uk=U(:,1:k);
        for j=1:numel(us)
            labels=find(mask1(us(j),:));
            items=find(utest(:,1)==us(j));
            numlbs=numel(labels);
            if numlbs>0
                yL=M(us(j),labels)';
                KL=K(labels,:);
                a=pinv(Uk'*(KL'*KL+klk)*Uk)*(Uk'*KL'*yL);
                f=K*Uk*a;
            else
                allitem=1:In;
                idx=find(sum(mask)==0);
                if ~isempty(idx)
                    f(idx)=sum(M(:).*mask1(:))/sum(mask1(:));
                    allitem(idx)=[];
                end
                f(allitem)=sum(M(:,allitem).*mask1(:,allitem))./sum(mask1(:,allitem));
            end
            estr(items)=f(utest(items,2));
        end
        mti(r)=mtu(r)+toc(t0);
       
        err=abs(estr-utest(:,3));
        mae=mean(err);
        err=(estr-utest(:,3)).^2;
        rmse=sqrt(mean(err));
        mae_it(r)=mae;
        rmse_it(r)=rmse;

        %% RKHS+PSD  （Yang-2022）
        t0=tic;
        [G,K,as,mr,mrc]=GKas(s0',mask',gama2,true);
        
        psd=my_psd_estimate(G,as,mask');
        psd=psd/max(psd);
        p=@(x) exp(-rr2*x);
        wf=p(psd);
        wf=G.lmax*wf/max(wf);
        
        f=KBreconstucter(G,as,mask',K,alpha2,beta2,10,wf);
        f=f+mr+mrc;
%         f=f+mi';
        ff(rs,cs)=f';
        idx=find(ff>5); %if a rating is larger than 5 then set it to 5
        ff(idx)=5;
        idx=find(ff<1); %if a rating is  less than 1 then set it to 1
        ff(idx)=1;

        mti(r+10)=mti(r+10)+toc(t0);
        err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
        mae=mean(err);
        rmse=sqrt(mean(err.^2));
        mae_it(r+10)=mae;
        rmse_it(r+10)=rmse;

        %% GFPSD+appro  (Ours method)
        t0=tic;
        si=CF(si0',mask');
        S=cov(si');
        [U,psd]=gsp_FB_estimate(S);  
        psd=psd/max(psd);
        wf=@(x)exp(-x*rr);%1./(x.^rr+eps);%
        w=wf(psd);
        w=w/max(w);
        f = reconbyFB_impro(U,si,mask',beta,w,20);
        f=f'+mi;
        ff(rs,cs)=f;
        idx=find(ff>5); %if a rating is larger than 5 then set it to 5
        ff(idx)=5;
        idx=find(ff<1); %if a rating is  less than 1 then set it to 1
        ff(idx)=1;

        mti(r+20)=mti(r+20)+toc(t0);
        err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
        mae=mean(err);
        rmse=sqrt(mean(err.^2));
        mae_it(r+20)=mae;
        rmse_it(r+20)=rmse;

    end
    GFPSDvskBRPSDnetf=[mae_us,rmse_us,mtu,mae_it,rmse_it,mti];
    
    save('data/GFPSDvskBRPSDnetf.mat','GFPSDvskBRPSDnetf');
else
    load('GFPSDvskBRPSDnetf.mat');
end

disp('User-based: mae+-std+user-first---rmse+-std+user-first---the average cpu time----mae+-std+item-first----rmse+-std+item-first---the average cpu time');
disp(['RKHS+kBR    ',num2str(mean(GFPSDvskBRPSDnetf(6:10,1))),'+-',num2str(std(GFPSDvskBRPSDnetf(6:10,1))),'     ',num2str(mean(GFPSDvskBRPSDnetf(6:10,2))),'+-', ...
   num2str(std(GFPSDvskBRPSDnetf(6:10,2))),'     ',num2str(mean(GFPSDvskBRPSDnetf(6:10,3))),'     ',...
    num2str(mean(GFPSDvskBRPSDnetf(1:5,1))),'+-',num2str(std(GFPSDvskBRPSDnetf(1:5,1))),'     ',num2str(mean(GFPSDvskBRPSDnetf(1:5,2))),'+-', ...
   num2str(std(GFPSDvskBRPSDnetf(1:5,2))) ,'     ',num2str(mean(GFPSDvskBRPSDnetf(1:5,3)))]);
disp(['RKHS+PSD  ',num2str(mean(GFPSDvskBRPSDnetf(16:20,1))),'+-',num2str(std(GFPSDvskBRPSDnetf(16:20,1))),'     ',num2str(mean(GFPSDvskBRPSDnetf(16:20,2))),'+-', ...
   num2str(std(GFPSDvskBRPSDnetf(16:20,2))),'     ',num2str(mean(GFPSDvskBRPSDnetf(16:20,3))),'     ',...
    num2str(mean(GFPSDvskBRPSDnetf(11:15,1))),'+-',num2str(std(GFPSDvskBRPSDnetf(11:15,1))),'     ',num2str(mean(GFPSDvskBRPSDnetf(11:15,2))),'+-', ...
   num2str(std(GFPSDvskBRPSDnetf(11:15,2))) ,'     ',num2str(mean(GFPSDvskBRPSDnetf(11:15,3)))]);
disp(['Our method   ',num2str(mean(GFPSDvskBRPSDnetf(26:30,1))),'+-',num2str(std(GFPSDvskBRPSDnetf(26:30,1))),'     ',num2str(mean(GFPSDvskBRPSDnetf(26:30,2))),'+-', ...
   num2str(std(GFPSDvskBRPSDnetf(26:30,2))),'     ',num2str(mean(GFPSDvskBRPSDnetf(26:30,3))),'     ',...
    num2str(mean(GFPSDvskBRPSDnetf(21:25,1))),'+-',num2str(std(GFPSDvskBRPSDnetf(21:25,1))),'     ',num2str(mean(GFPSDvskBRPSDnetf(21:25,2))),'+-', ...
   num2str(std(GFPSDvskBRPSDnetf(21:25,2))) ,'     ',num2str(mean(GFPSDvskBRPSDnetf(21:25,3)))]);

disp('Item-based: mae+user-first---rmse+user-first---the average cpu time---mae+item-first---rmse+item-first---the average cpu time');
disp(['RKHS+kBR    ',num2str(mean(GFPSDvskBRPSDnetf(6:10,4))),'+-',num2str(std(GFPSDvskBRPSDnetf(6:10,4))),'     ',num2str(mean(GFPSDvskBRPSDnetf(6:10,5))),'+-', ...
   num2str(std(GFPSDvskBRPSDnetf(6:10,5))),'     ',num2str(mean(GFPSDvskBRPSDnetf(6:10,6))),'     ',...
    num2str(mean(GFPSDvskBRPSDnetf(1:5,4))),'+-',num2str(std(GFPSDvskBRPSDnetf(1:5,4))),'     ',num2str(mean(GFPSDvskBRPSDnetf(1:5,5))),'+-', ...
   num2str(std(GFPSDvskBRPSDnetf(1:5,5))) ,'     ',num2str(mean(GFPSDvskBRPSDnetf(1:5,6)))]);
disp(['RKHS+PSD   ',num2str(mean(GFPSDvskBRPSDnetf(16:20,4))),'+-',num2str(std(GFPSDvskBRPSDnetf(16:20,4))),'     ',num2str(mean(GFPSDvskBRPSDnetf(16:20,5))),'+-', ...
   num2str(std(GFPSDvskBRPSDnetf(16:20,5))),'     ',num2str(mean(GFPSDvskBRPSDnetf(16:20,6))),'     ',...
    num2str(mean(GFPSDvskBRPSDnetf(11:15,4))),'+-',num2str(std(GFPSDvskBRPSDnetf(11:15,4))),'     ',num2str(mean(GFPSDvskBRPSDnetf(11:15,5))),'+-', ...
   num2str(std(GFPSDvskBRPSDnetf(11:15,5))) ,'     ',num2str(mean(GFPSDvskBRPSDnetf(11:15,6)))]);
disp(['Our method   ',num2str(mean(GFPSDvskBRPSDnetf(26:30,4))),'+-',num2str(std(GFPSDvskBRPSDnetf(26:30,4))),'     ',num2str(mean(GFPSDvskBRPSDnetf(26:30,5))),'+-', ...
   num2str(std(GFPSDvskBRPSDnetf(26:30,5))),'     ',num2str(mean(GFPSDvskBRPSDnetf(26:30,6))),'     ',...
    num2str(mean(GFPSDvskBRPSDnetf(21:25,4))),'+-',num2str(std(GFPSDvskBRPSDnetf(21:25,4))),'     ',num2str(mean(GFPSDvskBRPSDnetf(21:25,5))),'+-', ...
   num2str(std(GFPSDvskBRPSDnetf(21:25,5))) ,'     ',num2str(mean(GFPSDvskBRPSDnetf(21:25,6)))]);

