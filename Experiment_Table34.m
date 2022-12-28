%%MovieLens-100k,compare the proposed method, RKHS+kBR, and RKHS+PSD
close all;
addpath('functions','data');
load('data/u1base.mat');
u{1}.base=u1base;
clear u1base;
load('data/u1test.mat');
u{1}.test=u1test;
clear u1test;
load('data/u2base.mat');
u{2}.base=u2base;
clear u2base;
load('data/u2test.mat');
u{2}.test=u2test;
clear u2test;
load('data/u3base.mat');
u{3}.base=u3base;
clear u3base;
load('data/u3test.mat');
u{3}.test=u3test;
clear u3test;
load('data/u4base.mat');
u{4}.base=u4base;
clear u4base;
load('data/u4test.mat');
u{4}.test=u4test;
clear u4test;
load('data/u5base.mat');
u{5}.base=u5base;
clear u5base;
load('data/u5test.mat');
u{5}.test=u5test;
clear u5test;

warning('off');

Un=943;
In=1682;
implement=0;
%0: load results,
%1: run code, 
if implement
    
    mae_us=zeros(15,1);   
    rmse_us=zeros(15,1);
    mae_it=zeros(15,1);
    rmse_it=zeros(15,1);
    mtu=zeros(15,1);
    mti=zeros(15,1);
    for r=1:5
        ubase=u{r}.base;
        utest=u{r}.test;
        s=zeros(Un,In);
        s((ubase(:,2)-1)*Un+ubase(:,1))=ubase(:,3);
        
        [Un,In]=size(s);
        [su0,si0,mu,mi,mui,mask,rs,cs,ff]=preprocess(s);
       
        %%%%user-based%%%%%%%%%%
        %%RKHS+kBR
        t0=tic;
        alpha1=0.005;
        beta1=0.001;
        gama1=200;
        
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
        k=10;
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

        %%RKHS+PSD%%
        [s0,~,~,~,~]=preprocess_GRPSD(s);
        t0=tic;
        alpha=0.0001;
        gama=200;
        beta=0.01;
        rr=1/200;

        [G,K,as,mr,mrc]=GKas(s0,mask,gama,true);
        psd=my_psd_estimate(G,as,mask);
        psd=psd/max(psd);
        
        p=@(x)exp(-x/rr);
        wf=p(psd);
        wf=G.lmax*wf/max(wf);
        f=KBreconstucter(G,as,mask,K,alpha,beta,10,wf);
        f=f+mr+mrc;
        ff(rs,cs)=f;
        idx=find(ff>5); %if a rating is larger than 5 then set it to 5
        ff(idx)=5;
        idx=find(ff<1); %if a rating is  less than 1 then set it to 1
        ff(idx)=1;      
        mtu(r+5)=mtu(r+5)+toc(t0);
        
        err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
        mae=mean(err);
        rmse=sqrt(mean(err.^2));
        mae_us(r+5)=mae;
        rmse_us(r+5)=rmse;


        %%the proposed method%%
        t0=tic;
        su=CF(su0,mask);
        S=cov(su');
        [U,psd]=gsp_FB_estimate(S);  
        psd=psd/max(psd);
        
        rr=500;
        beta=50;
        
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
        mtu(r+10)=mtu(r+10)+toc(t0);
        
        err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
        mae=mean(err);
        rmse=sqrt(mean(err.^2));
        mae_us(r+10)=mae;
        rmse_us(r+10)=rmse;

        %%%%item-based%%%%%%%%%%

        %%RKHS+kBR
        t0=tic;
        alpha1=0.005;
        beta1=0.001;
        gama1=200;

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

        %%RKHS+PSD%%,
        t0=tic;
        alpha=0.0001;
        gama=200;
        beta=0.001;
        rr=1/200;
 
        [G,K,as,mr,mrc]=GKas(s0',mask',gama,true);
        psd=my_psd_estimate(G,as,mask');
        psd=psd/max(psd);
        
        p=@(x)exp(-x/rr);
        wf=p(psd);
        wf=G.lmax*wf/max(wf);
        f=KBreconstucter(G,as,mask',K,alpha,beta,10,wf);
        f=f+mr+mrc;
        ff(rs,cs)=f';
        idx=find(ff>5); %if a rating is larger than 5 then set it to 5
        ff(idx)=5;
        idx=find(ff<1); %if a rating is  less than 1 then set it to 1
        ff(idx)=1;
        mti(r+5)=mti(r+5)+toc(t0);

        err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
        mae=mean(err);
        rmse=sqrt(mean(err.^2));
        mae_it(r+5)=mae;
        rmse_it(r+5)=rmse;

        %%the proposed method%% 
        t0=tic;
        [urs,itms]=size(mask);
        si=CF(si0',mask');
        S=cov(si');
        [U,psd]=gsp_FB_estimate(S);   
        psd=psd/max(psd);
        
        rr=500;
        beta=50;
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
        mti(r+10)=mti(r+10)+toc(t0);
        err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
        mae=mean(err);
        rmse=sqrt(mean(err.^2));
        mae_it(r+10)=mae;
        rmse_it(r+10)=rmse;

    end

    GFPSDvskBRPSDml=[mae_us,rmse_us,mtu,mae_it,rmse_it,mti];
    
    save('data/GFPSDvskBRPSDml.mat','GFPSDvskBRPSDml');
else
    load('GFPSDvskBRPSDml.mat');
end

disp('User-based: mae+-std--------rmse+-std--------the average cpu time');
disp(['RKHS+kBR    ',num2str(mean(GFPSDvskBRPSDml(1:5,1))),'+-',num2str(std(GFPSDvskBRPSDml(1:5,1))),...
    '     ',num2str(mean(GFPSDvskBRPSDml(1:5,2))),'+-',num2str(std(GFPSDvskBRPSDml(1:5,2))), '     ',num2str(mean(GFPSDvskBRPSDml(1:5,3)))]);
disp(['RKHS+PSD   ',num2str(mean(GFPSDvskBRPSDml(6:10,1))),'+-',num2str(std(GFPSDvskBRPSDml(6:10,1))),...
    '     ',num2str(mean(GFPSDvskBRPSDml(6:10,2))),'+-',num2str(std(GFPSDvskBRPSDml(6:10,2))), '     ',num2str(mean(GFPSDvskBRPSDml(6:10,3)))]);
disp(['Our method  ',num2str(mean(GFPSDvskBRPSDml(11:15,1))),'+-',num2str(std(GFPSDvskBRPSDml(11:15,1))),...
    '     ',num2str(mean(GFPSDvskBRPSDml(11:15,2))),'+-',num2str(std(GFPSDvskBRPSDml(11:15,2))), '     ',num2str(mean(GFPSDvskBRPSDml(11:15,3)))]);

disp('Item-based: mae+-std------------rmse+-std--------the average cpu time');
disp(['RKHS+kBR    ',num2str(mean(GFPSDvskBRPSDml(1:5,4))),'+-',num2str(std(GFPSDvskBRPSDml(1:5,4))),...
    '     ',num2str(mean(GFPSDvskBRPSDml(1:5,5))),'+-',num2str(std(GFPSDvskBRPSDml(1:5,5))), '     ',num2str(mean(GFPSDvskBRPSDml(1:5,6)))]);
disp(['RKHS+PSD   ',num2str(mean(GFPSDvskBRPSDml(6:10,4))),'+-',num2str(std(GFPSDvskBRPSDml(6:10,4))),...
    '     ',num2str(mean(GFPSDvskBRPSDml(6:10,5))),'+-',num2str(std(GFPSDvskBRPSDml(6:10,5))), '     ',num2str(mean(GFPSDvskBRPSDml(6:10,6)))]);
disp(['Our method   ',num2str(mean(GFPSDvskBRPSDml(11:15,4))),'+-',num2str(std(GFPSDvskBRPSDml(11:15,4))),...
    '     ',num2str(mean(GFPSDvskBRPSDml(11:15,5))),'+-',num2str(std(GFPSDvskBRPSDml(11:15,5))), '     ',num2str(mean(GFPSDvskBRPSDml(11:15,6)))]);



