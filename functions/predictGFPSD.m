function GFPSD_user=predictGFPSD_user(s,utest,alpha,beta,gama,rr,k)

        %%%%%%%%%%%%%%%user-based%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [s,mask,rs,cs,ff]=preprocess(s);
        [urs,itms]=size(mask);

        %%%%Predict score by collaborative filtering (CF)%%%
        %%基于用户预测
        [as,mU,mUI]=AdjustUI(s);     %%按行分别预测本行缺失标签的数据
        sim=3;
        if sim==3
            W=Simxy(as,mask,1);      %%行与行之间的相似度要放到整个评分表去看
        else
            W=Simxy(s,mask,sim);
        end

        s_user=s;
        inter=[0,900,itms];
        for j=1:2
        sj=s(:,inter(j)+1:inter(j+1));
        maskj=mask(:,inter(j)+1:inter(j+1));
        [idxr,idxc]=find(sj==0);
        idx0=[idxr,idxc];  %%idx0:s中待预测元素的行列位置
        CF_pre=Estimate(sj,maskj,W,idx0,k);   %%列向量，idx0中每个对应元素的估计评分
        s_user((idxc-1)*Un+idxr)=CF_pre;           %%s_user是user-based的预测评分表
        end

        
        %%基于用户预测评分矩阵
        s_pre=s_user;
        
        %%%%Estimation for Fourier basis and PSD%%%%%%%
        S=cov(s_pre');       %%协方差，变量要写成行向量的形式，基于用户，用户是变量
        [U,psd,~]=gsp_FB_estimate(S);   %%U：Un*Un
        psd=psd/max(psd);
        p=@(x)exp(-x/rr);
        wf=p(psd);
        wf=wf/max(wf); 
        
        %%利用估计的U和psd，由s的数据去估计u_test位置的数据，并与真实的u_test对比
        [K,as,mr,mrc]=Kas(s,mask,gama,true);
        f=KBreconstucterU(U,as,mask,K,alpha,beta,10,wf);
        f=f+mr+mrc;
        ff(rs,cs)=f;
        idx=find(ff>5); %if a rating is larger than 5 then set it to 5
        ff(idx)=5;
        idx=find(ff<1); %if a rating is  less than 1 then set it to 1
        ff(idx)=1;
        err=abs(ff((utest(:,2)-1)*Un+utest(:,1))-utest(:,3));
        
        
        mae_us=[mae_us;mean(err)];
        rmse_us=[rmse_us;sqrt(mean(err.^2))];
        
   
        GFPSD_user=[mae_us,rmse_us];

%     save('data/GFPSD.mat','GFPSD');


% disp('User-based: mae+-std------------------rmse+-std');
% disp(['            ',num2str(mean(GFPSD(:,1))),'+-',num2str(std(GFPSD(:,1))),...
%     '     ',num2str(mean(GFPSD(:,3))),'+-',num2str(std(GFPSD(:,3)))]);
% disp('Item-based: mae+-std------------------rmse+-std');
% disp(['            ',num2str(mean(GFPSD(:,2))),'+-',num2str(std(GFPSD(:,2))),...
%     '     ',num2str(mean(GFPSD(:,4))),'+-',num2str(std(GFPSD(:,4)))]);