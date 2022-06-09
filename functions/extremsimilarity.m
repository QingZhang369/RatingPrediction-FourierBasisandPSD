function [w,c]=extremsimilarity(s,mask,rm,a,b)
[M,N]=size(s);
mri=sum(s)./sum(mask);
w=zeros(M,M);
c=w;
for i=1:M-1
    tt1=repmat(mask(i,:),M-i,1);
    tt2=tt1.*mask(i+1:end,:);
    for j=1:M-i
        coratings=find(tt2(j,:)>0);
        if ~isempty(coratings)
            li=sum(mask(i,:));
            lj=sum(mask(i+j,:));
            c(i,i+j)=numel(coratings)/(li+lj-numel(coratings));            
            ss1=1./(1+exp(-a*(abs(s(i,coratings)-rm).*...
                abs(s(i+j,coratings)-rm))));
            ss2=1./(1+exp(-b*(abs(s(i,coratings)-mri(coratings)).*...
                abs(s(i+j,coratings)-mri(coratings)))));
            w(i,i+j)=sum(ss1.*ss2)/sqrt(sum(ss1.^2)*sum(ss2.^2));
        end
    end
end
idx=find(w>0);
i=mod(idx,M);
j=fix(idx/M)+1;
w((i-1)*M+j)=w(idx);
c((i-1)*M+j)=c(idx);