function [C,V]= PCA(data,num)
         %data 300*N, C vector , V value
         data(sum(isnan(data),2)>0,:)=[];
         [~,N] = size(data); 
         mn = mean(data,2); 
         data = data - repmat(mn,1,N); 
         covariance = 1 / (N-1) * data * data'; 
         [PC, V] = eigs(covariance); 
         V = real(diag(V)); 
         % sort the value in decreasing order 
         [~, rindices] = sort(-1*V); 
         V = V(rindices);
         C =PC(:,rindices);
         C =C(:,1:num);
end