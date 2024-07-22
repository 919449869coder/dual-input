% train1234=tezhengshuju(:,5:end);
% test5=tezhengshujuS1(:,5:end);
clc
clear all

test5=train1235;

for i=1:length(test5)
test5tezhengdata=test5{i};
% test5tezhengdata=cell2mat(test5tezhengdata);
test5tezhengd(:,1+5)=test5tezhengdata(:,:,1+5);
test5tezhengd(:,2+5)=test5tezhengdata(:,:,9+5);
test5tezhengd(:,3+5)=test5tezhengdata(:,:,10+5);
test5tezhengd(:,4+5)=test5tezhengdata(:,:,13+5);
test5tezhengd(:,5+5)=test5tezhengdata(:,:,16+5);
test5tezhengd(:,6+5)=test5tezhengdata(:,:,17+5);
test5tezheng{i}=test5tezhengd;
end
