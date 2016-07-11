depth=200;
l=200;
maxdice=0;
maxdiceindex=0;
for iter=100:100:100
    %[status,cmdout]=system(['python /home/ljp/code/caffe-3d/examples/cmr/cmr_test.py ',int2str(iter)]);
    tempmax=0;
    for i=4:5
    
    shrink_ans=zeros(l,depth,l);
    for j=0:depth-1
        temp=imread(['/home/ljp/code/caffe-3d/examples/cmr/result/',int2str(i-3),'/',int2str(j),'.png']);
        %temp=imrotate(temp,-90);
        shrink_ans(:,j+1,:)=temp;
    end
    label=load_untouch_nii(['./sa/training_sa_crop_pat',int2str(i),'-label.nii.gz']);
    %label=load_nii(['./sa/training_sa_crop_pat',int2str(i),'-label.nii.gz']);
    [ny,nx,nz]=size(label.img);
    [y,x,z]=ndgrid(linspace(1,size(shrink_ans,1),ny),...
          linspace(1,size(shrink_ans,2),nx),...
          linspace(1,size(shrink_ans,3),nz));
    shrink_ans(shrink_ans==128)=1;
    shrink_ans(shrink_ans==255)=2;
    shrink_ans=double(shrink_ans);
    shrink_ans=interp3(shrink_ans,x,y,z,'nearest');
    shrink_ans=uint16(shrink_ans);
    
    temp=shrink_ans;
    temp=0;
    %temp(label.img<shrink_ans)=1;
    %temp(label.img==shrink_ans)=0;
    temp(((label.img~=shrink_ans)+(label.img==2))==2)=1;
    
    
    temp1=(shrink_ans==1);
    temp2=(label.img==1);
    sum(sum(sum(shrink_ans==label.img)))/sum(sum(sum(label.img>-1)));
    dice1=2*sum(sum(sum((temp1+temp2)==2)))/(sum(sum(sum(temp1==1)))+sum(sum(sum(temp2==1))))
    tempmax=tempmax+dice1;
    temp1=(shrink_ans==2);
    temp2=(label.img==2);
    dice1=2*sum(sum(sum((temp1+temp2)==2)))/(sum(sum(sum(temp1==1)))+sum(sum(sum(temp2==1))))
    label.img=shrink_ans;
    tempmax=tempmax+dice1;
    %label.img=temp;
    save_untouch_nii(label,['/home/ljp/code/caffe-3d/examples/cmr/result/training_sa_crop_pat',int2str(i),'-labelOur.nii.gz'])
    end
    if(tempmax>maxdice)
        maxdice=tempmax;
        maxdiceindex=iter;
        disp(maxdice); disp(maxdiceindex);
    end
end
    