depth=200;
l=200;
for i=8:9
    shrink_ans_c1=zeros(l,depth,l);
    shrink_ans_c2=zeros(l,depth,l);
    for j=0:depth-1
        temp=imread(['/home/ljp/code/caffe-3d/examples/cmr/result/pro/',int2str(i-7),'/class1/',int2str(j),'.png']);
        shrink_ans_c1(:,j+1,:)=temp;
        temp=imread(['/home/ljp/code/caffe-3d/examples/cmr/result/pro/',int2str(i-7),'/class2/',int2str(j),'.png']);
        shrink_ans_c2(:,j+1,:)=temp;
    end
    label=load_untouch_nii(['./sa/training_sa_crop_pat',int2str(i),'.nii.gz']);
    [ny,nx,nz]=size(label.img);
    [y,x,z]=ndgrid(linspace(1,size(shrink_ans_c1,1),ny),...
          linspace(1,size(shrink_ans_c1,2),nx),...
          linspace(1,size(shrink_ans_c1,3),nz));
    shrink_ans_c1=double(shrink_ans_c1);
    shrink_ans_c1=interp3(shrink_ans_c1,x,y,z,'nearest');
    shrink_ans_c2=double(shrink_ans_c2);
    shrink_ans_c2=interp3(shrink_ans_c2,x,y,z,'nearest');
    shrink_ans_c1=shrink_ans_c1/255.0;
    shrink_ans_c2=shrink_ans_c2/255.0;
    shrink_ans_c1(shrink_ans_c1>0.95)=0.95;
    shrink_ans_c2(shrink_ans_c2>0.95)=0.95;
    temp=shrink_ans_c1+shrink_ans_c2;
    sum(sum(sum(temp>1)))
    %label.hdr.dime.datatype=16;
    %label.hdr.dime.bitpix=32;
    label.img=shrink_ans_c1;
    save_untouch_nii(label,['/home/ljp/code/caffe-3d/examples/cmr/result/training_sa_crop_pat',int2str(i),'-labelOurc1.nii.gz'])
    label.img=shrink_ans_c2;
    save_untouch_nii(label,['/home/ljp/code/caffe-3d/examples/cmr/result/training_sa_crop_pat',int2str(i),'-labelOurc2.nii.gz'])
    %temp=load_nii(['/home/ljp/code/caffe-3d/examples/cmr/result/training_sa_crop_pat',int2str(i),'-labelOurc1.nii.gz']);
%     temp.hdr.dime.datatype=16;
%     temp.hdr.dime.bitpix=32;
%     save_nii(temp,['/home/ljp/code/caffe-3d/examples/cmr/result/training_sa_crop_pat',int2str(i),'-labelOurc1.nii.gz']);
%     temp=load_nii(['/home/ljp/code/caffe-3d/examples/cmr/result/training_sa_crop_pat',int2str(i),'-labelOurc2.nii.gz']);
%     temp.hdr.dime.datatype=16;
%     temp.hdr.dime.bitpix=32;
%     save_nii(temp,['/home/ljp/code/caffe-3d/examples/cmr/result/training_sa_crop_pat',int2str(i),'-labelOurc2.nii.gz']);
end
    