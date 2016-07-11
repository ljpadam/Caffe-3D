depth=200;
l=200;
for i=9
    shrink_ans=load_untouch_nii(['/home/ljp/code/dense3dCrf-master/applicationAndExamples/cmr/results/denseCrf3dSegmMap.nii.gz']);
    label=load_untouch_nii(['./sa/training_sa_crop_pat',int2str(i),'-label.nii.gz']);
    shrink_ans=shrink_ans.img;
    shrink_ans=uint16(shrink_ans);
    temp1=(shrink_ans==1);
    temp2=(label.img==1);
    sum(sum(sum(shrink_ans==label.img)))/sum(sum(sum(label.img>-1)))
    dice1=2*sum(sum(sum((temp1+temp2)==2)))/(sum(sum(sum(temp1==1)))+sum(sum(sum(temp2==1))))
    temp1=(shrink_ans==2);
    temp2=(label.img==2);
    dice1=2*sum(sum(sum((temp1+temp2)==2)))/(sum(sum(sum(temp1==1)))+sum(sum(sum(temp2==1))))
    label.img=shrink_ans;
    save_untouch_nii(label,['/home/ljp/code/caffe-3d/examples/cmr/result/training_sa_crop_pat',int2str(i),'-labelOurcrf.nii.gz'])
end
    