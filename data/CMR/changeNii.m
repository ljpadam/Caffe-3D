
for i=0:9
    label=load_untouch_nii(['/home/ljp/temp/crf2/training_sa_crop_pat',int2str(i),'-labelOurcrf.nii.gz']);
    temp=label.img;
    temp(temp>0)=1;
    CC = bwconncomp(temp);
    ma=0;
    maindex=0;
    CCsize=size(CC.PixelIdxList);
    CCsize=CCsize(1)*CCsize(2);
    for j=1:CCsize
        sizenow=size(CC.PixelIdxList{j});
        sizenow=sizenow(1);
        if(sizenow>ma)
            ma=sizenow;
            maindex=j;
        end
    end
    temp(:)=0;
    temp(CC.PixelIdxList{maindex})=label.img(CC.PixelIdxList{maindex});
    label.img=temp;
    save_untouch_nii(label,['/home/ljp/temp/crf2/small/training_sa_crop_pat',int2str(i),'-label.nii.gz']);
end