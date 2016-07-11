clc;
clear;
trian_set=cell(1);
train_num=1;
tot=0;
img_set=cell(1,1);
label_set=cell(1,1);
label1_set=cell(1,1);
label2_set=cell(1,1);

i=0;
k=0;
w=0;
dl=30;
for ii=6:7
    j=0;
    label1=load_untouch_nii(['/home/ljp/code/caffe-3d/examples/cmr/result/pronii/training_sa_crop_pat',int2str(ii),'-labelOurc1.nii.gz']);
    label2=load_untouch_nii(['/home/ljp/code/caffe-3d/examples/cmr/result/pronii/training_sa_crop_pat',int2str(ii),'-labelOurc2.nii.gz']);
    [r,c,d]=size(label1.img);
    label=label1.img;
    label(:)=0;
    [r,c,d]=size(label1.img);
    for x=0:ceil(r/dl)-1
        for y=0:ceil(c/dl)-1
            for z=0:ceil(d/dl)-1
                xstart=x*dl+1;
                xend=xstart+dl-1;
                ystart=y*dl+1;
                yend=ystart+dl-1;
                zstart=z*dl+1;
                zend=zstart+dl-1;
                if(xend>r)
                    xend=r;
                    xstart=r-dl+1;
                end
                if(yend>c)
                    yend=c;
                    ystart=c-dl+1;
                end
                if(zend>d)
                    zend=d;
                    zstart=d-dl+1;
                end
                above=sum(sum(sum(label1.img(xstart:xend,ystart:yend,zstart:zend)>0.1)))+sum(sum(sum(label2.img(xstart:xend,ystart:yend,zstart:zend)>0.2)));
                if(above<=200)
                    continue;
                end
                for m=ystart:yend
                    imgresult=imread(['/home/ljp/code/caffe-3d/examples/cmr/result/',int2str(ii-5),'/',int2str(j),'.png']);
                    imgresult(imgresult==128)=1;
                    imgresult(imgresult==255)=2;
                    label(xstart:xend,m,zstart:zend)=imgresult;
                    j=j+1;
                end
            end
        end
    end
    label1.img=label;
    save_untouch_nii(label1,['/home/ljp/code/caffe-3d/examples/cmr/result/training_sa_crop_pat',int2str(ii),'-labelOursmall.nii.gz'])
end

                    