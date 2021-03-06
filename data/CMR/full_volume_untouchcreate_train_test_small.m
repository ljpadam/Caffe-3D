clc;
clear;
trian_set=cell(1);
train_num=1;
tot=0;
img_set=cell(1,1);
label_set=cell(1,1);
label1_set=cell(1,1);
label2_set=cell(1,1);
ii=[0,1,2,3,4,5];
for j=1:6
    i=ii(j);
    label=load_untouch_nii(['./sa/training_sa_crop_pat',int2str(i),'-label.nii.gz']);
    label1=load_untouch_nii(['/home/ljp/code/caffe-3d/examples/cmr/result/pronii/training_sa_crop_pat',int2str(i),'-labelOurc1.nii.gz']);
    label2=load_untouch_nii(['/home/ljp/code/caffe-3d/examples/cmr/result/pronii/training_sa_crop_pat',int2str(i),'-labelOurc2.nii.gz']);
    img=load_untouch_nii(['./sa/training_sa_crop_pat',int2str(i),'.nii.gz']);
    img=img.img;
    label=label.img;
    label1=label1.img;
    label2=label2.img;
    tot=tot+1;
    %label(label==1)=0;
    %label(label==2)=1;
    img_set{tot}=img;
    label=double(label);
    label1=double(label1);
    label2=double(label2);
    label_set{tot}=label;
    label1_set{tot}=label1;
    label2_set{tot}=label2;
    
end
tottemp=tot;
for i=1:tottemp
    img=img_set{i};
    label=label_set{i};
    label1=label1_set{i};
    label2=label2_set{i};
    for theta=1:3
        thetan=theta*pi/2;
        tform = affine3d([cos(thetan),-sin(thetan),0,0;sin(thetan),cos(thetan),0,0;0,0,1,0;0,0,0,1]);
        imgt=imwarp(img,tform);
        labelt=imwarp(label,tform,'nearest');
        labelt1=imwarp(label1,tform,'nearest');
        labelt2=imwarp(label2,tform,'nearest');
        tot=tot+1;
        img_set{tot}=imgt;
        label_set{tot}=labelt;
        label1_set{tot}=labelt1;
        label2_set{tot}=labelt2;
    end
end
i=0;
k=0;
w=0;
dl=30;
for ii=1:tot
    i=i+1;
    j=0;
    img=img_set{ii};
    label=label_set{ii};
    label1=label1_set{ii};
    label2=label2_set{ii};
    [r,c,d]=size(img);
    mkdir(['/home/ljp/hd1/CMR/small/',int2str(i)]);                                                                                                                                                                                                                                                                                                                                                                                                       
    img=uint16(img);
    label=uint8(label); 
    disp('k'); disp(k)
    disp('w'); disp(w)
    k=0;
    w=0;
    [r,c,d]=size(img);
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
                above=sum(sum(sum(label1(xstart:xend,ystart:yend,zstart:zend)>0.1)))+sum(sum(sum(label2(xstart:xend,ystart:yend,zstart:zend)>0.2)));
                if(above<=200)
                    k=k+1;
                    continue;
                else
                    w=w+1;
                end
                for m=ystart:yend
                    imgtemp=reshape(img(xstart:xend,m,zstart:zend),[xend-xstart+1,zend-zstart+1]);           
                    labeltemp=reshape(label(xstart:xend,m,zstart:zend),[xend-xstart+1,zend-zstart+1]);
                    %labeltemp=labeltemp*255;
                    
                    imwrite(imgtemp,['/home/ljp/hd1/CMR/small/',int2str(i),'/img_',int2str(j),'.png']);
                    imwrite(labeltemp,['/home/ljp/hd1/CMR/small/',int2str(i),'/label_',int2str(j),'.png']);
                    labeltemp=labeltemp*128;
                    imgtemp=imgtemp*20;
                    imwrite(imgtemp,['/home/ljp/hd1/CMR/small/',int2str(i),'/img_O',int2str(j),'.png']);
                    imwrite(labeltemp,['/home/ljp/hd1/CMR/small/',int2str(i),'/label_O',int2str(j),'.png']);
                    if(mod(j,dl)==0)
                        train_set{train_num}=['/home/ljp/hd1/CMR/small/',int2str(i),' ',int2str(j),' /home/ljp/hd1/CMR/small/',int2str(i)];
                    train_num=train_num+1;
                    end
                    j=j+1;
                end
            end
        end
    end
end
fileID=fopen('smalltrain.txt','w');
for i=1:train_num-1
    fprintf(fileID,'%s\n',train_set{i});
end
fclose(fileID);
                    