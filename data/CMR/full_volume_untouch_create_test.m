clc;
clear;
trian_set=cell(1);
train_num=1;
tot=0;
img_set=cell(1,1);
label_set=cell(1,1);
for i=8:9
    label=load_untouch_nii(['./sa/training_sa_crop_pat',int2str(i),'-label.nii.gz']);
    img=load_untouch_nii(['./sa/training_sa_crop_pat',int2str(i),'.nii.gz']);
    %img=load_nii(['./sa/training_sa_crop_pat',int2str(i),'.nii.gz']);
    img=img.img;
    label=label.img;
    tot=tot+1;
    %label(label==1)=0;
    %label(label==2)=1;
    img_set{tot}=img;
    label_set{tot}=label;
    
end
tottemp=tot;
% for i=1:tottemp
%     img=img_set{i};
%     label=label_set{i};
%     for theta=1:7
%         thetan=theta*pi/4;
%         tform = affine3d([cos(thetan),-sin(thetan),0,0;sin(thetan),cos(thetan),0,0;0,0,1,0;0,0,0,1]);
%         imgt=imwarp(img,tform);
%         labelt=imwarp(label,tform);
%         tot=tot+1;
%         img_set{tot}=imgt;
%         label_set{tot}=labelt;
%     end
% end
i=0;
for ii=1:tot
    i=i+1;
    j=0;
    img=img_set{ii};
    label=label_set{ii};
    mkdir(['/home/ljp/hd1/CMR/',int2str(i)]);
    
    dl=200;
    [x y z]=ndgrid(linspace(1,size(img,1),dl),...
          linspace(1,size(img,2),dl),...
          linspace(1,size(img,3),dl));
    img=double(img);
    img=interp3(img,y,x,z,'cubic');
    [x y z]=ndgrid(linspace(1,size(label,1),dl),...
          linspace(1,size(label,2),dl),...
          linspace(1,size(label,3),dl));
    
    label=double(label);
    label=interp3(label,y,x,z,'nearest');   
    img=uint16(img);
    label=uint8(label);
    %label(label==0)=255;
    
    %img=img*20;
    %label=label*255;                                                                                                                                                                                                                                                                                                                                                                                                           
    [r,c,d]=size(img);
    %continue;
    for x=1:ceil(r/r)
        for y=1:ceil(c/c)
            for z=1:ceil(d/d)
                xstart=1;
                ystart=1;
                zstart=1;
                xend=r;
                yend=c;
                zend=d;
%                
                for m=ystart:yend
                    imgtemp=reshape(img(xstart:xend,m,zstart:zend),[xend-xstart+1,zend-zstart+1]);           
                    labeltemp=reshape(label(xstart:xend,m,zstart:zend),[xend-xstart+1,zend-zstart+1]);
                    %labeltemp=labeltemp*255;
                    
                    imwrite(imgtemp,['/home/ljp/hd1/CMR/test/',int2str(i),'/img_',int2str(j),'.png']);
                    imwrite(labeltemp,['/home/ljp/hd1/CMR/test/',int2str(i),'/label_',int2str(j),'.png']);
                    labeltemp=labeltemp*128;
                    imgtemp=imgtemp*20;
                    imwrite(imgtemp,['/home/ljp/hd1/CMR/test/',int2str(i),'/img_O',int2str(j),'.png']);
                    imwrite(labeltemp,['/home/ljp/hd1/CMR/test/',int2str(i),'/label_O',int2str(j),'.png']);
                    if(mod(j,dl)==0)
                        train_set{train_num}=['/home/ljp/hd1/CMR/test/',int2str(i),' ',int2str(j),' /home/ljp/hd1/CMR/test/',int2str(i)];
                    train_num=train_num+1;
                    end
                    j=j+1;
                end
            end
        end
    end
end
fileID=fopen('train.txt','w');
for i=1:train_num-1
    fprintf(fileID,'%s\n',train_set{i});
end
fclose(fileID);