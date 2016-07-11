labelnum=9;
parameter_set=cell(35,1);
parameter_set{1}='-numberOfModalitiesAndFiles';
parameter_set{2}='1';
parameter_set{3}=['/home/ljp/code/dense3dCrf-master/applicationAndExamples/cmr/training_sa_crop_pat',int2str(labelnum),'.nii.gz'];
parameter_set{4}='-numberOfForegroundClassesAndProbMapFiles';
parameter_set{5}='2';
parameter_set{6}=['/home/ljp/code/dense3dCrf-master/applicationAndExamples/cmr/training_sa_crop_pat',int2str(labelnum),'-labelOurc1.nii.gz'];
parameter_set{7}=['/home/ljp/code/dense3dCrf-master/applicationAndExamples/cmr/training_sa_crop_pat',int2str(labelnum),'-labelOurc2.nii.gz'];
parameter_set{8}='-imageDimensions';
imgforsize=load_untouch_nii(parameter_set{6});
[x,y,z]=size(imgforsize.img);

parameter_set{9}='3';
parameter_set{10}=int2str(x);
parameter_set{11}=int2str(y);
parameter_set{12}=int2str(z);
parameter_set{13}='-minMaxIntensities';
parameter_set{14}='0';
parameter_set{15}='2500';
parameter_set{16}='-outputFolder';
parameter_set{17}='/home/ljp/code/dense3dCrf-master/applicationAndExamples/cmr/results/';
parameter_set{18}='-prefixForOutputSegmentationMap';
parameter_set{19}='denseCrf3dSegmMap';
parameter_set{20}='-prefixForOutputProbabilityMaps';
parameter_set{21}='denseCrf3dProbMapClass';
parameter_set{22}='-pRCZandW';
parameter_set{23}='3.0';
parameter_set{24}='3.0';
parameter_set{25}='3.0';
parameter_set{26}='3.0';
parameter_set{27}='-bRCZandW';
maxdice1=0;
maxdice1Index='';
maxdice2=0;
maxdice2Index='';
maxdiceavg=0;
maxdiceavgIndex='';
for i=2  %2:200
    for j=620 %300:10:780
        for k=5 %1:10
            parameter_set{28}=int2str(i);
            parameter_set{29}=int2str(i);
            parameter_set{30}=int2str(i);
            parameter_set{31}=int2str(j);
            parameter_set{32}='-bMods';
            parameter_set{33}=int2str(k);
            parameter_set{34}='-numberOfIterations';
            parameter_set{35}='10';
            
            fileID=fopen('./crfParameter.txt','w');
            for e=1:35
                fprintf(fileID,'%s\n',parameter_set{e});
            end
            fclose(fileID);
            !/home/ljp/code/dense3dCrf-master/build/applicationAndExamples/dense3DCrfInferenceOnNiis -c ./crfParameter.txt
%             shrink_ans=load_untouch_nii(['/home/ljp/code/dense3dCrf-master/applicationAndExamples/cmr/results/denseCrf3dSegmMap.nii.gz']);
%             label=load_untouch_nii(['./sa/training_sa_crop_pat4-label.nii.gz']);
%             shrink_ans=shrink_ans.img;
%             shrink_ans=uint16(shrink_ans);
%             temp1=(shrink_ans==1);
%             temp2=(label.img==1);
%             dice1=2*sum(sum(sum((temp1+temp2)==2)))/(sum(sum(sum(temp1==1)))+sum(sum(sum(temp2==1))));
%             if dice1>maxdice1
%                 maxdice1=dice1;
%                 maxdice1Index=[int2str(i),' ',int2str(j),' ',int2str(k)];
%             end
%             temp1=(shrink_ans==2);
%             temp2=(label.img==2);
%             dice2=2*sum(sum(sum((temp1+temp2)==2)))/(sum(sum(sum(temp1==1)))+sum(sum(sum(temp2==1))));
%             if dice2>maxdice2
%                 maxdice2=dice2;
%                 maxdice2Index=[int2str(i),' ',int2str(j),' ',int2str(k)];
%             end
%             if (dice1+dice2)/2>maxdiceavg
%                 maxdiceavg=(dice1+dice2)/2;
%                 maxdiceavgIndex=[int2str(i),' ',int2str(j),' ',int2str(k)];
%             end
            fileID=fopen('./bestcrfParameter.txt','w');
            fprintf(fileID,'%f %s\n',maxdice1,maxdice1Index);
            fprintf(fileID,'%f %s\n',maxdice2,maxdice2Index);
            fprintf(fileID,'%f %s\n',maxdiceavg,maxdiceavgIndex);
            fprintf(fileID,'%f %f %f\n',i,j,k);
            fclose(fileID);
        end
    end
end

