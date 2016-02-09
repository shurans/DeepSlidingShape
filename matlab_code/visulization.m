function visulization(SUNRGBDMeta,cnn3d_model,foldername,imdb,suffix)

            cnn3d_model.webfolder = '../../';
            rootfolderWeb =fullfile(cnn3d_model.webfolder,'web',[foldername suffix],'/');
            fprintf('%s\n',rootfolderWeb);
            mkdir(rootfolderWeb)
            fopen(fullfile(rootfolderWeb,'index.html'));

            fid = fopen([rootfolderWeb '/viewer.html'],'w');
            fprintf(fid,'<!DOCTYPE html> \n<html>\n <body>');
            fprintf(fid,'<style>\nimg { border: 5px blue solid; margin: 5px;}\n</style>');
            fprintf(fid,'<center><h2>click on the AP plot to look at detailed results for each category<br><br><br></center>');
            for i =1:length(cnn3d_model.classes)
                mkdir_if_missing(fullfile(rootfolderWeb,cnn3d_model.classes{i}));
                fprintf(fid,'<td><a href="%s/%s.html">',cnn3d_model.classes{i},cnn3d_model.classes{i});
                fprintf(fid,'<img src="%s" height="%d"></a><td>',[cnn3d_model.classes{i} '/' cnn3d_model.classes{i} '.jpg'],300);
            end
            fprintf(fid,'\n</body>\n</html>');
            fclose(fid);

            for i =1:length(cnn3d_model.classes)
                outputpath = fullfile(rootfolderWeb,cnn3d_model.classes{i});
                resultfile =[cnn3d_model.conf.cache_dir '/' cnn3d_model.classes{i} '_pr_' imdb.name suffix '.mat'];
                load(resultfile);
                % plot AP
                f= figure;
                plot(recall, prec,'LineWidth',3);
                title(sprintf('%s AP:%.4f',strrep(cnn3d_model.classes{i},'_',' '),ap_auc),'FontSize', 40);
                ylim([0 1]);xlim([0 1]);
                set(gca,'FontSize',15)
                print(gcf, '-djpeg', '-r100', [rootfolderWeb '/' cnn3d_model.classes{i} '/' cnn3d_model.classes{i} '.jpg']);
                close(f);

                % gen image for each box
                numofExample = 100;
                fieldNames ={'conf','TP','FP'};
                fieldvalues = zeros(numofExample,length(fieldNames));
                imagesNameall = cell(numofExample,1);
                for boxid = 1:numofExample
                    [~,ind]=ismember(imdb.image_ids{all_imageid(boxid)},{SUNRGBDMeta.sequenceName});
                    data = SUNRGBDMeta(ind);

                    imagesNameall{boxid} = sprintf('%04d-%s',boxid,strrep([imdb.image_ids{all_imageid(boxid)} '.jpg'],'/','__'));
                    % genimage
                    gtBB =[];
                    if ~isempty(data.groundtruth3DBB_tight)
                        data.groundtruth3DBB_tight = data.groundtruth3DBB_tight(ismember({data.groundtruth3DBB_tight.classname},cnn3d_model.classes));
                        [gtidx] = find(ismember({data.groundtruth3DBB_tight.classname},cnn3d_model.classes{i}));
                        gtBB = data.groundtruth3DBB_tight(gtidx);
                    end
                    [rgb,points,depthInpaint,imsize]=read3dPoints(data);
                    imageMaskImag = vis_applyBbMaskTightBB(rgb,points,data,all_boxes(boxid),gtBB,[0,1,0]);
                    [~,Bb2dDraw] = projectStructBbsTo2d(all_boxes(boxid),data.Rtilt,[1,1],data.K);
                    imageMaskImag = drawBbOnImage(imageMaskImag,Bb2dDraw,[0 1 1]);
                    [~,gtBb2dDraw] = projectStructBbsTo2d(gtBB,data.Rtilt,[1,1],data.K);
                    for j = 1:size(gtBb2dDraw,1), % draw gt BB
                         imageMaskImag = drawBbOnImage(imageMaskImag,gtBb2dDraw(j,:),[1 1 0]);
                    end
                    % write image
                    imwrite(imageMaskImag,[outputpath '/' imagesNameall{boxid}],'quality',25);
                    fieldvalues(boxid,1) = all_boxes(boxid).conf;
                    fieldvalues(boxid,2) = tp(boxid);
                    fieldvalues(boxid,3) = 1-tp(boxid);
                end


                htmlFullName = fullfile(rootfolderWeb,[cnn3d_model.classes{i} '/' cnn3d_model.classes{i} '.html']);
                textstr = 'Green: detection box. Yellow: ground truth box. conf: detection confidence TP: true postive FP: false postive';
                gensorttableWeb(htmlFullName,imagesNameall,fieldvalues,fieldNames,textstr)    
            end


end