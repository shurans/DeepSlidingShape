INFOroot = '/home/shurans/deepDetectLocal/marvinlog/';
figuredir = '/n/fs/modelnet/deepDetect/web/plot_loss/';
fnameArray = dir([INFOroot '*.log']);
fnameArray = {fnameArray.name};

for ff = 1:length(fnameArray)
    fname = fullfile(INFOroot,fnameArray{ff})
    if exist(fname,'file')
        try 
            fid = fopen(fname,'r');
            tline = fgetl(fid);
            iterArray = zeros(0,3);
            testArray = zeros(0,3);
            %{
            Iteration 5331  learning_rate = 0.001 loss = 0.336912 loss = 0.336912
            Iteration 5300 accuracy = 0.911719 loss = 0.237851
            %} 
            iter = 0;
            loss = 0;
            acc = 0;
             while ischar(tline)
                 p1 = strfind(tline,'Iteration');              
                 p2 = strfind(tline,'learning_rate = ');
                 p4 = strfind(tline,'accuracy = ');
                 p3 = strfind(tline,'loss = ');
                 if ~isempty(p1)&&~isempty(p2)&&~isempty(p3)
                     iter = str2num(tline(p1+10:p2-1));
                     loss = str2num(tline(p3+7:p3+7+4));
                     p3_2 = strfind(tline(p3+7+4:end),'loss = ');
                     loss_2 =[];
                     if ~isempty(p3_2)
                         loss_2 = str2num(tline(p3+7+4+p3_2+7-1:min(p3+7+4+p3_2+7+4-1,length(tline))));
                     end
                     if isempty(loss_2)
                         loss_2 = 0;
                     end
                     iterArray(end+1,1:3) = [iter loss loss_2];
                 end
                 if  ~isempty(p4)
                     if ~isempty(p1)
                        iter = str2num(tline(p1+10:p4-1));
                     end
                     try
                     acc = str2num(tline(p4+10:p4+10+4));
                     end
                     if isempty(acc)
                         acc = 0;
                     end
                     if ~isempty(p3)
                        loss = str2num(tline(p3+7:p3+7+3));
                     else
                        loss =0;
                     end
                     testArray(end+1,1:3) = [iter loss acc];
                 end
                 tline = fgetl(fid);
             end
              fclose(fid);
            if ~isempty(iterArray)
                f= figure;
                iterArray(iterArray(:,2)>3,2)=3;
                plot(iterArray(:,1),iterArray(:,2),'-b');
                hold on;
                plot(iterArray(:,1),iterArray(:,3),'-g');

                plot(testArray(:,1),testArray(:,2),'-k');


                plot(testArray(:,1),max(0,2*(testArray(:,3)-0.5)),'+-r');
                text(testArray(:,1),testArray(:,3),num2str(round(testArray(:,3)*1000)/10));
                xlabel('iteration');
                ylabel('loss');
                legend('train loss','test loss','test accuracy');
                title(fnameArray{ff});
                saveas(f,fullfile(figuredir,[fnameArray{ff} '.png']))
                close(f)
            end
        end
    end
end