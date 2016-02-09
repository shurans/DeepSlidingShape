function Allfiles = dirAll (path,string)
folders = regexp(genpath(path), pathsep, 'split');
folders = folders(1:end-1);
cnt = 0;
for f=1:length(folders)
    files = dir([folders{f} '/*' string]);
    for j=1:length(files)
        if ~files(j).isdir
            cnt = cnt + 1;
            Allfiles{cnt} = fullfile(folders{f},files(j).name);
        end
    end
end
end