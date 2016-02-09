function downloadData(path2save,urlroot,filetype)
         if ~exist('filetype','var')
            filetype = '.mat';
         end
         if ~exist('urlroot','var')
            urlroot = ' http://dss.cs.princeton.edu/Release/sunrgbd_dss_data/';
         end
         load('./external/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat');
       
         for i = 1:length(SUNRGBDMeta)
             sequenceName = SUNRGBDMeta(i).sequenceName;
             src_fullpath = fullfile(urlroot,[sequenceName filetype]);
             dest_fullpath = [fullfile(path2save,sequenceName) filetype];
             ind = find(dest_fullpath=='/');
             mkdir(dest_fullpath(1:ind(end)-1));
             cmd = sprintf('/usr/local/bin/wget -O %s %s', dest_fullpath,src_fullpath);
             system(cmd);
         end
end


% example usage: imageFiles = dirSmart(fullfile(...),'jpg');
function files = dirSmart(page, tag)
  [files, status] = urldir(page, tag);
    if status == 0
      files = dir(fullfile(page, ['*.' tag]));
    end
end

function [files, status] = urldir(page, tag)
    if nargin == 1
      tag = '/';
    else
        tag = lower(tag);
        if strcmp(tag, 'dir')
            tag = '/';
        end
        if strcmp(tag, 'img')
	tag = 'jpg';
        end
    end
    nl = length(tag);
    nfiles = 0;
    files = [];

    % Read page
    page = strrep(page, '\', '/');
    [webpage, status] = urlread(page);

    if status
        % Parse page
        j1 = findstr(lower(webpage), '<a href="');
        j2 = findstr(lower(webpage), '</a>');
        Nelements = length(j1);
        if Nelements>0
            for f = 1:Nelements
                % get HREF element
                chain = webpage(j1(f):j2(f));
                jc = findstr(lower(chain), '">');
                chain = deblank(chain(10:jc(1)-1));

                % check if it is the right type
                if length(chain)>length(tag)-1
                    if strcmp(chain(end-nl+1:end), tag)
                        nfiles = nfiles+1;
                        chain = strrep(chain, '%20', ' '); % replace space character
                        files(nfiles).name = chain;
                        files(nfiles).bytes = 1;
                    end
                end
            end
        end
    end
end