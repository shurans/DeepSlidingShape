function imdb = dss_cread_imdb(NYUonly,train_or_test,cls)
if ~any(strcmp(train_or_test, {'train', 'test','all'}))
    error('train_or_test should be either a "train" or "test" string');
end
if NYUonly
    imdbname ='NYU';
    load('./external/SUNRGBDtoolbox/traintestSUNRGBD/test_kv1NYU.mat');
    load('./external/SUNRGBDtoolbox/traintestSUNRGBD/train_kv1NYU.mat');
    if strcmp(train_or_test, 'train')
           allpath = trainSeq;
           imdb_filename = fullfile('./imdb/',['imdb_' imdbname '_train']);
    elseif strcmp(train_or_test, 'test')
           allpath = testSeq;
           imdb_filename = fullfile('./imdb/',['imdb_' imdbname '_test']);
    elseif strcmp(train_or_test, 'all')
           allpath = [testSeq,trainSeq];
           imdb_filename = fullfile('./imdb/',['imdb_' imdbname '_test']);
    end
else
    imdbname ='SUNrgbd';
    load('./external/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')
    if strcmp(train_or_test, 'train')
       allpath = alltrain;
       imdb_filename = fullfile('./imdb/',['imdb_' imdbname '_train']);
    elseif strcmp(train_or_test, 'test')
       allpath = alltest;
       imdb_filename = fullfile('./imdb/',['imdb_' imdbname '_test']);
    elseif strcmp(train_or_test, 'all')
       allpath = [alltest,alltrain];
       imdb_filename = fullfile('./imdb/',['imdb_' imdbname '_test']);
    end
end

seqnames = cell(1,length(allpath));
for i =1:length(allpath)
    seqnames{i} = getSequenceName(allpath{i});
end
imdb = [];
imdb.name = imdbname;
imdb.classes = cls;
imdb.class_ids = 1 : length(imdb.classes);
imdb.image_ids = seqnames;

end