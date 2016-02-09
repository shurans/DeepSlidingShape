function tensors = readTensor(filename)

% use this function to load .tensor files (features or weights) into Matlab

count = 0;

fp = fopen(filename, 'rb');

while ~feof(fp)
	count = count + 1;
	lenName=fread(fp,1,'int32');
	if feof(fp)
		break;
	end
	if lenName>0
		str=char(fread(fp,lenName,'char*1')');
		disp(str);
	else
		str='';
	end
	nbDims=fread(fp,1,'int32');
	if nbDims==0 % maybe zero paddings at the end of a feature file
		break;
	end
	dim = fread(fp,nbDims,'int32');
	dim = dim(:)';
	values = single(fread(fp,prod(dim),'float'));
	if nbDims>1
		values = reshape(values,dim(end:-1:1));
	end
	tensors(count).name = str;
	tensors(count).value = values;
end

fclose(fp);

