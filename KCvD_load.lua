----------------------------------------------------------------------
-- This script loads the Kaggle Dogs vs Cats dataset
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Options:')
	cmd:option('-path','/usr/local/data/jtaylor/Databases/Kaggle-Dogs_vs_Cats/same_size_128',
		'path to data')
	cmd:option('-val',0.04,'portion of training data for holdout validation')
   cmd:option('-size', 'full', 'how many samples do we load: small | full | extended')
	cmd:option('-ext','jpg','file extension to load')
   cmd:text()
   opt = cmd:parse(arg or {})
end

opt.noTest = opt.noTest or false
opt.val = opt.val or 0.04
opt.ext = opt.ext or 'jpg'
opt.path = opt.path or '/usr/local/data/jtaylor/Databases/Kaggle-Dogs_vs_Cats/same_size_128'

torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
-- train/test size
if opt.size == 'small' then
	print '==> loading reduced train & test sets for speed'
	trsize = 200--500
	tesize = 20--250
else
	print '==> loading full train & test sets'
	trsize = 25000
	tesize = 12500
end

----------------------------------------------------------------------
-- store file names and training labels in tables

-- training:
trfiles = {}
trlabels = {}
trpath = opt.path .. '/train'
for file in paths.files(trpath) do
	-- find files w/ matching extension
	if file:find(opt.ext) then
	-- *note: hardcoded class names
		if file:sub(1,3) == 'dog' then
			table.insert(trlabels,2)
		else
			table.insert(trlabels,1)
		end
		-- table.insert(trlabels,file:sub(1,3))
		table.insert(trfiles, paths.concat(trpath,file))
	end
end

-- test:
if not opt.noTest then
  tefiles = {}
  tepath = opt.path .. '/test'
  for file in paths.files(tepath) do
    -- find files w/ matching extension
    if file:find(opt.ext) then
      table.insert(tefiles, paths.concat(tepath,file))
    end
  end
end

if #trfiles == 0 then
	error('given directory doesnt contain /train and/or /test sub-folders w/ images')
end

----------------------------------------------------------------------
-- load images

trimages = torch.Tensor(trsize,3,128,128)
for i,file in ipairs(trfiles) do
	trimages[i] = image.load(file)
	if i==trsize then
		break
	end
end
if not opt.noTest then
  teimages = torch.Tensor(tesize,3,128,128)
  teind = {}
  for i,file in ipairs(tefiles) do
    local fileind = string.gsub(string.gsub(string.gsub(file,'.jpg',''),tepath,''),'/','')
    teind[#teind+1] = tonumber(fileind)
    teimages[i] = image.load(file)
    if i==tesize then
      break
    end
  end
end

----------------------------------------------------------------------
-- randomly assign to validation set

shuffle = torch.randperm(trsize)
vlsize = math.floor(trsize*opt.val)
vlind = shuffle[{{1,vlsize}}]
trind = shuffle[{{vlsize+1,trsize}}]
trsize = trsize-vlsize

if opt.size=='extended' then
	vlimages = torch.Tensor(2*vlsize,3,128,128)
	vllabels = torch.Tensor(2*vlsize)
	trimages2 = torch.Tensor(2*trsize,3,128,128)
	trlabels2 = torch.Tensor(2*trsize)
else
	vlimages = torch.Tensor(vlsize,3,128,128)
	vllabels = torch.Tensor(vlsize)
	trimages2 = torch.Tensor(trsize,3,128,128)
	trlabels2 = torch.Tensor(trsize)
end

for i=1,vlsize do 
	vlimages[{i,{},{},{}}] = trimages[{vlind[i],{},{},{}}]
	vllabels[i] = trlabels[vlind[i]]
end

for i=1,trsize do
	trimages2[{i,{},{},{}}] = trimages[{trind[i],{},{},{}}]
	trlabels2[i] = trlabels[trind[i]]
end
-- note: the above :apply(function()... copies the tensors
--			so the originals should be cleared for memory cleanup
trimages = nil
trlabels = nil

----------------------------------------------------------------------
-- generate horizontally flipped copies for extended training set
if opt.size=='extended' then
	print('==> generating horizontally-flipped copies for extended training')
	for i=1,vlsize do
		vlimages[{vlsize+i,{},{},{}}] = image.hflip(vlimages[{i,{},{},{}}])
		vllabels[vlsize+i] = vllabels[i]
	end
	for i=1,trsize do
		trimages2[{trsize+i,{},{},{}}] = image.hflip(trimages2[{i,{},{},{}}])
		trlabels2[trsize+i] = trlabels2[i]
	end
	vlsize = 2*vlsize
	trsize = 2*trsize
end

----------------------------------------------------------------------
-- data structures

valData = {
	data = vlimages,
	labels = vllabels,
	size = function() return vlsize end
}

trainData = {
	data = trimages2,
	labels = trlabels2,
	size = function() return trsize end
}

if not opt.noTest then
  testData = {
     data = teimages,
     size = function() return tesize end
  }
end

----------------------------------------------------------------------
-- Preprocessing
-- 1. images are mapped into YUV space, to separate luminance & color
-- 2. color channels are normalized globally; each color component 
--    has 0-mean and 1-norm across the dataset.
-- 3. luminance channel (Y) is locally normalized: for each 
--    neighborhood, defined by a Gaussian kernel, the mean is 
--    suppressed and the standard deviation is normalized to 1


print '==> preprocessing data'

-- Preprocessing requires a floating point representation
valData.data = valData.data:float()
trainData.data = trainData.data:float()
if not opt.noTest then
  testData.data = testData.data:float()
end

-- 1. Convert all images to YUV
print '==> preprocessing data: colorspace RGB -> YUV'
for i = 1,valData:size() do
	valData.data[i] = image.rgb2yuv(valData.data[i])
end
for i = 1,trainData:size() do
   trainData.data[i] = image.rgb2yuv(trainData.data[i])
end
if not opt.noTest then
  for i = 1,testData:size() do
     testData.data[i] = image.rgb2yuv(testData.data[i])
  end
end
channels = {'y','u','v'} -- name channels for convenience

-- 2. Normalize each channel, and store mean/std per channel
print '==> preprocessing data: normalize each channel globally'
mean = {}
std = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   valData.data[{ {},i,{},{} }]:add(-mean[i])
   valData.data[{ {},i,{},{} }]:div(std[i])
   if not opt.noTest then
     testData.data[{ {},i,{},{} }]:add(-mean[i])
     testData.data[{ {},i,{},{} }]:div(std[i])
    end
end

-- 3. Local normalization
print '==> preprocessing data: normalize all three channels locally'

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
if opt.type=='cuda' then
	neighborhood = image.gaussian({size=27,sigma=9}) -- was 15,3, then 27,9
	normalization = nn.SpatialContrastiveNormalization(1,neighborhood,1):cuda()
	function norm(x) x=x:cuda(); y = normalization:forward(x); return y:float() end
else
	neighborhood = image.gaussian1D(15) -- was size=13 in tutorial, but for small images
	normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
	function norm(x) y = normalization:forward(x); return y end
end	

-- Normalize all channels locally:
for c in ipairs(channels) do
	for i = 1,valData:size() do
		valData.data[{ i,{c},{},{} }] = norm(valData.data[{ i,{c},{},{} }])
	end
   for i = 1,trainData:size() do
      trainData.data[{ i,{c},{},{} }] = norm(trainData.data[{ i,{c},{},{} }])
   end
   if not opt.noTest then
     for i = 1,testData:size() do
        testData.data[{ i,{c},{},{} }] = norm(testData.data[{ i,{c},{},{} }])
     end
  end
end

-- Verify statistics
print '==> verify statistics'

for i,channel in ipairs(channels) do
   trainMean = trainData.data[{ {},i }]:mean()
   trainStd = trainData.data[{ {},i }]:std()

   print('training data, '..channel..'-channel, mean: ' .. trainMean)
   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

end

-- Convert back to original intended Tensor type
if opt.type == 'double' then
	print '==> switching back to double'
	valData.data = valData.data:double()
	trainData.data = trainData.data:double()
  if not opt.noTest then
    testData.data = testData.data:double()
  end
end







