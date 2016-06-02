-- Load & preprocess data
--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if opt.size == 'full' then
  if paths.filep(opt.path .. '/val.t7') then
    print('... Loading pre-enhanced dataset tensors')
    valData = torch.load(opt.path .. '/val.t7','ascii')
    trainData = torch.load(opt.path .. '/train.t7','ascii')
    if not opt.noTest then
      testData = torch.load(opt.path .. '/test.t7','ascii')
    end
  else
    dofile '/home/rain/jtaylor/Documents/Torch_Workspace/visatt/KCvD_load.lua'
    torch.save(opt.path .. '/val.t7',valData,'ascii')
    torch.save(opt.path .. '/train.t7',trainData,'ascii')
    if not opt.noTest then
      torch.save(opt.path .. '/test.t7',testData,'ascii')
    end
  end
elseif opt.size == 'small' then
  if paths.filep(opt.path .. '/val_small.t7') then
    print('... Loading pre-enhanced dataset tensors')
    valData = torch.load(opt.path .. '/val_small.t7','ascii')
    trainData = torch.load(opt.path .. '/train_small.t7','ascii')
    if not opt.noTest then
      testData = torch.load(opt.path .. '/test_small.t7','ascii')
    end
  else
    dofile '/home/rain/jtaylor/Documents/Torch_Workspace/visatt/KCvD_load.lua'
    torch.save(opt.path .. '/val_small.t7',valData,'ascii')
    torch.save(opt.path .. '/train_small.t7',trainData,'ascii')
    if not opt.noTest then
      torch.save(opt.path .. '/test_small.t7',testData,'ascii')
    end
  end
elseif opt.size == 'extended' then
  if paths.filep(opt.path .. '/val_ext.t7') then
    print('... Loading pre-enhanced dataset tensors')
    valData = torch.load(opt.path .. '/val_ext.t7','ascii')
    trainData = torch.load(opt.path .. '/train_ext.t7','ascii')
    if not opt.noTest then
      testData = torch.load(opt.path .. '/test_ext.t7','ascii')
    end
  else
    dofile '/home/rain/jtaylor/Documents/Torch_Workspace/visatt/KCvD_load.lua'
    torch.save(opt.path .. '/val_ext.t7',valData,'ascii')
    torch.save(opt.path .. '/train_ext.t7',trainData,'ascii')
    if not opt.noTest then
      torch.save(opt.path .. '/test_ext.t7',testData,'ascii')
    end
  end
end

print('==> YUV -> RGB')
for i = 1,valData.data:size(1) do
	valData.data[i] = image.yuv2rgb(valData.data[i])
end
for i = 1,trainData.data:size(1) do
   trainData.data[i] = image.yuv2rgb(trainData.data[i])
end
if not opt.noTest then
  for i = 1,testData.data:size(1) do
     testData.data[i] = image.yuv2rgb(testData.data[i])
  end
end
imH = trainData.data:size(3)
imW = trainData.data:size(4)


-- Wrap data into a dp:dataSource
--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print('==> wrap as dp:dataSource')
trainInput = dp.ImageView('bchw', trainData.data:cuda())
trainTarget = dp.ClassView('b', trainData.labels)
validInput = dp.ImageView('bchw', valData.data:cuda())
validTarget = dp.ClassView('b', valData.labels)

trainTarget:setClasses({'dog', 'cat'})
validTarget:setClasses({'dog', 'cat'})

train = dp.DataSet{inputs=trainInput,targets=trainTarget,which_set='train'}
valid = dp.DataSet{inputs=validInput,targets=validTarget,which_set='valid'}

ds = dp.DataSource{train_set=train,valid_set=valid}
ds:classes{'dog', 'cat'}