-- Run an RNN-based visual attention mechanism on the Kaggle Cats vs Dogs dataset
-- Note: uses the 2-layer RNN model from "Multiple Object Recognition with Visual Attention"
--       with data augmentation

-- Preliminaries
--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
require 'nn'
require 'image'
require 'rnn'
require 'dp'
require 'cutorch'
require 'math'
require 'RecurrentAttentionInitAug'
require 'ImageScaler'
require 'debugger'
require 'Augment'


-- Terminal args
--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Recurrent Model for Visual Attention')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-learningRate', 0.01, 'learning rate at t=0')
cmd:option('-minLR', 0.00001, 'minimum learning rate')
cmd:option('-saturateEpoch', 800, 'epoch at which linear decayed LR will reach minLR')
cmd:option('-momentum', 0.9, 'momentum')
cmd:option('-maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('-cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
cmd:option('-batchSize', 40, 'number of examples per batch')
cmd:option('-cuda', false, 'use CUDA')
cmd:option('-useDevice', 1, 'sets the device (GPU) to use')
cmd:option('-maxEpoch', 2000, 'maximum number of epochs to run')
cmd:option('-maxTries', 100, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('-transfer', 'ReLU', 'activation function')
cmd:option('-uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
-- data:
cmd:option('-path','/usr/local/data/jtaylor/Databases/Kaggle-Dogs_vs_Cats/same_size_128',
	'path to data')
cmd:option('-val',0.04,'portion of training data for holdout validation')
cmd:option('-ext','jpg','file extension to load')
cmd:option('-size','full','how many samples do we load: small | full | extended')
cmd:option('-noTest','true','skip the test set')
cmd:option('-cropSize',100,'random crop to take for data augmentation')
-- glimpse:
cmd:option('-scales',1,'number of scales in scale-space')
cmd:option('-glimpsePatchSize', 25, 'size of glimpse patch at highest res (height = width)')
cmd:option('-glimpseScale', 2, 'scale of successive patches w.r.t. original input image')
cmd:option('-glimpseDepth', 2, 'number of concatenated downscaled patches')
cmd:option('-locatorHiddenSize',64,'size of locator hidden layer')
cmd:option('-glimpseHiddenSize', 256, 'size of glimpse hidden layer')
cmd:option('-inputHiddenSize', 256, 'size of combined glimpse + location input to RNN')
cmd:option('-dropout', 0.3, 'dropout rate: use zero for no dropout')
-- RNN
cmd:option('-rho', 3, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('-r1HiddenSize', 256, 'size of rnn1 hidden layer')
cmd:option('-r2HiddenSize', 128, 'size of rnn2 hidden layer')
cmd:option('-fastLSTM', false, 'use LSTM in RNN layers')
-- reinforce
cmd:option('-rewardScale', 1, "scale of positive reward (negative is 0)")
cmd:option('-locatorStd', 0.1, 'stdev of gaussian location sampler (between 0 and 1) (low values may cause NaNs)')
cmd:option('-stochastic', false, 'Reinforce modules forward inputs stochastically during evaluation')

opt = cmd:parse(arg or {})

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')


-- Load dataset and wrap in dp:DataSource
--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dofile 'LoadKCvD.lua'


-- Load pre-trained convnet for glimpse network
--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pretrain = false
if pretrain then
  print('==> loading pre-trained convnet')
  require 'loadcaffe'
  require 'matio'
  modeldir = '/usr/local/data/jtaylor/Pretrained_Nets/CNN_M_128/'
  prototxt = modeldir .. 'VGG_mean.binaryproto'
  binary = modeldir .. 'VGG_CNN_M_128.caffemodel'
  imnetCNN = loadcaffe.load(prototxt,binary)
  -- note: model takes 224x224 BGR crops; chop final layers to use a smaller window
  remove = 10 -- final layers to remove
  for l = 1,remove do
    imnetCNN:remove()
  end

  -- overload accGradParameters to fix the weights (~30% faster than not fixing them)
  imnetCNN.accGradParameters = function(self) end
  
  -- wrap in sequential()
  glimpseCNN = nn.Sequential()
  glimpseCNN:add(imnetCNN)
  -- could try adding dropout here too...
  glimpseCNN:add(nn.Collapse(3))
  glimpseCNN:add(nn.Linear(512,opt.glimpseHiddenSize))
  glimpseCNN:add(nn[opt.transfer]())
else
  glimpseCNN = nn.Sequential() --25x25
  glimpseCNN:add(nn.SpatialConvolution(3*opt.glimpseDepth,32,3,3)) --23x23
  glimpseCNN:add(nn[opt.transfer]())
  glimpseCNN:add(nn.SpatialConvolution(32,64,3,3)) --21x21
  glimpseCNN:add(nn[opt.transfer]())
  glimpseCNN:add(nn.SpatialMaxPooling(2,2,2,2)) --10x10
  glimpseCNN:add(nn.SpatialConvolution(64,64,3,3)) --8x8
  glimpseCNN:add(nn[opt.transfer]())
  --glimpseCNN:add(nn.SpatialMaxPooling(2,2,2,2)) --4x4
  glimpseCNN:add(nn.Collapse(3))
  glimpseCNN:add(nn.Linear(64*8*8,opt.glimpseHiddenSize))
end

-- Define the model
--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print('==> defining the network')

-- data augmentation network
augmentNet = nn.Augment() -- THIS IS A SEQUENTIAL CONTAINER; RENAMED TO FIND IN VISUALIZE POLICY
augmentNet:add(nn.SpatialUniformCrop(opt.cropSize,opt.cropSize))
print(augmentNet)

-- location network (inputs location)
if opt.scales > 1 then locdim = 3 else locdim = 2 end
locationNet = nn.Sequential()
--locationNet:add(nn.debugger('locationNet input'))
locationNet:add(nn.SelectTable(2))
locationNet:add(nn.Linear(locdim, opt.locatorHiddenSize))
locationNet:add(nn[opt.transfer]())

-- glimpse network
glimpseNet = nn.Sequential()
--glimpseNet:add(nn.debugger('glimpseNet input'))
glimpseNet:add(nn.DontCast(nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale):float(),true))
glimpseNet:add(glimpseCNN)

-- input network (combines location and glimpse)
inputNet = nn.Sequential()
inputNet:add(nn.ConcatTable():add(locationNet):add(glimpseNet))
inputNet:add(nn.JoinTable(1,1))
inputNet:add(nn.Linear(opt.glimpseHiddenSize+opt.locatorHiddenSize, opt.inputHiddenSize))
inputNet:add(nn[opt.transfer]())
inputNet:add(nn.Linear(opt.inputHiddenSize, opt.r1HiddenSize))

-- initializing network (downsampled full image)
initialNet = nn.Sequential()
initialNet:add(nn.ImageScaler(opt.glimpsePatchSize,opt.glimpsePatchSize))
initialNet:add(nn.Collapse(3))
initialNet:add(nn.Linear(3*(opt.glimpsePatchSize^2),opt.glimpseHiddenSize))
initialNet:add(nn[opt.transfer]())
initialNet:add(nn.Linear(opt.glimpseHiddenSize, opt.inputHiddenSize))
initialNet:add(nn[opt.transfer]())
initialNet:add(nn.Linear(opt.inputHiddenSize, opt.r1HiddenSize))

-- RNN1: classification representation
r1Hidden = nn.Sequential()
if opt.fastLSTM then
  r1Hidden:add(nn.FastLSTM(opt.r1HiddenSize,opt.r1HiddenSize))
else
  r1Hidden:add(nn.Linear(opt.r1HiddenSize, opt.r1HiddenSize))
end
r1Hidden:add(nn.Dropout(opt.dropout))
r1Net = nn.Recurrent(opt.r1HiddenSize, inputNet, r1Hidden, nn[opt.transfer](), opt.rho)

-- Intra-RNN net (one layer from size r1Hidden to size r2Hidden w/ dropout)
r1to2Net = nn.Sequential()
r1to2Net:add(nn.Linear(opt.r1HiddenSize,opt.r2HiddenSize))
r1to2Net:add(nn[opt.transfer]())
r1to2Net:add(nn.Dropout(opt.dropout))

-- RNN2: attention policy representation
r2Hidden = nn.Sequential()
if opt.fastLSTM then
  r2Hidden:add(nn.FastLSTM(opt.r2HiddenSize,opt.r2HiddenSize))
else
  r2Hidden:add(nn.Linear(opt.r2HiddenSize, opt.r2HiddenSize))
end
r2Hidden:add(nn.Dropout(opt.dropout))
r2Net = nn.Recurrent(opt.r2HiddenSize, r1to2Net, r2Hidden, nn[opt.transfer](), opt.rho)

-- locator (agent output)
actionNet = nn.Sequential()
actionNet:add(r2Net)
actionNet:add(nn.Linear(opt.r2HiddenSize,locdim))
actionNet:add(nn.HardTanh()) -- bounds to [-1,1]
actionNet:add(nn.ReinforceNormal(2*opt.locatorStd, opt.stochastic))
actionNet:add(nn.HardTanh())
actionNet:add(nn.MulConstant((math.max(imW,imH)-opt.glimpsePatchSize/2)/math.max(imW,imH)))

-- recurrent attention
attention = nn.RecurrentAttentionInitAug(initialNet, r1Net, actionNet, augmentNet, opt.rho)

-- predictor
agent = nn.Sequential()
agent:add(attention)
agent:add(nn.SelectTable(-1)) -- input is full history of hidden states, this selects the last one
agent:add(nn.Dropout(opt.dropout))
agent:add(nn.Linear(opt.r1HiddenSize,2)) -- 2 classes
agent:add(nn.LogSoftMax())

-- baseline reward
seq = nn.Sequential()
seq:add(nn.Constant(0.5,1))
seq:add(nn.Add(1))
concat = nn.ConcatTable():add(nn.Identity()):add(seq)
concat2 = nn.ConcatTable():add(nn.Identity()):add(concat)

-- output will be : {classpred, {classpred, basereward}}
-- NLL criterion requires ^
-- VRClassReward requires         ^Table of classpred, basereward
-- NLL criterion backpropagates like a normal NN
-- VRClass reward is broadcast to Reinforce modules
agent:add(concat2)

-- initialize parameters
if opt.uniform > 0 then
   for k,param in ipairs(agent:parameters()) do
      param:uniform(-opt.uniform, opt.uniform)
   end
end


-- Training and Validation setup
--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print('==> training setup')

opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch

train = dp.Optimizer{
   loss = nn.ParallelCriterion(true)
      :add(nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert())) -- BACKPROP
      :add(nn.ModuleCriterion(nn.VRClassReward(agent, opt.rewardScale), nil, nn.Convert())) -- REINFORCE
   ,
   epoch_callback = function(model, report) -- called every epoch
      if report.epoch > 0 then
         opt.learningRate = opt.learningRate + opt.decayFactor
         opt.learningRate = math.max(opt.minLR, opt.learningRate)
         if not opt.silent then
            print("learningRate", opt.learningRate)
         end
      end
   end,
   callback = function(model, report)       
      if opt.cutoffNorm > 0 then
         local norm = model:gradParamClip(opt.cutoffNorm) -- affects gradParams
         opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
         if opt.lastEpoch < report.epoch and not opt.silent then
            print("mean gradParam norm", opt.meanNorm)
         end
      end
      model:updateGradParameters(opt.momentum) -- affects gradParams
      model:updateParameters(opt.learningRate) -- affects params
      model:maxParamNorm(opt.maxOutNorm) -- affects params
      model:zeroGradParameters() -- affects gradParams 
   end,
   feedback = dp.Confusion{output_module=nn.SelectTable(1)},  
   sampler = dp.ShuffleSampler{
      epoch_size = opt.trainEpochSize, batch_size = opt.batchSize
   },
   progress = opt.progress
}

valid = dp.Evaluator{
   feedback = dp.Confusion{output_module=nn.SelectTable(1)},  
   sampler = dp.Sampler{epoch_size = opt.validEpochSize, batch_size = opt.batchSize},
   progress = opt.progress
}
if not opt.noTest then
   tester = dp.Evaluator{
      feedback = dp.Confusion{output_module=nn.SelectTable(1)},  
      sampler = dp.Sampler{batch_size = opt.batchSize} 
   }
end


-- Experiment
--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xp = dp.Experiment{
   model = agent,
   optimizer = train,
   validator = valid,
   tester = tester,
   observer = {
      ad,
      dp.FileLogger(),
      dp.EarlyStopper{
         max_epochs = opt.maxTries, 
         error_report={'validator','feedback','confusion','accuracy'},
         maximize = true
      }
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}

--[[GPU or CPU]]--
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
   xp:cuda()
end

xp:verbose(not opt.silent)
if not opt.silent then
   print"Agent :"
   print(agent)
end

xp.opt = opt

xp:run(ds)




