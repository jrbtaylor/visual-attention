-- modified from:
-- https://github.com/Element-Research/rnn/blob/master/scripts/evaluate-rva.lua
-- to run on Kaggle Cats vs Dogs

require 'nn'
require 'dp'
require 'rnn'
require 'optim'
require 'RecurrentAttentionInit'
require 'RecurrentAttentionInitAug' -- work with either one
require 'ImageScaler'
require 'Augment'

-- References :
-- A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate a Recurrent Model for Visual Attention')
cmd:text('Options:')
cmd:option('-xpPath', '', 'path to a previously saved model')
cmd:option('-cuda', false, 'model was saved with cuda')
cmd:option('-stochastic', false, 'evaluate the model stochatically. Generate glimpses stochastically')
cmd:option('-overwrite', false, 'overwrite checkpoint')
cmd:option('-path','/usr/local/data/jtaylor/Databases/Kaggle-Dogs_vs_Cats/same_size_128',
	'path to data')
cmd:option('-size','full','how many samples do we load: small | full | extended')
cmd:option('-ext','jpg','file extension to load')
cmd:text()
opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')

-- check that saved model exists
assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')

if opt.cuda then
   require 'cunn'
end

xp = torch.load(opt.xpPath)
model = xp:model().module 
tester = xp:tester() or xp:validator() -- dp.Evaluator
tester:sampler()._epoch_size = nil
conf = tester:feedback() -- dp.Confusion
cm = conf._cm -- optim.ConfusionMatrix

print("Last evaluation of "..(xp:tester() and 'test' or 'valid').." set :")
print(cm)

-- Load dataset and wrap in dp:DataSource
dofile 'LoadKCvD.lua'

ra = model:findModules('nn.RecurrentAttentionInit')[1] 
if not ra then
  ra = model:findModules('nn.RecurrentAttentionInitAug')[1]
  augment = true
else
  augment = false
end
sg = model:findModules('nn.SpatialGlimpse')[1]

-- stochastic or deterministic
for i=1,#ra.actions do
   local rn = ra.action:getStepModule(i):findModules('nn.ReinforceNormal')[1]
   rn.stochastic = opt.stochastic
end

inputs = ds:get('valid','inputs')
targets = ds:get('valid','targets', 'b')

input = inputs:narrow(1,1,math.min(10,inputs:size(1)))

-- Set model to training, otherwise RNN doesn't save intermediate time-step states
model:training()

if augment then
  aug = ra:findModules('nn.Augment')[1]
  -- Set augmentation modules to testing
  for i = 1,aug:size() do
    aug.modules[i].train = false
  end
end

if not opt.stochastic then
   for i=1,#ra.actions do
      local rn = ra.action:getStepModule(i):findModules('nn.ReinforceNormal')[1]
      rn.stdev = 0 -- deterministic
   end
end
output = model:forward(input)

function drawBox(img, bbox, channel)
    channel = channel or 1

    local x1, y1 = torch.round(bbox[1]), torch.round(bbox[2])
    local x2, y2 = torch.round(bbox[1] + bbox[3]), torch.round(bbox[2] + bbox[4])

    x1, y1 = math.max(1, x1), math.max(1, y1)
    x2, y2 = math.min(img:size(3), x2), math.min(img:size(2), y2)

    local max = img:max()

    for i=x1,x2 do
        img[channel][y1][i] = max
        img[channel][y2][i] = max
    end
    for i=y1,y2 do
        img[channel][i][x1] = max
        img[channel][i][x2] = max
    end

    return img
end

locations = ra.actions

input = nn.Convert(ds:ioShapes(),'bchw'):forward(input)
glimpses = {}
patches = {}

params = nil
for i=1,input:size(1) do
    if augment then
      img = aug.output[i]
    else
      img = input[i]
    end
   for j,location in ipairs(locations) do
      local glimpse = glimpses[j] or {}
      glimpses[j] = glimpse
      local patch = patches[j] or {}
      patches[j] = patch
      
      local xy = location[i]
      -- (-1,-1) top left corner, (1,1) bottom right corner of image
      local x, y = xy:select(1,1), xy:select(1,2)
      -- (0,0), (1,1)
      x, y = (x+1)/2, (y+1)/2
      -- (1,1), (input:size(3), input:size(4))
      x, y = x*(img:size(2)-1)+1, y*(img:size(3)-1)+1
      
      local gimg = img:clone()
      for d=1,sg.depth do
         --local size = sg.height*(sg.scale^(d-1))
         local size = sg.size*(sg.scale^(d-1))
         local bbox = {y-size/2, x-size/2, size, size}
         drawBox(gimg, bbox, 1)
      end
      glimpse[i] = gimg
      
      local sg_, ps
      if j == 1 then
         sg_ = ra.rnn.initialModule:findModules('nn.SpatialGlimpse')[1]
      else
         sg_ = ra.rnn.sharedClones[j]:findModules('nn.SpatialGlimpse')[1]
      end
      patch[i] = image.scale(img:clone():float(), sg_.output[i]:narrow(1,1,img:size(1)):float())
      
      collectgarbage()
   end
end

paths.mkdir('glimpse')
for j,glimpse in ipairs(glimpses) do
   local g = image.toDisplayTensor{input=glimpse,nrow=10,padding=3}
   local p = image.toDisplayTensor{input=patches[j],nrow=10,padding=3}
   image.save("glimpse/glimpse"..j..".png", g)
   image.save("glimpse/patch"..j..".png", p)
end