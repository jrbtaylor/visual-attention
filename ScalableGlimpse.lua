------------------------------------------------------------------------
--[[ ScalableGlimpse ]]--
-- a glimpse is a concatenation of scaled cropped patches
-- around a given location in an image.
-- input is a pair of tensors and a scalar: {image, location, scale}
-- locations are x,y coordinates of the center of the cropped patch
-- coordinates are between -1,-1 (top-left) and 1,1 (bottom-right)
-- scale scales the smallest glimpse patch (e.g. 1 = original res)
-- constructor arguments are the base glimpse size (height=width),
-- the depth (number of patches/scales), and the scale between patches
--
-- based on dpnn.SpatialGlimpse by nicholas-leonard
-- github.com/nicholas-leonard/dpnn/blob/master/SpatialGlimpse.lua
-- the motivation is for the agent to learn to attend to objects/regions
-- at varying scales, including initializing itself appropriately
------------------------------------------------------------------------
local ScalableGlimpse, parent = torch.class("nn.ScalableGlimpse","nn.Module")

function ScalableGlimpse:__init(size,depth,scale)
	require 'nnx'
	self.size = size
	self.depth = depth or 3
	self.scale = scale or 2

	assert(torch.type(self.size) == 'number')
	assert(torch.type(self.depth) == 'number')
	assert(torch.type(self.scale) == 'number')
	parent.__init(self)
	self.gradInput = {torch.Tensor(), torch.Tensor()}
	if self.scale == 2 then
		self.module = nn.SpatialAveragePooling(2,2,2,2)
	else
		self.module = nn.SpatialReSampling{oheight=size,owidth=size}
	end
	self.modules = {self.module}
end

function ScalableGlimpse:updateOutput(inputTable)
  assert(torch.type(inputTable) == 'table')
  assert(#inputTable >= 2)
  local input, location = unpack(inputTable)
  input, location = self:toBatch(input,3), self:toBatch(location,1)
  assert(input:dim() == 4 and location:dim() == 2)

  self.output:resize(input:size(1), self.depth, input:size(2), self.size, self.size)

  self._crop = self._crop or self.output.new()
  self._pad = self._pad or input.new()

  for sampleIdx=1,self.output:size(1) do
    local outputSample = self.output[sampleIdx]
    local inputSample = input[sampleIdx]
    local xys = location[sampleIdx]
    -- (-1,-1) top left corner, (1,1) bottom right corner of image
    local x, y, sc = xys:select(1,1), xys:select(1,2), xys:select(1,3)
    -- (0,0), (1,1)
    x, y = (x+1)/2, (y+1)/2
    -- scale from 1 to 6
    sc = (sc+1)*3

    -- for each depth of glimpse : pad, crop, downscale
    local glimpseSize = self.size
    for depth=1,self.depth do
      local dst = outputSample[depth]
      if depth > 1 then
          glimpseSize = glimpseSize*self.scale
      end

      -- add zero padding (glimpse could be partially out of bounds)
      local padSize = math.floor((sc*glimpseSize-1)/2)
      self._pad:resize(input:size(2), input:size(3)+padSize*2, input:size(4)+padSize*2):zero()
      local center = self._pad:narrow(2,padSize+1,input:size(3)):narrow(3,padSize+1,input:size(4))
      center:copy(inputSample)

      -- crop it
      local h, w = self._pad:size(2)-sc*glimpseSize, self._pad:size(3)-sc*glimpseSize
      local x, y = math.min(h,math.max(0,x*h)),  math.min(w,math.max(0,y*w))

      if depth == 1 and sc == 1 then
        dst:copy(self._pad:narrow(2,x+1,glimpseSize):narrow(3,y+1,glimpseSize))
      else
        self._crop:resize(input:size(2), sc*glimpseSize, sc*glimpseSize)
        self._crop:copy(self._pad:narrow(2,x+1,sc*glimpseSize):narrow(3,y+1,sc*glimpseSize))
        dst:copy(image.scale(self._crop,self.size,self.size,'simple'))
      end
    end
  end

  self.output:resize(input:size(1), self.depth*input:size(2), self.size, self.size)
  self.output = self:fromBatch(self.output, 1)
  return self.output
end

function ScalableGlimpse:updateGradInput(inputTable, gradOutput)
   local input, location = unpack(inputTable)
   local gradInput, gradLocation = unpack(self.gradInput)
   input, location = self:toBatch(input, 3), self:toBatch(location, 1)
   gradOutput = self:toBatch(gradOutput, 3)

   gradInput:resizeAs(input):zero()
   gradLocation:resizeAs(location):zero() -- no backprop through location

   gradOutput = gradOutput:view(input:size(1), self.depth, input:size(2), self.size, self.size)

   for sampleIdx=1,gradOutput:size(1) do
      local gradOutputSample = gradOutput[sampleIdx]
      local gradInputSample = gradInput[sampleIdx]
      local xys = location[sampleIdx] -- height, width, scale
      -- (-1,-1) top left corner, (1,1) bottom right corner of image
      local x, y, sc = xys:select(1,1), xys:select(1,2), xys:select(1,3)
      -- (0,0), (1,1)
      x, y = (x+1)/2, (y+1)/2
      -- scale from 1 to 6
      sc = (sc+1)*3
      -- somehow still getting errors where sc<1...
      sc = math.max(sc,1)

      -- for each depth of glimpse : pad, crop, downscale
      local glimpseSize = self.size
      for depth=1,self.depth do
         local src = gradOutputSample[depth]
         if depth > 1 then
            glimpseSize = glimpseSize*self.scale
         end

         -- add zero padding (glimpse could be partially out of bounds)
         local padSize = math.floor((sc*glimpseSize-1)/2)
         self._pad:resize(input:size(2), input:size(3)+padSize*2, input:size(4)+padSize*2):zero()

         local h, w = self._pad:size(2)-sc*glimpseSize, self._pad:size(3)-sc*glimpseSize
         local x, y = math.min(h,math.max(0,x*h)),  math.min(w,math.max(0,y*w))
         local pad = self._pad:narrow(2, x+1, self.size):narrow(3, y+1, self.size)

         -- upscale glimpse for different depths
         if depth == 1 and sc == 1 then
            pad:copy(src)
         else
            self._crop:resize(input:size(2), sc*glimpseSize, sc*glimpseSize)
            pad:copy(image.scale(self._crop,self.size,self.size,'simple'))
         end

         -- copy into gradInput tensor (excluding padding)
         gradInputSample:add(self._pad:narrow(2, padSize+1, input:size(3)):narrow(3, padSize+1, input:size(4)))
      end
   end

   self.gradInput[1] = self:fromBatch(gradInput, 1)
   self.gradInput[2] = self:fromBatch(gradLocation, 1)

   return self.gradInput
end
