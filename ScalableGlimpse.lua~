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
	assert(#inputTable >= 3)
	local input, location, scale = unpack(inputTable)
	input, location, scale = self:toBatch(input,3), self:toBatch(location,1), self:toBatch(scale,1)
	assert(input:dim() == 4 and location:dim() == 2 and scale:dim() == 1)

    self.output:resize(input:size(1), self.depth, input:size(2), self.size, self.size)

    self._crop = self._crop or self.output.new()
    self._pad = self._pad or input.new()

    for sampleIdx=1,self.output:size(1) do
        local outputSample = self.output[sampleIdx]
        local inputSample = input[sampleIdx]
        local xy = location[sampleIdx]
        -- (-1,-1) top left corner, (1,1) bottom right corner of image
        local x, y = xy:select(1,1), xy:select(1,2)
        -- (0,0), (1,1)
        x, y = (x+1)/2, (y+1)/2
        local sc = scale[sampleIdx]

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
				
				if sc == 1 and torch.type(self.module) == 'nn.SpatialAveragePooling' then
					local poolSize = glimpseSize/self.size
					assert(poolSize % 2 == 0)
					self.module.kW = poolSize
					self.module.kH = poolSize
					self.module.dW = poolSize
					self.module.dH = poolSize
					dst:copy(self.module:updateOutput(self._crop))
				else
					dst:copy(nn.SpatialReSampling{oheight=self.size, owidth=self.size}:forward(self.crop))
				end
			end
		end
	end

   self.output:resize(input:size(1), self.depth*input:size(2), self.size, self.size)
   self.output = self:fromBatch(self.output, 1)
   return self.output
end

function SpatialGlimpse:updateGradInput(inputTable, gradOutput)
   local input, location, scale = unpack(inputTable)
   local gradInput, gradLocation, gradScale = unpack(self.gradInput)
   input, location, scale = self:toBatch(input, 3), self:toBatch(location, 1), self:toBatch(scale, 1)
   gradOutput = self:toBatch(gradOutput, 3)

   gradInput:resizeAs(input):zero()
   gradLocation:resizeAs(location):zero() -- no backprop through location
   gradScale:resizeAs(scale):zero() -- no backprop through scale

   gradOutput = gradOutput:view(input:size(1), self.depth, input:size(2), self.size, self.size)

   for sampleIdx=1,gradOutput:size(1) do
      local gradOutputSample = gradOutput[sampleIdx]
      local gradInputSample = gradInput[sampleIdx]
      local xy = location[sampleIdx] -- height, width
      -- (-1,-1) top left corner, (1,1) bottom right corner of image
      local x, y = xy:select(1,1), xy:select(1,2)
      -- (0,0), (1,1)
      x, y = (x+1)/2, (y+1)/2
	  local sc = scale[sampleIdx]

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
         local pad = self._pad:narrow(2, x+1, sc*glimpseSize):narrow(3, y+1, sc*glimpseSize)

         -- upscale glimpse for different depths
         if depth == 1 and sc == 1 then
            pad:copy(src)
         else
            self._crop:resize(input:size(2), sc*glimpseSize, sc*glimpseSize)

			if sc == 1 and torch.type(self.module) == 'nn.SpatialAveragePooling' then
               local poolSize = glimpseSize/self.size
               assert(poolSize % 2 == 0)
               self.module.kW = poolSize
               self.module.kH = poolSize
               self.module.dW = poolSize
               self.module.dH = poolSize
			   pad:copy(self.module:updateGradInput(self._crop, src))
			else
			   pad:copy(nn.SpatialReSampling{oheight=self.size, owidth=self.size}:updateGradInput(self.crop, src))
            end
         end

         -- copy into gradInput tensor (excluding padding)
         gradInputSample:add(self._pad:narrow(2, padSize+1, input:size(3)):narrow(3, padSize+1, input:size(4)))
      end
   end

   self.gradInput[1] = self:fromBatch(gradInput, 1)
   self.gradInput[2] = self:fromBatch(gradLocation, 1)

   return self.gradInput
end
