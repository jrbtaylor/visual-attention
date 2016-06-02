local ImageScaler, parent = torch.class('nn.ImageScaler', 'nn.Module')
require 'nnx'
--local cv = require 'cv'
--require 'cv.cudawarping' -- cv.imgproc on CPU, cv.cudawarping on GPU

function ImageScaler:__init(rows,cols)
  parent.__init(self)
  self.rows = rows
  self.cols = cols
  self._input = torch.Tensor()
  self._gradOutput = torch.Tensor()
end

function ImageScaler:updateOutput(input)
  
  -- ~580 examples/second
  resizer = nn.SpatialReSampling{owidth=self.cols,oheight=self.rows}
  self.output = resizer:forward(input:float())
  return self.output:cuda()
  
--  ~400 examples/second  
--  out = torch.Tensor(input:size(1),input:size(2),self.rows,self.cols):cuda()
--  for b=1,input:size(1) do
--    -- opencv requires images in [h,w,c] instead of [b,c,h,w] like Torch
--    x = input[{b,{},{},{}}]:permute(2,3,1)
--    if x:size(3)==1 then -- bug where opencv resize() squeezes singleton dimensions
--      y0 = cv.cuda.resize{x,{self.rows,self.cols}}
--      -- copy lets you add singleton dimension but doubles memory
--      y = torch.Tensor(self.rows,self.cols,1):copy(y0) 
--    else
--      y = cv.cuda.resize{x,{self.rows,self.cols,x:size(1)}}
--    end
--    out[{b,{},{},{}}] = y:permute(3,1,2)
--  end
--  self.output = out
--  return self.output
  
  
end

function ImageScaler:updateGradInput(input, gradOutput)
   resizer = nn.SpatialReSampling{owidth=self.cols,oheight=self.rows}
   self.gradInput = resizer:updateGradInput(input:float(), gradOutput:float())
   return self.gradInput:cuda()
   
   --self.gradInput = torch.Tensor(input:size()):zeros()
   --return self.gradInput
end