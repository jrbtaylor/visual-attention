local debugger, parent = torch.class('nn.debugger', 'nn.Module')

function debugger:__init(name)
  parent.__init(self)
  self._name = name or ''
  self._input = torch.Tensor()
  self._gradOutput = torch.Tensor()
end

function debugger:updateOutput(input)
  print(self._name .. ':')
  if type(input)=='table' then
    print(input)
  else
    print(input:type())
    print(input:size())
  end
  self.output = input
  return self.output
end

function debugger:updateGradInput(input, gradOutput)
  --self.gradInput:viewAs(gradOutput, input)
  print(self._name .. ' backwards:')
  if type(gradOutput)=='table' then
   print(gradOutput)
  else
    print(gradOutput:type())
    print(gradOutput:size())
  end
  self.gradInput = gradOutput
  return self.gradInput
end