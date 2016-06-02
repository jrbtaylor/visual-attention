------------------------------------------------------------------------
--[[ RecurrentAttentionInit ]]-- 
-- modified to work with two RNNs like in 
-- "Multiple Object Recognition with Visual Attention"
-- RNN1 predicts the class label
-- RNN2 controls the attention mechanism
------------------------------------------------------------------------
local RecurrentAttention2RNN, parent = torch.class("nn.RecurrentAttention2RNN", "nn.AbstractSequencer")

function RecurrentAttention2RNN:__init(init, rnn_class, rnn_attn, action, nStep)
   parent.__init(self)
   assert(torch.isTypeOf(action, 'nn.Module'))
   assert(torch.type(nStep) == 'number')
   
   self.init = init
   self.rnn_class = rnn_class
   self.rnn_attn = rnn_attn
   self.action =  (not torch.isTypeOf(action, 'nn.AbstractRecurrent')) and nn.Recursor(action) or action 
   self.nStep = nStep
   
   self.output = {} -- classifier RNN outputs
   self.attn = {}
   self.actions = {} -- action output
   
   self.outputGrad = {}
   self.attnGrad = {}
   
   self.forwardActions = false
   
end

function RecurrentAttention2RNN:updateOutput(input)
   self.rnn_class:forget()
   self.rnn_attn:forget()
   self.action:forget()
   
   for step=1,self.nStep do
      if step == 1 then
         self._initInput = self.init:updateOutput(input)
         self.attn = self.rnn_attn:updateOutput(self._initInput)
         self.actions[1] = self.action:updateOutput(self.attn)
      else
         -- sample actions from previous hidden activation (rnn2 output)
         self.actions[step] = self.action:updateOutput(self.out2[step-1])
      end
      
      -- rnn handles the recurrence internally
	  self.out1[step] = self.rnn1:updateOutput{input, self.actions[step]}
      self.out2[step] = self.rnn2:updateOutput{self.out1[step]}
   end
   
   return self.out1 -- rnn1 output is used for classification
end

function RecurrentAttention2RNN:updateGradInput(input, gradOutput)
    assert(self.rnn1.step - 1 == self.nStep, "inconsistent rnn steps")
    assert(self.rnn2.step - 1 == self.nStep, "inconsistent rnn steps")
    assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
    assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    
    -- back-propagate through time (BPTT)
    for step=self.nStep,1,-1 do
        -- 1. backward through the action layer
        local gradOutput_, gradAction_ = gradOutput[step]

        -- Note : gradOutput is ignored by REINFORCE modules so we give a zero Tensor instead
        self._gradAction = self._gradAction or self.action.output.new()
        if not self._gradAction:isSameSizeAs(self.action.output) then
            self._gradAction:resizeAs(self.action.output):zero()
        end
        gradAction_ = self._gradAction

        if step == self.nStep then
            self.gradHidden2[step] = nn.rnn.recursiveCopy(self.gradHidden2[step], gradOutput_)
            self.gradHidden1[step] = nn.rnn.recursiveCopy(self.gradHidden1[step], self.gradHidden2[step])
        else
            -- gradHidden = gradOutput + gradAction
            nn.rnn.recursiveAdd(self.gradHidden2[step], gradOutput_)
            nn.rnn.recursiveAdd(self.gradHidden1[step], self.gradHidden2[step])
        end

        if step == 1 then
            -- backward through initial starting actions
            self.action:updateGradInput(self._initInput, gradAction_)
        else
            local gradAction = self.action:updateGradInput(self.output[step-1], gradAction_)
            self.gradHidden2[step-1] = nn.rnn.recursiveCopy(self.gradHidden2[step-1], gradAction)
            self.gradHidden1[step-1] = nn.rnn.recursiveCopy(self.gradHidden1[step-1], self.gradHidden2[step-1])
        end

        -- 2. backward through the rnn layer
        local gradInput = self.rnn1:updateGradInput({input, self.actions[step]}, self.gradHidden1[step])[1]
        if step == self.nStep then
            self.gradInput:resizeAs(gradInput):copy(gradInput)
        else
            self.gradInput:add(gradInput)
        end
    end

    return self.gradInput
end

function RecurrentAttention2RNN:accGradParameters(input, gradOutput, scale)
   assert(self.rnn1.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(self.rnn2.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
   
   -- back-propagate through time (BPTT)
   for step=self.nStep,1,-1 do
      -- 1. backward through the action layer
      local gradAction_ = self.forwardActions and gradOutput[step][2] or self._gradAction
            
      if step == 1 then
         -- backward through initial starting actions
         self.action:accGradParameters(self._initInput, gradAction_, scale)
      else
         self.action:accGradParameters(self.output[step-1], gradAction_, scale)
      end
      
      -- 2. backward through the rnn layer
      self.rnn2:accGradParameters({input, self.actions[step]}, self.gradHidden2[step], scale)
      self.rnn1:accGradParameters({input, self.actions[step]}, self.gradHidden1[step], scale)
   end
end

function RecurrentAttention2RNN:accUpdateGradParameters(input, gradOutput, lr)
   assert(self.rnn1.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(self.rnn2.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    
   -- backward through the action layers
   for step=self.nStep,1,-1 do
      -- 1. backward through the action layer
      local gradAction_ = self.forwardActions and gradOutput[step][2] or self._gradAction
      
      if step == 1 then
         -- backward through initial starting actions
         self.action:accUpdateGradParameters(self._initInput, gradAction_, lr)
      else
         -- Note : gradOutput is ignored by REINFORCE modules so we give action.output as a dummy variable
         self.action:accUpdateGradParameters(self.output[step-1], gradAction_, lr)
      end
      
      -- 2. backward through the rnn layer
      self.rnn2:accUpdateGradParameters({input, self.actions[step]}, self.gradHidden2[step], lr)
      self.rnn1:accUpdateGradParameters({input, self.actions[step]}, self.gradHidden1[step], lr)
   end
end

function RecurrentAttention2RNN:type(type)
   self._input = nil
   self._actions = nil
   self._crop = nil
   self._pad = nil
   self._byte = nil
   return parent.type(self, type)
end

function RecurrentAttention2RNN:__tostring__()
   local tab = '  '
   local line = '\n'
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {'
   str = str .. line .. tab .. 'action : ' .. tostring(self.action):gsub(line, line .. tab .. ext)
   str = str .. line .. tab .. 'rnn     : ' .. tostring(self.rnn):gsub(line, line .. tab .. ext)
   str = str .. line .. '}'
   return str
end
