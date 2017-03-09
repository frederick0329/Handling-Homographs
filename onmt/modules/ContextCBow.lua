local ContextCBow, parent = torch.class('onmt.ContextCBow', 'onmt.Network')

--[[ Construct an encoder layer.

Parameters:

  * `inputNetwork` - input module.
  * `rnn` - recurrent module.
]]
function ContextCBow:__init(inputNetwork, cbowNetwork, inputSize)
  self.name = 'ContextCBow'
  self.inputNet = inputNetwork
  self.cbowNet = cbowNetwork

  self.args = {}
  self.args.rnnSize = inputSize 
  parent.__init(self, self:_buildModel())

end

--[[ Return a new Encoder using the serialized data `pretrained`. ]]
function ContextCBow.load(pretrained)
  local self = torch.factory('onmt.ContextCBow')()
  self.name = 'ContextCBow'
  self.args = pretrained.args
  parent.__init(self, pretrained.modules[1])


  return self
end

function ContextCBow:serialize()
  return {
    name = 'ContextCBow',
    modules = self.modules,
    args = self.args
  }
end

function ContextCBow:maskPadding()
  self.maskPad = true
end

--[[ Build one time-step of an encoder

Returns: An nn-graph mapping

  $${(c^1_{t-1}, h^1_{t-1}, .., c^L_{t-1}, h^L_{t-1}, x_t) =>
  (c^1_{t}, h^1_{t}, .., c^L_{t}, h^L_{t})}$$

  Where $$c^l$$ and $$h^l$$ are the hidden and cell states at each layer,
  $$x_t$$ is a sparse word to lookup.
--]]

function ContextCBow:_buildModel()
  local inputs = {}

  -- Input word.
  local x = nn.Identity()() -- batchSize
  table.insert(inputs, x)

  -- Compute input network.
  local wordEmb = self.inputNet(x)
  local output = self.cbowNet(wordEmb)  

  return nn.gModule(inputs, { output })
end
--]]
--[[Compute the context representation of an input.

Parameters:

  * `batch` - as defined in batch.lua.

Returns:

  1. - final hidden states
  2. - context matrix H
--]]
function ContextCBow:forward(batch)

  -- TODO: Change `batch` to `input`.

  local input = batch.sourceInput:transpose(1,2)
  self.input = input
  local cbow = self:get(1):forward(input)
  return cbow
  --]]
end

--[[ Backward pass (only called during training)

--]]
function ContextCBow:backward(batch, gradContextOutput)
  -- TODO: change this to (input, gradOutput) as in nngraph.
  --
  local gradInput = self:get(1):backward(self.input, gradContextOutput)
  return gradInput

end
