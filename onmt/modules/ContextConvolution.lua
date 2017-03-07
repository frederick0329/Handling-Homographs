local ContextConvolution, parent = torch.class('onmt.ContextConvolution', 'onmt.Network')

--[[ Construct an encoder layer.

Parameters:

  * `inputNetwork` - input module.
  * `rnn` - recurrent module.
]]
function ContextConvolution:__init(inputNetwork, convNetwork, inputSize)
  self.inputNet = inputNetwork
  self.convNet = convNetwork

  self.args = {}
  self.args.inputSize = inputSize
    
  parent.__init(self, self:_buildModel())

end

--[[ Return a new Encoder using the serialized data `pretrained`. ]]
function ContextConvolution.load(pretrained)
  local self = torch.factory('onmt.ContextConvolution')()

  self.args = pretrained.args
  parent.__init(self, pretrained.modules[1])


  return self
end

function ContextConvolution:serialize()
  return {
    name = 'ContextConvolution',
    modules = self.modules,
    args = self.args
  }
end

function ContextConvolution:maskPadding()
  self.maskPad = true
end

--[[ Build one time-step of an encoder

Returns: An nn-graph mapping

  $${(c^1_{t-1}, h^1_{t-1}, .., c^L_{t-1}, h^L_{t-1}, x_t) =>
  (c^1_{t}, h^1_{t}, .., c^L_{t}, h^L_{t})}$$

  Where $$c^l$$ and $$h^l$$ are the hidden and cell states at each layer,
  $$x_t$$ is a sparse word to lookup.
--]]

function ContextConvolution:_buildModel()
  local inputs = {}

  -- Input word.
  local x = nn.Identity()() -- batchSize
  table.insert(inputs, x)

  -- Compute input network.
  local wordEmb = self.inputNet(x)
  local resizeEmb = nn.Unsqueeze(2)(wordEmb) 
  local output = self.convNet(resizeEmb)  

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
function ContextConvolution:forward(batch)

  -- TODO: Change `batch` to `input`.

  local input = batch.sourceInput:transpose(1,2)
  self.input = input
  local conv = self:get(1):forward(input)

  return conv
  --]]
end

--[[ Backward pass (only called during training)

--]]
function ContextConvolution:backward(batch, gradContextOutput)
  -- TODO: change this to (input, gradOutput) as in nngraph.
  --
  local gradInput = self:get(1):backward(self.input, gradContextOutput)
  return gradInput

end
