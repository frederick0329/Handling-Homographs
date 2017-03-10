--[[ Generic Model class. ]]
local nnq = require 'nnquery'
local Model = torch.class('Model')

local options = {
  {'-model_type', 'seq2seq',  [[Type of the model to train.
                              This option impacts all options choices]],
                     {enum={'lm','seq2seq'}}},
  {'-param_init', 0.1, [[Parameters are initialized over uniform distribution with support (-param_init, param_init)]],
                       {valid=function(v) return v >= 0 and v <= 1 end}},
  --{'-share', false, [[share contextnet lookupTable with encoder]], {enum={true, false}}}
}

function Model.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
end

function Model:__init(args)
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)
  self.args.train_from = args.train_from
  self.models = {}
end

function Model:getInputLabelsCount(batch)
  return batch.sourceInput:ne(onmt.Constants.PAD):sum()
end

function Model:getOutputLabelsCount(batch)
  return self:getOutput(batch):ne(onmt.Constants.PAD):sum()
end

function Model:evaluate()
  for _, m in pairs(self.models) do
    m:evaluate()
  end
end

function Model:training()
  for _, m in pairs(self.models) do
    m:training()
  end
end

function Model:initParams(verbose)
  local numParams = 0
  local params = {}
  local gradParams = {}

  if verbose then
    _G.logger:info('Initializing parameters...')
  end

  -- Order the model table because we need all replicas to have the same order.
  local orderedIndex = {}
  for key in pairs(self.models) do
    table.insert(orderedIndex, key)
  end
  table.sort(orderedIndex)

  for _, key in ipairs(orderedIndex) do
    local mod = self.models[key]
    local p, gp = mod:getParameters()

    if key == 'gatingNetwork' and self.args.share then
      if mod.name == 'BiEncoder' then --contextBiEncoder
        -- dirty code... 
        -- p, gp = mod.fwd.network:getParameters()
        local fwd = mod.modules[1].modules[1]
        p, gp = nnq(fwd):descendants()[3]:val().data.module:getParameters()

        if self.args.train_from:len() == 0 then
          p:uniform(-self.args.param_init, self.args.param_init)

          mod:apply(function (m)
            if m.postParametersInitialization then
              m:postParametersInitialization()
            end
          end)
        end

        numParams = numParams + p:size(1)
        table.insert(params, p)
        table.insert(gradParams, gp)
        local bwd = mod.modules[2].modules[1]
        p, gp = nnq(bwd):descendants()[3]:val().data.module:getParameters()
        -- p, gp = mod.bwd.rnn:getParameters()
      elseif mod.name == 'Encoder' then -- leave_one_out
        -- p, gp = mod.rnn:getParameters()
        p, gp = nnq(mod.modules[1]):descendants()[3]:val().data.module:getParameters() -- this line needs to check later
      elseif mod.name == 'ContextConvolution' then
        -- print (nnq(mod.modules[1]):descendants()[4]:val().data.module)
        p, gp = nnq(mod.modules[1]):descendants()[4]:val().data.module:getParameters()
      end
    end

    if self.args.train_from:len() == 0 then
      p:uniform(-self.args.param_init, self.args.param_init)

      mod:apply(function (m)
        if m.postParametersInitialization then
          m:postParametersInitialization()
        end
      end)
    end

    numParams = numParams + p:size(1)
    table.insert(params, p)
    table.insert(gradParams, gp)
  end

  if verbose then
    _G.logger:info(' * number of parameters: ' .. numParams)
  end
  -- manually share the lookupTable
  if self.models.gatingNetwork and self.args.share then
    -- local p, gp = self.models['encoder'].inputNet.modules[1].modules[1]:parameters()
    
    local p, gp
    if self.models['encoder'].name == 'Encoder' then
      p, gp = nnq(self.models['encoder'].modules[1]):descendants()[13]:val().data.module.modules[1].modules[1]:parameters()
    elseif self.models['encoder'].name == 'BiEncoder' then
      p, gp = nnq(self.models['encoder'].modules[1].modules[1]):descendants()[13]:val().data.module.modules[1].modules[1]:parameters()
    end
    local cloneP, cloneGP
    if self.models.gatingNetwork.name == 'BiEncoder' then
      -- cloneP, cloneGP = self.models['gatingNetwork'].fwd.inputNet:parameters()
      local fwd = self.models['gatingNetwork'].modules[1].modules[1]
      cloneP, cloneGP = nnq(fwd):descendants()[9]:val().data.module:parameters()
      for i = 1, #p do
        cloneP[i]:set(p[i])
        cloneGP[i]:set(gp[i])
      end
      -- cloneP, cloneGP = self.models['gatingNetwork'].bwd.inputNet:parameters()
      local bwd = self.models['gatingNetwork'].modules[2].modules[1]
      cloneP, cloneGP = nnq(bwd):descendants()[9]:val().data.module:parameters()
      for i = 1, #p do
        cloneP[i]:set(p[i])
        cloneGP[i]:set(gp[i])
      end
    elseif self.models.gatingNetwork.name == 'Encoder' then
      -- cloneP, cloneGP = self.models['gatingNetwork'].inputNet:parameters()
      cloneP, cloneGP = nnq(self.models['gatingNetwork'].modules[1]):descendants()[9]:val().data.module:parameters() -- this line needs to check later
      for i = 1, #p do
        cloneP[i]:set(p[i])
        cloneGP[i]:set(gp[i])
      end
    elseif self.models.gatingNetwork.name == 'ContextConvolution' or self.models.gatingNetwork.name == 'ContextCBow' then
        -- print (nnq(self.models['gatingNetwork'].modules[1]):descendants())
        cloneP, cloneGP = nnq(self.models['gatingNetwork'].modules[1]):descendants()[2]:val().data.module:parameters()
        for i = 1, #p do
          cloneP[i]:set(p[i])
          cloneGP[i]:set(gp[i])
        end
    end
  end
  return params, gradParams
end

return Model
