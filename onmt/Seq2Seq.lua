--[[ Sequence to sequence model with attention. ]]
local Seq2Seq, parent = torch.class('Seq2Seq', 'Model')

local options = {
  -- main network
  {'-layers', 2,           [[Number of layers in the RNN encoder/decoder]],
                     {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-rnn_size', 500, [[Size of RNN hidden states]],
                     {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-rnn_type', 'LSTM', [[Type of RNN cell]],
                     {enum={'LSTM','GRU'}}},
  {'-word_vec_size', 0, [[Common word embedding size. If set, this overrides -src_word_vec_size and -tgt_word_vec_size.]],
                     {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-src_word_vec_size', '500', [[Comma-separated list of source embedding sizes: word[,feat1,feat2,...].]]},
  {'-tgt_word_vec_size', '500', [[Comma-separated list of target embedding sizes: word[,feat1,feat2,...].]]},
  {'-feat_merge', 'concat', [[Merge action for the features embeddings]],
                     {enum={'concat','sum'}}},
  {'-feat_vec_exponent', 0.7, [[When features embedding sizes are not set and using -feat_merge concat, their dimension
                                will be set to N^exponent where N is the number of values the feature takes.]]},
  {'-feat_vec_size', 20, [[When features embedding sizes are not set and using -feat_merge sum, this is the common embedding size of the features]],
                     {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-input_feed', 1, [[Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.]],
                     {enum={0,1}}},
  {'-residual', false, [[Add residual connections between RNN layers.]]},
  {'-brnn', false, [[Use a bidirectional encoder]]},
  {'-brnn_merge', 'sum', [[Merge action for the bidirectional hidden states]],
                     {enum={'concat','sum'}}},
  {'-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings on the encoder side.
                                     See README for specific formatting instructions.]],
                         {valid=onmt.utils.ExtendedCmdLine.fileNullOrExists}},
  {'-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load
                                       pretrained word embeddings on the decoder side.
                                       See README for specific formatting instructions.]],
                         {valid=onmt.utils.ExtendedCmdLine.fileNullOrExists}},
  {'-fix_word_vecs_enc', false, [[Fix word embeddings on the encoder side]]},
  {'-fix_word_vecs_dec', false, [[Fix word embeddings on the decoder side]]},
  {'-dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]]},
  {'-gate', false, [[Use the gating mechanism]]},

  -- gating network
  {'-gating_layers', 1,           [[Number of layers in the RNN encoder/decoder]],
                     {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-gating_rnn_size', 500, [[Size of RNN hidden states]],
                     {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-gating_rnn_type', 'LSTM', [[Type of RNN cell]],
                     {enum={'LSTM','GRU'}}},
  {'-gating_word_vec_size', 0, [[Common word embedding size. If set, this overrides -src_word_vec_size and -tgt_word_vec_size.]],
                     {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-gating_src_word_vec_size', '500', [[Comma-separated list of source embedding sizes: word[,feat1,feat2,...].]]},
  {'-gating_tgt_word_vec_size', '500', [[Comma-separated list of target embedding sizes: word[,feat1,feat2,...].]]},
  {'-gating_feat_merge', 'concat', [[Merge action for the features embeddings]],
                     {enum={'concat','sum'}}},
  {'-gating_feat_vec_exponent', 0.7, [[When features embedding sizes are not set and using -feat_merge concat, their dimension
                                will be set to N^exponent where N is the number of values the feature takes.]]},
  {'-gating_feat_vec_size', 20, [[When features embedding sizes are not set and using -feat_merge sum, this is the common embedding size of the features]],
                     {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-gating_input_feed', 1, [[Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.]],
                     {enum={0,1}}},
  {'-gating_residual', false, [[Add residual connections between RNN layers.]]},
  {'-gating_brnn', true, [[Use a bidirectional encoder]]},
  {'-gating_brnn_merge', 'sum', [[Merge action for the bidirectional hidden states]],
                     {enum={'concat','sum'}}},
  {'-gating_pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings on the encoder side.
                                     See README for specific formatting instructions.]],
                         {valid=onmt.utils.ExtendedCmdLine.fileNullOrExists}},
  {'-gating_pre_word_vecs_dec', '', [[If a valid path is specified, then this will load
                                       pretrained word embeddings on the decoder side.
                                       See README for specific formatting instructions.]],
                         {valid=onmt.utils.ExtendedCmdLine.fileNullOrExists}},
  {'-gating_fix_word_vecs_enc', false, [[Fix word embeddings on the encoder side]]},
  {'-gating_fix_word_vecs_dec', false, [[Fix word embeddings on the decoder side]]},
  {'-gating_dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]]},
  {'-gating_type', 'contextBiEncoder', [[Gating Network]],
                    {enum={'contextBiEncoder', 'leave_one_out', 'conv'}}}

}

function Seq2Seq.declareOpts(cmd)
  cmd:setCmdLineOptions(options, Seq2Seq.modelName())
end

function Seq2Seq:__init(args, dicts, verbose)
  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, options))

  self.gate = args.gate
  if self.gate then
    self.models.gatingNetwork = onmt.Factory.buildGatingNetwork(args, dicts.src, verbose)
    self.gatingType = args.gating_type
  end

  self.models.encoder = onmt.Factory.buildWordEncoder(args, dicts.src, verbose)
  self.models.decoder = onmt.Factory.buildWordDecoder(args, dicts.tgt, verbose)
  self.criterion = onmt.ParallelClassNLLCriterion(onmt.Factory.getOutputSizes(dicts.tgt))
end

function Seq2Seq.load(args, models, dicts, isReplica)
  local self = torch.factory('Seq2Seq')()

  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, options))
  self.gate = args.gate
  if args.gate then
    self.models.gatingNetwork = onmt.Factory.loadGatingNetwork(models.gatingNetwork, isReplica)
    self.gatingType = args.gating_type
  end
  self.models.encoder = onmt.Factory.loadEncoder(models.encoder, isReplica)
  self.models.decoder = onmt.Factory.loadDecoder(models.decoder, isReplica)
  self.criterion = onmt.ParallelClassNLLCriterion(onmt.Factory.getOutputSizes(dicts.tgt))

  return self
end

-- Returns model name.
function Seq2Seq.modelName()
  return 'Sequence to Sequence with Attention'
end

-- Returns expected dataMode.
function Seq2Seq.dataType()
  return 'bitext'
end

function Seq2Seq:enableProfiling()
  if self.gate then
    _G.profiler.addHook(self.models.gatingNetwork, 'gatingNetwork')
  end
  _G.profiler.addHook(self.models.encoder, 'encoder')
  _G.profiler.addHook(self.models.decoder, 'decoder')
  _G.profiler.addHook(self.models.decoder.modules[2], 'generator')
  _G.profiler.addHook(self.criterion, 'criterion')
end

function Seq2Seq:getOutput(batch)
  return batch.targetOutput
end

function Seq2Seq:forwardComputeLoss(batch)
  local gatingEncStates = nil
  local gatingContext = nil
  if self.gate then
    -- gatingContext: batch x rho x dim tensor
    if self.gatingType == 'contextBiEncoder' then
      gatingEncStates, gatingContext = self.models.gatingNetwork:forward(batch)
    elseif self.gatingType == 'leave_one_out' then
      gatingContext = {}
      for t = 1, batch.sourceLength do
        local gateInputBatch = onmt.utils.Tensor.deepClone(batch)
        gateInputBatch.sourceInput[t]:fill(onmt.Constants.DOL)
        local finalStates, context = self.models.gatingNetwork:forward(gateInputBatch)
        table.insert(gatingContext, finalStates[#finalStates]) -- gatingContext then becomes rho x batch x dim -> need to transpose later
      end
      gatingContext = torch.cat(gatingContext, 1):resize(batch.sourceLength, batch.size, self.models.gatingNetwork.args.rnnSize)
      gatingContext = gatingContext:transpose(1,2) -- swapping dim1 with dim2 -> batch x rho x dim
    elseif self.gatingType == 'conv' then
      gatingContext = self.models.gatingNetwork:forward(batch)
    end
    batch:setGateTensor(gatingContext)
  end

  local encoderStates, context = self.models.encoder:forward(batch)
  return self.models.decoder:computeLoss(batch, encoderStates, context, self.criterion)
end

function Seq2Seq:trainNetwork(batch, dryRun)
  local rnnSize = nil
  local gatingEncStates = nil
  local gatingContext = nil
  if self.gate then
<<<<<<< HEAD
    --print (torch.sum(self.models.gatingNetwork.bwd.inputNet.net.weight))
=======
    --[[
    print ('----------------------')
    print (torch.sum(self.models.encoder.inputNet.modules[1].modules[1].net.weight))
    if self.models.gatingNetwork.fwd then
      print (torch.sum(self.models.gatingNetwork.fwd.inputNet.net.weight))
      -- print (torch.sum(self.models.gatingNetwork.bwd.inputNet.net.weight))
      local tmpF, tmpGPF = self.models.gatingNetwork.fwd:parameters()
      local tmpB, tmpGPB = self.models.gatingNetwork.bwd:parameters()
      print (torch.sum(tmpF[1]))
      print (torch.sum(tmpB[1]))
    else
      print (torch.sum(self.models.gatingNetwork.inputNet.net.weight))
      local tmpP, tmpGP = self.models.gatingNetwork:parameters()
      print (torch.sum(tmpP[1]))
    end
    --]]

    rnnSize = self.models.gatingNetwork.args.rnnSize
>>>>>>> 8c89337957f53b21d25654d0b530552faf79051d
    -- gatingContext: batch x rho x dim tensor
    if self.gatingType == 'contextBiEncoder' then
      rnnSize = self.models.gatingNetwork.args.rnnSize
      gatingEncStates, gatingContext = self.models.gatingNetwork:forward(batch)
    elseif self.gatingType == 'leave_one_out' then
      rnnSize = self.models.gatingNetwork.args.rnnSize
      gatingContext = {}
      for t = 1, batch.sourceLength do
        local gateInputBatch = onmt.utils.Tensor.deepClone(batch)
        gateInputBatch.sourceInput[t]:fill(onmt.Constants.DOL)
        local finalStates, context = self.models.gatingNetwork:forward(gateInputBatch)
        table.insert(gatingContext, finalStates[#finalStates]) -- gatingContext then becomes rho x batch x dim -> need to transpose later
      end
      gatingContext = torch.cat(gatingContext, 1):resize(batch.sourceLength, batch.size, self.models.gatingNetwork.args.rnnSize)
      gatingContext = gatingContext:transpose(1,2) -- swapping dim1 with dim2 -> batch x rho x dim
    elseif self.gatingType == 'conv' then
        gatingContext = self.models.gatingNetwork:forward(batch)
    end
    batch:setGateTensor(gatingContext)
  end

  -- setSourceInput

  local encStates, context = self.models.encoder:forward(batch)
  local decOutputs = self.models.decoder:forward(batch, encStates, context)
  --print(context:size())
  if dryRun then
    decOutputs = onmt.utils.Tensor.recursiveClone(decOutputs)
  end

  local encGradStatesOut, gradContext, loss = self.models.decoder:backward(batch, decOutputs, self.criterion)
  --print(gradContext:size())
  local gradInputs = self.models.encoder:backward(batch, encGradStatesOut, gradContext)
  if self.gate then
    if self.gatingType == 'contextBiEncoder' then
      --print(gradInputs[1])
      local gradGatingContext = torch.Tensor(batch.size, batch.sourceLength, rnnSize):zero()
      if #onmt.utils.Cuda.gpuIds > 0 then
        gradGatingContext = gradGatingContext:cuda()
      end
      for t = 1, batch.sourceLength do
        gradGatingContext[{{}, t, {}}] = gradInputs[t][2]
      end
      self.models.gatingNetwork:backward(batch, nil, gradGatingContext)
    elseif self.gatingType == 'conv' then
      local gradGatingContext = torch.Tensor(batch.size, batch.sourceLength, 600):zero()
      if #onmt.utils.Cuda.gpuIds > 0 then
        gradGatingContext = gradGatingContext:cuda()
      end
      for t = 1, batch.sourceLength do
        gradGatingContext[{{}, t, {}}] = gradInputs[t][2]
      end
      self.models.gatingNetwork:backward(batch, gradGatingContext)  
    elseif self.gatingType == 'leave_one_out' then
      local gradStates = {}
      local gradContext
      for t = batch.sourceLength, 1, -1 do
        local gateInputBatch = onmt.utils.Tensor.deepClone(batch)
        gateInputBatch.sourceInput[t]:fill(onmt.Constants.DOL)
        
        local gradStates = torch.Tensor()
        gradStates = onmt.utils.Tensor.initTensorTable(self.models.gatingNetwork.args.numEffectiveLayers,
                                                       gradStates, { batch.size, rnnSize })

        local gradGatingContext = torch.Tensor(batch.size, batch.sourceLength, rnnSize):zero()
        gradStates[#gradStates] = gradInputs[t][2]

        if #onmt.utils.Cuda.gpuIds > 0 then
          for s = 1, #gradStates do
            gradStates[s] = gradStates[s]:cuda()
            gradGatingContext = gradGatingContext:cuda()
          end
        end

        self.models.gatingNetwork:backward(gateInputBatch, gradStates, gradGatingContext)
      end
    end
  end
  return loss
end

return Seq2Seq
