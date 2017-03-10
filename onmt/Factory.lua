local Factory = torch.class('Factory')

-- Return effective embeddings size based on user options.
local function resolveEmbSizes(opt, dicts, wordSizes)
  local wordEmbSize
  local featEmbSizes = {}

  wordSizes = onmt.utils.String.split(tostring(wordSizes), ',')

  if type(opt.word_vec_size) == 'number' and opt.word_vec_size > 0 then
    wordEmbSize = opt.word_vec_size
  else
    wordEmbSize = tonumber(wordSizes[1])
  end

  for i = 1, #dicts.features do
    local size

    if i + 1 <= #wordSizes then
      size = tonumber(wordSizes[i + 1])
    elseif opt.feat_merge == 'sum' then
      size = opt.feat_vec_size
    else
      size = math.floor(dicts.features[i]:size() ^ opt.feat_vec_exponent)
    end

    table.insert(featEmbSizes, size)
  end

  return wordEmbSize, featEmbSizes
end

local function buildGatedInputNetwork(opt, dicts, wordSizes, pretrainedWords, fixWords)
  local wordEmbSize, featEmbSizes = resolveEmbSizes(opt, dicts, wordSizes)

  local wordEmbedding = onmt.WordEmbedding.new(dicts.words:size(), -- vocab size
                                               wordEmbSize,
                                               pretrainedWords,
                                               fixWords)
  -- network for inputs
  local inputs
  local inputSize = wordEmbSize

  local multiInputs = #dicts.features > 0

  if multiInputs then
    inputs = nn.ParallelTable()
      :add(wordEmbedding)
  else
    inputs = wordEmbedding
  end

  -- Sequence with features.
  if #dicts.features > 0 then
    local vocabSizes = {}
    for i = 1, #dicts.features do
      table.insert(vocabSizes, dicts.features[i]:size())
    end

    local featEmbedding = onmt.FeaturesEmbedding.new(vocabSizes, featEmbSizes, opt.feat_merge)
    inputs:add(featEmbedding)
    inputSize = inputSize + featEmbedding.outputSize
  end

  local wordNetwork
  
  if multiInputs then
    wordNetwork = nn.Sequential()
      :add(inputs)
      :add(nn.JoinTable(2, 2))
  else
    wordNetwork = inputs
  end



  local inputNetwork
  if opt.gate == true then
    local context_size = nil
    if opt.gating_type == 'conv' then
      context_size = 600 
    else
      context_size = opt.gating_rnn_size
    end 
    local gateNetwork = nn.Sequential()
        :add(nn.Linear(context_size, inputSize))
        :add(nn.SoftMax())

    gate = nn.ParallelTable()
      :add(wordNetwork)
      :add(gateNetwork)

    inputNetwork = nn.Sequential()
        :add(gate)
        --:add(onmt.PrintIdentity())
        :add(nn.CMulTable())
	:add(nn.MulConstant(inputSize))
        --:add(onmt.PrintIdentity())
  else
    inputNetwork = wordNetwork
  end

  inputNetwork.inputSize = inputSize

  return inputNetwork
end

local function buildInputNetwork(opt, dicts, wordSizes, pretrainedWords, fixWords)
  local wordEmbSize, featEmbSizes = resolveEmbSizes(opt, dicts, wordSizes)

  local wordEmbedding = onmt.WordEmbedding.new(dicts.words:size(), -- vocab size
                                               wordEmbSize,
                                               pretrainedWords,
                                               fixWords)

  local inputs
  local inputSize = wordEmbSize

  local multiInputs = #dicts.features > 0

  if multiInputs then
    inputs = nn.ParallelTable()
      :add(wordEmbedding)
  else
    inputs = wordEmbedding
  end

  -- Sequence with features.
  if #dicts.features > 0 then
    local vocabSizes = {}
    for i = 1, #dicts.features do
      table.insert(vocabSizes, dicts.features[i]:size())
    end

    local featEmbedding = onmt.FeaturesEmbedding.new(vocabSizes, featEmbSizes, opt.feat_merge)
    inputs:add(featEmbedding)
    inputSize = inputSize + featEmbedding.outputSize
  end

  local inputNetwork

  if multiInputs then
    inputNetwork = nn.Sequential()
      :add(inputs)
      :add(nn.JoinTable(2, 2))
  else
    inputNetwork = inputs
  end

  inputNetwork.inputSize = inputSize

  return inputNetwork
end


function Factory.getOutputSizes(dicts)
  local outputSizes = { dicts.words:size() }
  for i = 1, #dicts.features do
    table.insert(outputSizes, dicts.features[i]:size())
  end
  return outputSizes
end

function Factory.buildEncoder(opt, inputNetwork)
  local encoder

  local RNN = onmt.LSTM
  if opt.rnn_type == 'GRU' then
    RNN = onmt.GRU
  end

  if opt.brnn then
    -- Compute rnn hidden size depending on hidden states merge action.
    local rnnSize = opt.rnn_size
    if opt.brnn_merge == 'concat' then
      if opt.rnn_size % 2 ~= 0 then
        error('in concat mode, rnn_size must be divisible by 2')
      end
      rnnSize = rnnSize / 2
    elseif opt.brnn_merge == 'sum' then
      rnnSize = rnnSize
    else
      error('invalid merge action ' .. opt.brnn_merge)
    end

    local rnn = RNN.new(opt.layers, inputNetwork.inputSize, rnnSize, opt.dropout, opt.residual)

    encoder = onmt.BiEncoder.new(inputNetwork, rnn, opt.brnn_merge)
  else
    local rnn = RNN.new(opt.layers, inputNetwork.inputSize, opt.rnn_size, opt.dropout, opt.residual)

    encoder = onmt.Encoder.new(inputNetwork, rnn)
  end
  return encoder
end

function Factory.buildWordEncoder(opt, dicts)
  local inputNetwork
  if opt.gate == true then
    inputNetwork = buildGatedInputNetwork(opt, dicts,
                                         opt.src_word_vec_size or opt.word_vec_size,
                                         opt.pre_word_vecs_enc, opt.fix_word_vecs_enc)
  else
    inputNetwork = buildInputNetwork(opt, dicts,
                                         opt.src_word_vec_size or opt.word_vec_size,
                                         opt.pre_word_vecs_enc, opt.fix_word_vecs_enc)
  end
  return Factory.buildEncoder(opt, inputNetwork)
end


function buildContextBiEncoder(opt, inputNetwork)
  local contextBiEncoder

  local RNN = onmt.LSTM
  if opt.gating_rnn_type == 'GRU' then
    RNN = onmt.GRU
  end

  local rnnSize = opt.gating_rnn_size
    if opt.gating_brnn_merge == 'concat' then
      if opt.gating_rnn_size % 2 ~= 0 then
        error('in concat mode, rnn_size must be divisible by 2')
      end
      rnnSize = rnnSize / 2
    elseif opt.gating_brnn_merge == 'sum' then
      rnnSize = rnnSize
    else
      error('invalid merge action ' .. opt.gating_brnn_merge)
    end

    local rnn = RNN.new(opt.gating_layers, inputNetwork.inputSize, rnnSize, opt.gating_dropout, opt.gating_residual)

    contextBiEncoder = onmt.BiEncoder.new(inputNetwork, rnn, opt.gating_brnn_merge)
    return contextBiEncoder
end

function buildConvNetwork(opt)
    local input_size = tonumber(opt.src_word_vec_size)
    local conv = nn.ConcatTable()
    local kernel_sizes = {3,5,7}
    local num_kernels = {200, 200, 200}
    for i = 1, #kernel_sizes do
        conv:add(nn.SpatialConvolution(1, num_kernels[i], input_size, kernel_sizes[i], 1, 1, 0, (kernel_sizes[i] - 1) / 2))
    end
    local convNet = nn.Sequential()
                :add(conv)
                :add(nn.JoinTable(1, 3))
                --:add(onmt.PrintIdentity())
                :add(nn.Sum(3, 3))
                :add(nn.Transpose({2, 3}))
                :add(nn.Tanh())
                --:add(onmt.PrintIdentity())
                
    return convNet
end 

function buildContextConvolution(opt, inputNetwork)
  local convNetwork = buildConvNetwork(opt)
  local contextConvolution = onmt.ContextConvolution.new(inputNetwork, convNetwork, opt.src_word_vec_size)
  return contextConvolution
end

function buildCBowNetwork(opt)
  local cbowNet = nn.Mean(1,2)
  return cbowNet
end


function buildContextCBow(opt, inputNetwork)
  local cbowNetwork = buildCBowNetwork(opt)
  local contextCBow = onmt.ContextCBow.new(inputNetwork, cbowNetwork, opt.src_word_vec_size)
  return contextCBow
end

function buildLeaveOneOut(opt, inputNetwork)
  local encoder

  local RNN = onmt.LSTM
  if opt.gating_rnn_type == 'GRU' then
    RNN = onmt.GRU
  end

  local rnn = RNN.new(opt.gating_layers, inputNetwork.inputSize, opt.gating_rnn_size, opt.gating_dropout, opt.gating_residual)
  encoder = onmt.Encoder.new(inputNetwork, rnn)
  return encoder
end


function Factory.buildGatingNetwork(opt, dicts)
  local inputNetwork = buildInputNetwork(opt, dicts,
                      opt.gating_src_word_vec_size or opt.gating_word_vec_size,
                      opt.gating_pre_word_vecs_enc, opt.gating_fix_word_vecs_enc)
  if opt.gating_type == 'leave_one_out' then
    return buildLeaveOneOut(opt, inputNetwork)
  elseif opt.gating_type == 'contextBiEncoder' then
    return buildContextBiEncoder(opt, inputNetwork)
  elseif opt.gating_type == 'conv' then
    return buildContextConvolution(opt, inputNetwork)
  elseif opt.gating_type == 'cbow' then
    return buildContextCBow(opt, inputNetwork)
  end

end

function Factory.loadGatingNetwork(pretrained, clone)
  if clone then
    pretrained = onmt.utils.Tensor.deepClone(pretrained)
  end
  if pretrained.name == 'BiEncoder' then
    return onmt.BiEncoder.load(pretrained)
  elseif pretrained.name == 'Encoder' then
    return onmt.Encoder.load(pretrained)
  elseif pretrained.name == 'ContextConvolution' then
    return onmt.ContextConvolution.load(pretrained)
  elseif pretrained.name == 'ContextCBow' then
    return onmt.ContextCBow.load(pretrained)
  end
  --[[
  if opt.gating_type == 'leave_one_out' then
    return buildLeaveOneOut(opt, inputNetwork)
  elseif opt.gating_type == 'contextBiEncoder' then
    return buildContextBiEncoder(opt, inputNetwork)
  --]]
end

function Factory.loadEncoder(pretrained, clone)
  if clone then
    pretrained = onmt.utils.Tensor.deepClone(pretrained)
  end

  if pretrained.name == 'Encoder' then
    return onmt.Encoder.load(pretrained)
  end
  if pretrained.name == 'BiEncoder' then
    return onmt.BiEncoder.load(pretrained)
  end

  -- Keep for backward compatibility.
  local brnn = #pretrained.modules == 2
  if brnn then
    return onmt.BiEncoder.load(pretrained)
  else
    return onmt.Encoder.load(pretrained)
  end
end

function Factory.buildDecoder(opt, inputNetwork, generator, verbose)
  local inputSize = inputNetwork.inputSize

  if opt.input_feed == 1 then
    if verbose then
      _G.logger:info(' * using input feeding')
    end
    inputSize = inputSize + opt.rnn_size
  end

  local RNN = onmt.LSTM
  if opt.rnn_type == 'GRU' then
    RNN = onmt.GRU
  end
  local rnn = RNN.new(opt.layers, inputSize, opt.rnn_size, opt.dropout, opt.residual)

  return onmt.Decoder.new(inputNetwork, rnn, generator, opt.input_feed == 1)
end

function Factory.buildWordDecoder(opt, dicts, verbose)
  local inputNetwork = buildInputNetwork(opt, dicts,
                                         opt.tgt_word_vec_size or opt.word_vec_size,
                                         opt.pre_word_vecs_dec, opt.fix_word_vecs_dec)

  local generator = Factory.buildGenerator(opt.rnn_size, dicts)

  return Factory.buildDecoder(opt, inputNetwork, generator, verbose)
end

function Factory.loadDecoder(pretrained, clone)
  if clone then
    pretrained = onmt.utils.Tensor.deepClone(pretrained)
  end

  return onmt.Decoder.load(pretrained)
end

function Factory.buildGenerator(rnnSize, dicts)
  if #dicts.features > 0 then
    return onmt.FeaturesGenerator(rnnSize, Factory.getOutputSizes(dicts))
  else
    return onmt.Generator(rnnSize, dicts.words:size())
  end
end

return Factory


