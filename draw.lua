require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'
local LSTM = require 'LSTM'             -- LSTM timestep and utilities

nngraph.setDebug(true)


n_features = 28 * 28
n_z = 20
rnn_size = 200
n_canvas = 300
seq_length = 10


--encoder 
x_raw = nn.Identity()()
x = nn.Reshape(28 * 28)(x_raw)
x_error_prev = nn.Identity()()
input = nn.JoinTable(2)({x, x_error_prev})
n_input = 2 * n_features

prev_h = nn.Identity()()
prev_c = nn.Identity()()

function new_input_sum()
    -- transforms input
    i2h            = nn.Linear(n_input, rnn_size)(input)
    -- transforms previous timestep's output
    h2h            = nn.Linear(rnn_size, rnn_size)(prev_h)
    return nn.CAddTable()({i2h, h2h})
end

in_gate          = nn.Sigmoid()(new_input_sum())
forget_gate      = nn.Sigmoid()(new_input_sum())
out_gate         = nn.Sigmoid()(new_input_sum())
in_transform     = nn.Tanh()(new_input_sum())

next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_transform})
})
next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

mu = nn.Linear(rnn_size, n_z)(next_h)
sigma = nn.Linear(rnn_size, n_z)(next_h)
sigma = nn.Exp()(sigma)

e = nn.Identity()()
sigma_e = nn.CMulTable()({sigma, e})
z = nn.CAddTable()({mu, sigma_e})
mu_squared = nn.Square()(mu)
sigma_squared = nn.Square()(sigma)
log_sigma_sq = nn.Log()(sigma_squared)
minus_log_sigma = nn.MulConstant(-1)(log_sigma_sq)
loss_z = nn.CAddTable()({mu_squared, sigma_squared, minus_log_sigma})
loss_z = nn.AddConstant(-1)(loss_z)
loss_z = nn.MulConstant(0.5)(loss_z)
loss_z = nn.Sum(2)(loss_z)
encoder = nn.gModule({x, x_error_prev, prev_c, prev_h, e}, {z, loss_z, next_c, next_h})


--decoder
x_raw = nn.Identity()()
x = nn.Reshape(28 * 28)(x_raw)
z = nn.Identity()()
prev_h = nn.Identity()()
prev_c = nn.Identity()()
prev_canvas = nn.Identity()()
n_input = n_z
input = z

function new_input_sum()
    -- transforms input
    i2h            = nn.Linear(n_input, rnn_size)(input)
    -- transforms previous timestep's output
    h2h            = nn.Linear(rnn_size, rnn_size)(prev_h)
    return nn.CAddTable()({i2h, h2h})
end

in_gate          = nn.Sigmoid()(new_input_sum())
forget_gate      = nn.Sigmoid()(new_input_sum())
out_gate         = nn.Sigmoid()(new_input_sum())
in_transform     = nn.Tanh()(new_input_sum())

next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_transform})
})
next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})


write_layer = nn.Linear(rnn_size, n_canvas)(next_h)
next_canvas = nn.CAddTable()({prev_canvas, write_layer})

mu = nn.Linear(n_canvas, n_features)(next_canvas)
sigma = nn.Linear(n_canvas, n_features)(next_canvas)
sigma = nn.Exp()(sigma)

neg_mu = nn.MulConstant(-1)(mu)
d = nn.CAddTable()({x, neg_mu})
d2 = nn.Power(2)(d)
sigma2_inv = nn.Power(-2)(sigma)
exp_arg = nn.CMulTable()({d2, sigma2_inv})
exp_arg = nn.Sum(2)(exp_arg)

sigma_mm = nn.Log()(sigma)
sigma_mm = nn.Sum(2)(sigma)
loss_x = nn.CAddTable()({exp_arg, sigma_mm})
loss_x = nn.AddConstant(0.5 * n_features * math.log((2 * math.pi)))(loss_x)

x_prediction = nn.Reshape(28, 28)(mu)
x_error = d

decoder = nn.gModule({x, z, prev_c, prev_h, prev_canvas}, {x_prediction, x_error, next_c, next_h, next_canvas})
decoder.name = 'decoder'


--train
trainset = mnist.traindataset()
testset = mnist.testdataset()
local n_data = 100

features_input = torch.zeros(n_data, 28, 28)

for i = 1, n_data do
    features_input[{{i}, {}, {}}] = trainset[i].x:gt(125)
end

params, grads = model_utils.combine_all_parameters(encoder, decoder)

encoder_clones = model_utils.clone_many_times(encoder, seq_length)
decoder_clones = model_utils.clone_many_times(decoder, seq_length)

initstate_c_dec = torch.zeros(1, rnn_size)
initstate_h_dec = torch.zeros(1, rnn_size)
initstate_c_enc = torch.zeros(1, rnn_size)
initstate_h_enc = torch.zeros(1, rnn_size)
initstate_canvas = torch.zeros(1, n_canvas)
initstate_x_error = torch.rand(1, n_features)

dfinalstate_c_dec = torch.zeros(1, rnn_size)
dfinalstate_h_dec = torch.zeros(1, rnn_size)
dfinalstate_c_enc = torch.zeros(1, rnn_size)
dfinalstate_h_enc = torch.zeros(1, rnn_size)
dinitstate_canvas = torch.zeros(1, n_canvas)

-- do fwd/bwd and return loss, grad_params
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()
    
    ------------------ get minibatch -------------------
    local x, y = loader:next_batch()

    ------------------- forward pass -------------------
    local lstm_c_enc = {[0]=initstate_c_enc}
    local lstm_h_enc = {[0]=initstate_h_enc}
    local lstm_c_dec = {[0]=initstate_c_dec}
    local lstm_h_dec = {[0]=initstate_h_dec}
    local x_error = {[0]=initstate_x_error}
    local x_prediction = {}
    local loss_z = {}
    local canvas = {[0]=initstate_canvas}
    
    
    local loss = 0

    for t=1,opt.seq_length do
      
      e = torch.randn(n_data, n_z)
      z[t], loss_z[t], lstm_c_enc[t], lstm_h_enc[t] = unpack(encoder:forward({x[t], x_error[t-1], lstm_c_enc[t-1], lstm_h_enc[t-1], e}))
      x_prediction[t], x_error[t], lstm_c_dec[t], lstm_h_dec[t], canvas[t] = unpack(decoder:forward({x, z, lstm_c_dec[t-1], lstm_h_dec[t-1], canvas[t-1]}))
      lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:forward{embeddings[t], lstm_c[t-1], lstm_h[t-1]})

      predictions[t] = clones.softmax[t]:forward(lstm_h[t])
      loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
  end
    loss = loss / opt.seq_length

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    local dembeddings = {}                              -- d loss / d input embeddings
    local dlstm_c = {[opt.seq_length]=dfinalstate_c}    -- internal cell states of LSTM
    local dlstm_h = {[opt.seq_length]=dfinalstate_h}                                  -- output values of LSTM
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
        dlstm_h[t] = clones.softmax[t]:backward(lstm_h[t], doutput_t)

        -- backprop through LSTM timestep
        dembeddings[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(clones.lstm[t]:backward(
            {embeddings[t], lstm_c[t-1], lstm_h[t-1]},
            {dlstm_c[t], dlstm_h[t]}
        ))

        -- backprop through embeddings
        clones.embed[t]:backward(x[{{}, t}], dembeddings[t])
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    initstate_c:copy(lstm_c[#lstm_c])
    initstate_h:copy(lstm_h[#lstm_h])

    -- clip gradient element-wise
    grad_params:clamp(-5, 5)

    return loss, grad_params
end

------------------------------------------------------------------------
-- optimization loop
--
optim_state = {learningRate = 1e-2}

for i = 1, 1000 do
  local _, loss = optim.adagrad(feval, params, optim_state)

  if i % 10 == 0 then
      print(string.format("iteration %4d, loss = %6.6f", i, loss[1]))
      --print(params)
      
  end
end

e = torch.randn(n_data, n_z)
z, loss_z = unpack(encoder:forward({features_input, e}))
output, _ = unpack(decoder:forward({torch.zeros(features_input:size()), z}))
print(output[1]:gt(0.5))
print(features_input[1]:gt(0.5))
print(output[2]:gt(0.5))
print(features_input[2]:gt(0.5))
print(output[3]:gt(0.5))
print(features_input[3]:gt(0.5))
print(output[4]:gt(0.5))
print(features_input[4]:gt(0.5))


z = torch.randn(n_data, n_z)
output, _ = unpack(decoder:forward({torch.zeros(features_input:size()), z}))
  for i = 1, 3 do
  print(output[i]:gt(0.25))
end

