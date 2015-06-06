require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'

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
encoder = nn.gModule({x_raw, x_error_prev, prev_c, prev_h, e}, {z, loss_z, next_c, next_h})
encoder.name = 'encoder'


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

decoder = nn.gModule({x_raw, z, prev_c, prev_h, prev_canvas}, {x_prediction, x_error, next_c, next_h, next_canvas, loss_x})
decoder.name = 'decoder'


--train
trainset = mnist.traindataset()
testset = mnist.testdataset()
local n_data = 100

features_input = torch.zeros(n_data, 28, 28)

for i = 1, n_data do
    features_input[{{i}, {}, {}}] = trainset[i].x:gt(125)
end
x = features_input

params, grad_params = model_utils.combine_all_parameters(encoder, decoder)

encoder_clones = model_utils.clone_many_times(encoder, seq_length)
decoder_clones = model_utils.clone_many_times(decoder, seq_length)



-- do fwd/bwd and return loss, grad_params
function feval(x_arg)
    if x_arg ~= params then
        params:copy(x_arg)
    end
    grad_params:zero()
    
    ------------------- forward pass -------------------
    lstm_c_enc = {[0]=torch.zeros(n_data, rnn_size)}
    lstm_h_enc = {[0]=torch.zeros(n_data, rnn_size)}
    lstm_c_dec = {[0]=torch.zeros(n_data, rnn_size)}
    lstm_h_dec = {[0]=torch.zeros(n_data, rnn_size)}
    x_error = {[0]=torch.rand(n_data, n_features)}
    x_prediction = {}
    loss_z = {}
    loss_x = {}
    canvas = {[0]=torch.rand(n_data, n_canvas)}
    x = {}
    
    
    local loss = 0

    for t = 1, seq_length do
      e[t] = torch.randn(n_data, n_z)
      x[t] = features_input
      z[t], loss_z[t], lstm_c_enc[t], lstm_h_enc[t] = unpack(encoder_clones[t]:forward({x[t], x_error[t-1], lstm_c_enc[t-1], lstm_h_enc[t-1], e[t]}))
      x_prediction[t], x_error[t], lstm_c_dec[t], lstm_h_dec[t], canvas[t], loss_x[t] = unpack(decoder_clones[t]:forward({x[t], z[t], lstm_c_dec[t-1], lstm_h_dec[t-1], canvas[t-1]}))
      
      loss = loss + torch.mean(loss_z[t]) + torch.mean(loss_x[t])
    end
    loss = loss / seq_length

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    dlstm_c_enc = {[seq_length] = torch.zeros(n_data, rnn_size)}
    dlstm_h_enc = {[seq_length] = torch.zeros(n_data, rnn_size)}
    dlstm_c_dec = {[seq_length] = torch.zeros(n_data, rnn_size)}
    dlstm_h_dec = {[seq_length] = torch.zeros(n_data, rnn_size)}
    dx_error = {[seq_length] = torch.zeros(n_data, n_features)}
    dx_prediction = {}
    dloss_z = {}
    dloss_x = {}
    dcanvas = {[seq_length] = torch.zeros(n_data, n_canvas)}
    dz = {}
    dx1 = {}
    dx2 = {}
    de = {}
    
    for t = seq_length,1,-1 do
      dloss_x[t] = torch.ones(n_data, 1)
      dloss_z[t] = torch.ones(n_data, 1)
      dx_prediction[t] = torch.zeros(n_data, 28, 28)
      dx1[t], dz[t], dlstm_c_dec[t-1], dlstm_h_dec[t-1], dcanvas[t-1] = unpack(decoder_clones[t]:backward({x[t], z[t], lstm_c_dec[t-1], lstm_h_dec[t-1], canvas[t-1]}, {dx_prediction[t], dx_error[t], dlstm_c_dec[t], dlstm_h_dec[t], dcanvas[t], dloss_x[t]}))
      dx2[t], dx_error[t-1], dlstm_c_enc[t-1], dlstm_h_enc[t-1], de[t] = unpack(encoder_clones[t]:backward({x[t], x_error[t-1], lstm_c_enc[t-1], lstm_h_enc[t-1], e[t]}, {dz[t], dloss_z[t], dlstm_c_enc[t], dlstm_h_enc[t]}))
    end

    -- clip gradient element-wise
    grad_params:clamp(-5, 5)

    return loss, grad_params
end

------------------------------------------------------------------------
-- optimization loop
--
optim_state = {learningRate = 1e-2}

for i = 1, 200 do
  local _, loss = optim.adagrad(feval, params, optim_state)

  if i % 10 == 0 then
      print(string.format("iteration %4d, loss = %6.6f", i, loss[1]))
      --print(params)
      
  end
end


--к чему стремимся
print(x[1][1]:gt(0.5))

--что получаем со временем
for t = seq_length,1,-1 do
  print(x_prediction[t][1]:gt(0.5))
end
--print(x_prediction[2]:gt(0.5))
--print(x[2]:gt(0.5))
--print(x_prediction[3]:gt(0.5))
--print(x[3]:gt(0.5))
--print(x_prediction[4]:gt(0.5))
--print(x[4]:gt(0.5))

