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
rnn_size = 100
n_canvas = 28 * 28
seq_length = 50

N = 3
A = 28
n_data = 20

function duplicate(x)
  local y = nn.Reshape(1)(x)
  local l = {}
  for i = 1, A do 
    l[#l + 1] = nn.Copy()(y)  
  end
  local z = nn.JoinTable(2)(l)
  return z
end

--encoder 
x = nn.Identity()()
x_error_prev = nn.Identity()()


--read
h_dec_prev = nn.Identity()()
gx = duplicate(nn.Linear(rnn_size, 1)(h_dec_prev))
gx = duplicate(nn.Linear(rnn_size, 1)(h_dec_prev))
gy = duplicate(nn.Linear(rnn_size, 1)(h_dec_prev))
delta = duplicate(nn.Linear(rnn_size, 1)(h_dec_prev))
gamma = duplicate(nn.Linear(rnn_size, 1)(h_dec_prev))
sigma = duplicate(nn.Linear(rnn_size, 1)(h_dec_prev))
delta = nn.Exp()(delta)
gamma = nn.Exp()(gamma)
sigma = nn.Exp()(sigma)
sigma = nn.Power(-2)(sigma)
sigma = nn.MulConstant(-1/2)(sigma)
gx = nn.AddConstant(1)(gx)
gy = nn.AddConstant(1)(gy)
gx = nn.MulConstant((A + 1) / 2)(gx)
gy = nn.MulConstant((A + 1) / 2)(gy)
delta = nn.MulConstant((math.max(A,A)-1)/(N-1))(delta)

ascending = nn.Identity()()

function genr_filters(g)
  filters = {}
  for i = 1, N do
      mu_i = nn.CAddTable()({g, nn.MulConstant(i - N/2 - 1/2)(delta)})
      mu_i = nn.MulConstant(-1)(mu_i)
      d_i = nn.CAddTable()({mu_i, ascending})
      d_i = nn.Power(2)(d_i)
      exp_i = nn.CMulTable()({d_i, sigma})
      exp_i = nn.Exp()(exp_i)
      exp_i = nn.View(n_data, 1, A)(exp_i)
      filters[#filters + 1] = nn.CMulTable()({exp_i, gamma})
  end
  filterbank = nn.JoinTable(2)(filters)
  return filterbank
end

filterbank_x = genr_filters(gx)
filterbank_y = genr_filters(gy)
patch = nn.MM()({filterbank_x, x})
patch = nn.MM(false, true)({patch, filterbank_y})
patch_error = nn.MM()({filterbank_x, x_error_prev})
patch_error = nn.MM(false, true)({patch_error, filterbank_y})
read_input = nn.JoinTable(3)({patch, patch_error})
read_input = nn.Reshape(2 * N * N)(read_input)
--read end

input = read_input
n_input = 2 * N * N

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
encoder = nn.gModule({x, x_error_prev, prev_c, prev_h, e, h_dec_prev, ascending}, {z, loss_z, next_c, next_h, patch})
encoder.name = 'encoder'


--decoder
x = nn.Identity()()
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


-- write layer
gx = duplicate(nn.Linear(rnn_size, 1)(next_h))
gx = duplicate(nn.Linear(rnn_size, 1)(next_h))
gy = duplicate(nn.Linear(rnn_size, 1)(next_h))
delta = duplicate(nn.Linear(rnn_size, 1)(next_h))
gamma = duplicate(nn.Linear(rnn_size, 1)(next_h))
sigma = duplicate(nn.Linear(rnn_size, 1)(next_h))
delta = nn.Exp()(delta)
gamma = nn.Exp()(gamma)
sigma = nn.Exp()(sigma)
sigma = nn.Power(-2)(sigma)
sigma = nn.MulConstant(-1/2)(sigma)
gx = nn.AddConstant(1)(gx)
gy = nn.AddConstant(1)(gy)
gx = nn.MulConstant((A + 1) / 2)(gx)
gy = nn.MulConstant((A + 1) / 2)(gy)
delta = nn.MulConstant((math.max(A,A)-1)/(N-1))(delta)

ascending = nn.Identity()()

function genr_filters(g)
  filters = {}
  for i = 1, N do
      mu_i = nn.CAddTable()({g, nn.MulConstant(i - N/2 - 1/2)(delta)})
      mu_i = nn.MulConstant(-1)(mu_i)
      d_i = nn.CAddTable()({mu_i, ascending})
      d_i = nn.Power(2)(d_i)
      exp_i = nn.CMulTable()({d_i, sigma})
      exp_i = nn.Exp()(exp_i)
      exp_i = nn.View(n_data, 1, A)(exp_i)
      filters[#filters + 1] = nn.CMulTable()({exp_i, gamma})
  end
  filterbank = nn.JoinTable(2)(filters)
  return filterbank
end

filterbank_x = genr_filters(gx)
filterbank_y = genr_filters(gy)
next_w = nn.Linear(rnn_size, N * N)(next_h)
next_w = nn.Reshape(N, N)(next_w)
write_layer = nn.MM(true, false)({filterbank_y, next_w})
write_layer = nn.MM()({write_layer, filterbank_x})
--write layer end

next_canvas = nn.CAddTable()({prev_canvas, write_layer})

mu = nn.Sigmoid()(next_canvas)

neg_mu = nn.MulConstant(-1)(mu)
d = nn.CAddTable()({x, neg_mu})
d2 = nn.Power(2)(d)
loss_x = nn.Sum(3)(d2)
loss_x = nn.Sum(2)(loss_x)


x_prediction = nn.Reshape(28, 28)(mu)
x_error = nn.Reshape(28, 28)(d)

decoder = nn.gModule({x, z, prev_c, prev_h, prev_canvas, ascending}, {x_prediction, x_error, next_c, next_h, next_canvas, loss_x})
decoder.name = 'decoder'


--train
trainset = mnist.traindataset()
testset = mnist.testdataset()

features_input = torch.zeros(n_data, 28, 28)

for i = 1, n_data do
    features_input[{{i}, {}, {}}] = trainset[i].x:gt(125)
end
x = features_input

params, grad_params = model_utils.combine_all_parameters(encoder, decoder)

encoder_clones = model_utils.clone_many_times(encoder, seq_length)
decoder_clones = model_utils.clone_many_times(decoder, seq_length)

ascending = torch.zeros(n_data, A)
for k = 1, n_data do
  for i = 1, A do 
      ascending[k][i] = i
  end
end


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
    x_error = {[0]=torch.rand(n_data, 28, 28)}
    x_prediction = {}
    loss_z = {}
    loss_x = {}
    canvas = {[0]=torch.rand(n_data, 28, 28)}
    x = {}
    patch = {}
    
    
    local loss = 0

    for t = 1, seq_length do
      e[t] = torch.randn(n_data, n_z)
      x[t] = features_input
      z[t], loss_z[t], lstm_c_enc[t], lstm_h_enc[t], patch[t] = unpack(encoder_clones[t]:forward({x[t], x_error[t-1], lstm_c_enc[t-1], lstm_h_enc[t-1], e[t], lstm_h_dec[t-1], ascending}))
      x_prediction[t], x_error[t], lstm_c_dec[t], lstm_h_dec[t], canvas[t], loss_x[t] = unpack(decoder_clones[t]:forward({x[t], z[t], lstm_c_dec[t-1], lstm_h_dec[t-1], canvas[t-1], ascending}))
      --print(patch[1]:gt(0.5))
      
      loss = loss + torch.mean(loss_z[t]) + torch.mean(loss_x[t])
    end
    loss = loss / seq_length

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    dlstm_c_enc = {[seq_length] = torch.zeros(n_data, rnn_size)}
    dlstm_h_enc = {[seq_length] = torch.zeros(n_data, rnn_size)}
    dlstm_c_dec = {[seq_length] = torch.zeros(n_data, rnn_size)}
    dlstm_h_dec = {[seq_length] = torch.zeros(n_data, rnn_size)}
    dlstm_h_dec1 = {[seq_length] = torch.zeros(n_data, rnn_size)}
    dlstm_h_dec2 = {[seq_length] = torch.zeros(n_data, rnn_size)}

    dx_error = {[seq_length] = torch.zeros(n_data, 28, 28)}
    dx_prediction = {}
    dloss_z = {}
    dloss_x = {}
    dcanvas = {[seq_length] = torch.zeros(n_data, 28, 28)}
    dz = {}
    dx1 = {}
    dx2 = {}
    de = {}
    dpatch = {}
    
    for t = seq_length,1,-1 do
      dloss_x[t] = torch.ones(n_data, 1)
      dloss_z[t] = torch.ones(n_data, 1)
      dx_prediction[t] = torch.zeros(n_data, 28, 28)
      dpatch[t] = torch.zeros(n_data, N, N)
      dx1[t], dz[t], dlstm_c_dec[t-1], dlstm_h_dec1[t-1], dcanvas[t-1], dascending1 = unpack(decoder_clones[t]:backward({x[t], z[t], lstm_c_dec[t-1], lstm_h_dec[t-1], canvas[t-1]}, {dx_prediction[t], dx_error[t], dlstm_c_dec[t], dlstm_h_dec[t], dcanvas[t], dloss_x[t]}))
      dx2[t], dx_error[t-1], dlstm_c_enc[t-1], dlstm_h_enc[t-1], de[t], dlstm_h_dec2[t-1], dascending2 = unpack(encoder_clones[t]:backward({x[t], x_error[t-1], lstm_c_enc[t-1], lstm_h_enc[t-1], e[t], lstm_h_dec[t-1], ascending}, {dz[t], dloss_z[t], dlstm_c_enc[t], dlstm_h_enc[t], dpatch[t]}))
      dlstm_h_dec[t-1] = dlstm_h_dec1[t-1] + dlstm_h_dec2[t-1]
    end

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


--к чему стремимся
print(x[1][1]:gt(0.5))

--что получаем со временем
for t = 1, seq_length do
  print(patch[t][1]:gt(0.5))
  print(x_prediction[t][1]:gt(0.5))
end


torch.save('x_prediction', x_prediction)

