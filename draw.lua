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

--encoder 
x = nn.Identity()()
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
decoder = nn.gModule({x, x_error_prev, prev_c, prev_h, e}, {z, loss_z, next_c, next_h})



--decoder
x = nn.Identity()()
z = nn.Identity()()
prev_h = nn.Identity()()
prev_c = nn.Identity()()
prev_canvas = nn.Identity()()
n_canvas = 300
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
canvas = nn.CAddTable()({prev_canvas, write_layer})

mu = nn.Linear(n_canvas, n_features)(canvas)
sigma = nn.Linear(n_canvas, n_features)(canvas)
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

x_prediction = mu
x_error = d

decoder = nn.gModule({x, z, prev_c, prev_h, prev_canvas}, {x_prediction, x_error, next_c, next_h, canvas})
decoder.name = 'decoder'


--train
trainset = mnist.traindataset()
testset = mnist.testdataset()
local n_data = 100

features_input = torch.zeros(n_data, 28, 28)
e = torch.randn(n_data, n_z)

for i = 1, n_data do
    features_input[{{i}, {}, {}}] = trainset[i].x:gt(125)
  
end


params, grads = model_utils.combine_all_parameters(encoder, decoder)
criterion = nn.MSECriterion()

-- return loss, grad
local feval = function(x)
  if x ~= params then
    params:copy(x)
  end
  grads:zero()

  --forward
  e = torch.randn(n_data, n_z)
  z, loss_z = unpack(encoder:forward({features_input, e}))
  local output, loss_x = unpack(decoder:forward({features_input, z}))
  local loss = torch.mean(loss_z) + torch.mean(loss_x)
  --print(output[7]:gt(0.3))
  --print(features_input[7]:gt(0.5))  
  --print('--')
  --print(torch.mean(loss_x))
  --print(torch.mean(loss_z))
  
  --backward
  
  dfeatures_input1, dz = unpack(decoder:backward({features_input, z}, {torch.zeros(output:size()), torch.ones(loss_x:size())}))
  dfeatures_input2, de = unpack(encoder:backward({features_input, e}, {dz, torch.ones(loss_z:size())}))

  return loss, grads
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

