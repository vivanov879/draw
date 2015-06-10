require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'

nngraph.setDebug(true)


N = 12
A = 28
h_dec_n = 100
n_data = 2



x = nn.Identity()()

gx_raw = nn.Identity()()
gy_raw = nn.Identity()()
sigma_raw = nn.Identity()()
delta_raw = nn.Identity()()

delta = nn.Exp()(delta_raw)
sigma = nn.Exp()(sigma_raw)
sigma = nn.Power(-2)(sigma)
sigma = nn.MulConstant(-1/2)(sigma)
gx = nn.AddConstant(1)(gx_raw)
gy = nn.AddConstant(1)(gy_raw)
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
      filters[#filters + 1] = exp_i
  end
  filterbank = nn.JoinTable(2)(filters)
  return filterbank
end

filterbank_x = genr_filters(gx)
filterbank_y = genr_filters(gy)
patch = nn.MM(true, false)({filterbank_y, x})
patch = nn.MM()({patch, filterbank_x})

m = nn.gModule({x, gx_raw, gy_raw, delta_raw, sigma_raw, ascending}, {patch})


trainset = mnist.traindataset()
testset = mnist.testdataset()

x = torch.load('read_patches')

ascending = torch.zeros(n_data, A)
for k = 1, n_data do
  for i = 1, A do 
      ascending[k][i] = i
  end
end


gx = torch.zeros(n_data, A)
gy = torch.zeros(n_data, A)
sigma = torch.zeros(n_data, A)
delta = torch.zeros(n_data, A)

z = m:forward({x, gx, gy, delta, sigma, ascending})

print(x:gt(0.5))
print(z:gt(0.5))



