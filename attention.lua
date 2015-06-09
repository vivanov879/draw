require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'

N = 12
A = 28 
B = 28 
h_dec_n = 100
x = nn.Identity()()
h_dec = nn.Identity()()
gx = nn.Linear(h_dec_n, 1)(h_dec)
gx = nn.Linear(h_dec_n, 1)(h_dec)
gy = nn.Linear(h_dec_n, 1)(h_dec)
delta = nn.Linear(h_dec_n, 1)(h_dec)
gamma = nn.Linear(h_dec_n, 1)(h_dec)
delta = nn.Exp()(delta)
gamma = nn.Exp()(gamma)
gx = nn.AddConstant(1)(gx)
gy = nn.AddConstant(1)(gy)
gx = nn.MulConstant((A + 1) / 2)(gx)
gy = nn.MulConstant((B + 1) / 2)(gy)
delta = nn.MulConstant((math.max(A,B)-1)/(n-1))(delta)

vozrast_x = nn.Identity()()
vozrast_y = nn.Identity()()

filtered = {}

for i = 1, N do
  for j = 1, N do
    mu_i = nn.CAddTable()({gx, nn.MulConstant(i - N/2 - 1/2)(delta)})
    mu_j = nn.CAddTable()({gx, nn.MulConstant(j - N/2 - 1/2)(delta)})
    sigma = nn.Power(-2)(sigma)
    sigma = nn.MulConstant(-1/2)(sigma)
    mu_i = nn.Linear(1, N * N)(mu_i)
    mu_i = nn.Reshape(N, N)(mu_i)
    mu_j = nn.Linear(1, N * N)(mu_j)
    mu_j = nn.Reshape(N, N)(mu_j)
    sigma = nn.Linear(1, N * N)(sigma)
    sigma = nn.Reshape(N, N)(sigma)
    vozrast_x = nn.MulConstant(-1)(vozrast_x) 
    vozrast_y = nn.MulConstant(-1)(vozrast_y) 
    d_i = nn.CAddTable()({mu_i, vozrast_x})
    d_j = nn.CAddTable()({mu_j, vozrast_y})
    d_i = nn.Power(2)(d_i)
    d_j = nn.Power(2)(d_j)
    exp_i = nn.CMulTable()({d_i, sigma})
    exp_j = nn.CMulTable()({d_j, sigma})
    exp_i = nn.Exp()(exp_i)
    exp_j = nn.Exp()(exp_j)
    filtered[#filtered + 1] = nn.CMulTable()({exp_i, exp_j, x})
    
filtered_x = nn.JoinTable()(filtered)
filtered_x = nn.Reshape(N, N)(filtered_x)

m = nn.gModule({x, vozrast_x, vozrast_y}, {filtered_x})



