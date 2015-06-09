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
y = nn.Reshape(1,1)(x)
l = {}
for i = 1, A do 
  l[#l + 1] = nn.Copy()(y)  
end
z = nn.JoinTable(2)(l)
l = {}
for i = 1, A do 
  l[#l + 1] = nn.Copy()(z)  
end
z = nn.JoinTable(3)(l) 
duplicate = nn.gModule({x}, {z})


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

ascending_x = nn.Identity()()
ascending_y = nn.Identity()()

filtered = {}

for i = 1, N do
  for j = 1, N do
    mu_i = nn.CAddTable()({gx, nn.MulConstant(i - N/2 - 1/2)(delta)})
    mu_j = nn.CAddTable()({gy, nn.MulConstant(j - N/2 - 1/2)(delta)})
    mu_i = nn.MulConstant(-1)(mu_i)
    mu_j = nn.MulConstant(-1)(mu_j)
    d_i = nn.CAddTable()({mu_i, ascending_x})
    d_j = nn.CAddTable()({mu_j, ascending_y})
    d_i = nn.Power(2)(d_i)
    d_j = nn.Power(2)(d_j)
    exp_i = nn.CMulTable()({d_i, sigma})
    exp_j = nn.CMulTable()({d_j, sigma})
    exp_i = nn.Exp()(exp_i)
    exp_j = nn.Exp()(exp_j)
    filtered[#filtered + 1] = nn.View(n_data, 1)(nn.Sum(2)(nn.Sum(3)(nn.CMulTable()({exp_i, exp_j, x}))))
  end
end
    
filtered_x = nn.JoinTable(2)(filtered)
filtered_x = nn.Reshape(N, N)(filtered_x)


m = nn.gModule({x, gx_raw, gy_raw, delta_raw, sigma_raw, ascending_x, ascending_y}, {filtered_x})


trainset = mnist.traindataset()
testset = mnist.testdataset()


x = torch.zeros(n_data, A, A)
for i = 1, n_data do
    x[{{i}, {}, {}}] = trainset[i].x:gt(125)
end


ascending_x = torch.zeros(n_data, A, A)
ascending_y = torch.zeros(n_data, A, A)
for k = 1, n_data do
  for i = 1, A do 
    for j = 1, A do 
      ascending_x[k][i][j] = i
      ascending_y[k][i][j] = j
    end
  end
end


gx = torch.zeros(n_data, A, A)
gy = torch.zeros(n_data, A, A)
sigma = torch.zeros(n_data, A, A)
delta = torch.zeros(n_data, A, A)

z = m:forward({x, gx, gy, delta, sigma, ascending_x, ascending_y})

print(x:gt(0.5))
print(z:gt(0.5))






