require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
nngraph.setDebug(true)


N = 2
A = 2 
h_dec_n = 100

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
h_dec = nn.Identity()()
gx = duplicate(nn.Linear(h_dec_n, 1)(h_dec))
gx = duplicate(nn.Linear(h_dec_n, 1)(h_dec))
gy = duplicate(nn.Linear(h_dec_n, 1)(h_dec))
delta = duplicate(nn.Linear(h_dec_n, 1)(h_dec))
gamma = duplicate(nn.Linear(h_dec_n, 1)(h_dec))
sigma = duplicate(nn.Linear(h_dec_n, 1)(h_dec))
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

vozrast_x = nn.Identity()()
vozrast_y = nn.Identity()()

filtered = {}

for i = 1, N do
  for j = 1, N do
    mu_i = nn.CAddTable()({gx, nn.MulConstant(i - N/2 - 1/2)(delta)})
    mu_j = nn.CAddTable()({gy, nn.MulConstant(j - N/2 - 1/2)(delta)})
    mu_i = nn.MulConstant(-1)(mu_i)
    mu_j = nn.MulConstant(-1)(mu_j)
    
    d_i = nn.CAddTable()({mu_i, vozrast_x})
    d_j = nn.CAddTable()({mu_j, vozrast_y})
    d_i = nn.Power(2)(d_i)
    d_j = nn.Power(2)(d_j)
    exp_i = nn.CMulTable()({d_i, sigma})
    exp_j = nn.CMulTable()({d_j, sigma})
    exp_i = nn.Exp()(exp_i)
    exp_j = nn.Exp()(exp_j)
    filtered[#filtered + 1] = nn.Sum(3)(nn.Sum(2)(nn.CMulTable()({exp_i, exp_j, x})))
  end
end
    
filtered_x = nn.JoinTable()(filtered)
filtered_x = nn.Reshape(N, N)(filtered_x)

m = nn.gModule({x, h_dec, vozrast_x, vozrast_y}, {filtered_x})
graph.dot(m.fg)

vozrast_x = torch.zeros(A, A)
vozrast_y = torch.zeros(A, A)
for i = 1, A do 
  for j = 1, A do 
    vozrast_x[i][j] = i
    vozrast_y[i][j] = j
  end
end

print(vozrast_x)
print(vozrast_y)





