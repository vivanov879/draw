require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'
nngraph.setDebug(true)


N = 12
A = 2 
B = 2

function duplicate(x)
  local y = nn.Reshape(1)(x)
  local l = {}
  for i = 1, A do 
    l[#l + 1] = nn.Copy()(y)  
  end
  local z = nn.JoinTable(2)(l)
  return z
end

x = nn.Identity()()
z = duplicate(x)
m = nn.gModule({x}, {z})

x = torch.rand(3,1)
z = m:forward(x)
print(x)
print(z)

z = 2
u = duplicate(nn.Identity()())

a = 1
