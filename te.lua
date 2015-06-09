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

x = nn.Identity()()
y = nn.Reshape(1,1)(x)
l = {}
for i = 1, A do 
  l[#l + 1] = nn.Copy()(y)  
end
z = nn.JoinTable(2)(l)
l = {}
for i = 1, B do 
  l[#l + 1] = nn.Copy()(z)  
end
z = nn.JoinTable(3)(l) 
duplicate = nn.gModule({x}, {z})

x = torch.rand(3,1)
z = duplicate:forward(x)
print(x)
print(z)

x = nn.Identity()()
z = duplicate(x)
m = nn.gModule({x}, {z})


x = torch.rand(3,1)
z = m:forward(x)
print(x)
print(z)

