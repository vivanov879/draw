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
z = nn.Reshape(1,1)(x)
m = nn.gModule({x}, {z})

x = torch.rand(3,1)
z = m:forward(x)

print(x)
print(z)

