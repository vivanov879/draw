require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'


x = nn.Identity()()

y = nn.Replicate(1, 2)(x)

m = nn.gModule({x}, {y})

x = torch.rand(4,1)
y = m:forward(x)

print(x)
print(y)



