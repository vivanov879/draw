require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'


x = nn.Identity()()
y = nn.Reshape(28 * 28)(x)
m = nn.gModule({x}, {y})

x = torch.rand(10,28,28)
y = m:forward(x)



