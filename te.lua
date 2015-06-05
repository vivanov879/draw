require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'



x = nn.Identity()()
y = nn.Linear(2,3)(x)
z = nn.Linear(2,4)(x)

m = nn.gModule({x}, {y,z})

x = nn.Identity()()
u, w = m()(x)