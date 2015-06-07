require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'

n = 12
a = 28 
b = 28 
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
