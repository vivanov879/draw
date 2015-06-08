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



for a = 1, A do 
  
  
  
  
end





