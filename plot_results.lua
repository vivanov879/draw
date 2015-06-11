require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'
nngraph.setDebug(true)

x_prediction = torch.load('x_prediction')
x = torch.zeros(#x_prediction, x_prediction[1]:size(2), x_prediction[1]:size(3)) 
for i = 1, x_prediction[1]:size(1) do
  for t = 1, #x_prediction do 
    x[{{t}, {}, {}}] = x_prediction[t][i]:gt(0.5)
    
  end
  image.display(x)
end

