require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'

local Selective, parent = torch.class('Selective', 'nn.Module')

function Selective:__init(gx, gy, delta, sigma, grid_size, image_size)
  parent.__init(self)
  self.gx = gx
  self.gy = gy
  self.delta = delta
  self.sigma = sigma
  self.gamma = gamma
  self.N = grid_size
  self.A = image_size
  self.B = image_size
  
end


function Selective:updateOutput(input)
  self.output:resize(input:size(1), self.N, self.N)
  for k = 1, input.size(1):
    mu_x = torch.zeros(self.N)
    mu_y = torch.zeros(self.N)
    for i = 1, self.N do
        mu_x[i] = self.gx + self.delta * (i - self.N/2 - 1/2) 
        mu_y[i] = self.gy + self.delta * (i - self.N/2 - 1/2)
    end
    f_x = torch.zeros(self.N, self.A)
    f_y = torch.zeros(self.N, self.A)
    for i = 1, self.N do
      for a = 1, self.A do
        f_x[i][a] = math.exp(- (a - mu_x[i])^2 / (2 * self.sigma ^ 2))
        f_y[i][a] = math.exp(- (a - mu_y[i])^2 / (2 * self.sigma ^ 2))
      end
    end
    for i = 1, self.N do 
      for j = 1, self.N do
        for a = 1, self.A do
          for b = 1, self.A do
            self.output[k][i][j] = self.output[k][i][j] + f_x[i][a] * f_y[j][b] * input[k][a][b]
          end
        end
      end
    end
  end
  
  
  return self.output
end

function Embedding:updateGradInput(input, gradOutput)
  if self.gradInput then
    self.gradInput:resize(input:size())
    return self.gradInput
  end
end

function Embedding:accGradParameters(input, gradOutput, scale)
  scale = scale or 1
  if scale == 0 then
    self.gradWeight:zero()
  end
  for i = 1, input:size(1) do
    local word = input[i]
    self.gradWeight[word]:add(gradOutput[i])
  end
end

-- we do not need to accumulate parameters when sharing
Embedding.sharedAccUpdateGradParameters = Embedding.accUpdateGradParameters
