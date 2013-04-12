require 'cubutterfly'
require 'nn'
require 'cunn'
require 'optim'

torch.class("HessianOptimizer")

function HessianOptimizer:__init(N, lambda, learningRate)
   self.n = math.ceil(math.log(N,2))
   self.N = 2^self.n
   self.Nv = N
   self.lambda = lambda

   self.input = torch.zeros(self.N):cuda()
   self.butterfly = bf2nn(torch.randn(self.n, self.N/2))
   self.net = nn.Sequential()
   self.net:add(self.butterfly)
   self.net:add(nn.CMul(self.N):cuda())
   self.net.modules[2].weight:fill(1)
   self.net:add(bfTranspose(self.butterfly, true))

   self.parameters, self.gradParameters = self.net:getParameters()
   self.criterion = nn.MSECriterion():cuda()
   self.config = {learningRate = learningRate or 0.001}

   --self.invnet = nn.Sequential()
   --self.invnet:add(self.net.modules[1])
   --self.invnet:add(self.net.modules[2]:clone())
   --self.invnet:add(self.net.modules[3])
   --self:updateInvnet()
end

--[[
function HessianOptimizer:updateInvnet()
   self.invnet.modules[2].weight:fill(1)
   self.invnet.modules[2].weight:cdiv(self.net.modules[2].weight)
end
--]]

function HessianOptimizer:newPoint(input, target)
   target = target
   self.input[{{1,self.Nv}}]:copy(input)
   local function lfeval(x)
      if x ~= self.parameters then
	 self.parameters:copy(x)
      end
      self.gradParameters:zero()
      local output = self.net:forward(self.input)
      output = output[{{1,self.Nv}}]
      local err = self.criterion:forward(output, target)
      local dfdo = self.criterion:backward(output, target)
      self.net:backward(self.input, dfdo)
      --TODO: renormalize
      bfDistanceToNormalizeAccGrad(self.butterfly, self.lambda)
      return err, self.gradParameters
   end
   optim.sgd(lfeval, self.parameters, self.config)
   
   bfHalfNormalize(self.butterfly)
  

   --self:updateInvnet() --TODO this might be lazy
end

function HessianOptimizer:invHessian(v)
   self.input:zero()
   self.input[{{1,self.Nv}}]:copy(v)
   return self.net:forward(self.input)[{{1,self.Nv}}]
end