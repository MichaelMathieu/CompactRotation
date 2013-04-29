require 'butterfly'
require 'nn'
require 'optim'

torch.class("HessianOptimizer")

function HessianOptimizer:__init(N, lambda, learningRate, normalizationRate, maxdg)
   self.n = math.ceil(math.log(N,2))
   self.N = 2^self.n
   self.Nv = N
   self.lambda = lambda
   self.normalizationRate = normalizationRate or 1
   self.normalizationCounter = 0
   self.maxdg = maxdg or 1e-2

   self.input = torch.zeros(self.N)
   self.dfdo_pad = torch.zeros(self.N)
   self.butterfly = bf2nn(torch.randn(self.n, self.N/2))
   self.net = nn.Sequential()
   self.net:add(self.butterfly)
   --self.net:add(nn.Identity())
   self.net:add(nn.CMul(self.N))
   self.net:add(self.net.modules[2]:clone('weight','gradWeight','bias', 'gradBias'))
   self.net.modules[2].weight:fill(1)
   self.net:add(bfTranspose(self.butterfly, true))
   --self.net:add(nn.Identity())

   self.parameters, self.gradParameters = self.net:getParameters()
   self.criterion = nn.MSECriterion()
   self.criterion.sizeAverage = false
   self.criterion.gradInput = self.dfdo_pad:narrow(1, 1, self.Nv)
   self.config = {learningRate = learningRate or 0.001, learningRateDecay = 0}

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

function HessianOptimizer:fprop(input)
   self.input[{{1,self.Nv}}]:copy(input)
   local output = self.net:forward(self.input)
   local output = output[{{1,self.Nv}}]
   return output
end

function HessianOptimizer:newPoint(input, target_)
   local target = target_:clone()
   local thtarget = target:clone():abs():gt(1e-3)
   print("thtarget " .. thtarget:sum())
   target:cmul(thtarget:double())
   self.normalizationCounter = self.normalizationCounter + 1
   self.input[{{1,self.Nv}}]:copy(input)
   local saturated = self.input[{{1,self.Nv}}]:clone():abs():gt(self.maxdg):double()
   local nsaturated = self.Nv - saturated:sum()
   print(nsaturated .. " saturated (" .. nsaturated/self.Nv .. ")")
   local sgns = self.input[{{1,self.Nv}}]:clone():sign()
   self.input[{{1,self.Nv}}]:cmul(saturated)
   saturated:mul(-1):add(1):cmul(sgns)
   self.input[{{1,self.Nv}}]:add(self.maxdg, saturated)
   local function lfeval(x)
      if x ~= self.parameters then
	 self.parameters:copy(x)
      end
      self.gradParameters:zero()
      local output = self.net:forward(self.input)
      local output = output[{{1,self.Nv}}]
      local err = self.criterion:forward(output, target)
      local dfdo = self.criterion:backward(output, target)
      self.net:backward(self.input, self.dfdo_pad)
      if self.normalizationCounter >= self.normalizationRate then
	 bfDistanceToNormalizeAccGrad(self.butterfly, self.lambda)
      end
      return err, self.gradParameters
   end
   optim.sgd(lfeval, self.parameters, self.config)
   local w = self.net.modules[2].weight
   local wth = w:gt(1e-3):double()
   w:cmul(wth)
   wth:mul(-1):add(1)
   w:add(1e-3, wth)
   --print(self.net.modules[2].gradWeight:clone():abs():max())
   print(self.gradParameters:clone():abs():max())
   if self.normalizationCounter >= self.normalizationRate then
      --bfHalfNormalize(self.butterfly)
      bfNormalize(self.butterfly)
      self.normalizationCounter = 0
   end

   --print(self.parameters:mean())
   
   --self:updateInvnet() --TODO this might be lazy
end

function HessianOptimizer:invHessian(v)
   self.input[{{1,self.Nv}}]:copy(v)
   return self.net:forward(self.input)[{{1,self.Nv}}]
end