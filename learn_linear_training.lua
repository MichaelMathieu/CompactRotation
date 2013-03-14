require 'butterfly'
require 'nn'
require 'optim'
require 'hessian_finite_difference'
require 'randomRotation'

local n = 3
local N = 2^n
local nEpochs = 100
local nSamples = 500
local nMinibatch = 50

torch.manualSeed(1)

local net = nn.Sequential()
net:add(nn.Linear(N-1,1))
local parameters, gradParameters = net:getParameters()

local criterion = nn.MSECriterion()
local config = {learningRate = 0.05, weightDecay = 0,
		momentum = 0, learningRateDecay = 1e-5}

local samples = torch.randn(nSamples, N-1)
--local Qsamples = randomRotation(N-1)
--[[
local Qsamples = torch.eye(N-1,N-1)
Qsamples[1][1] = 0.71
Qsamples[1][2] = 0.71
Qsamples[2][1] = -0.71
Qsamples[2][2] = 0.71
--]]
local Qsamples = torch.eye(N-1)
--local lambdaSamples = torch.rand(N-1):abs()
local lambdaSamples = torch.ones(N-1):mul(0.4)
lambdaSamples[2] = 1
for i = 1,nSamples do
   samples[i]:cmul(lambdaSamples)
   samples[i]:copy(torch.mm(Qsamples:t(),samples[i]:reshape(N-1,1)))
end
local targets = torch.Tensor(nSamples,1)
local M = torch.randn(N-1,1)
for i = 1,nSamples do
   targets[i][1] = M:dot(samples[i]) --+ torch.normal(0,0.01)
end

local H = torch.zeros(N,N)
for i = 1,nSamples do
   local a = torch.Tensor(N,1)
   a[{{1,N-1},{}}]:copy(samples[i])
   a[N] = 1
   H:add(torch.mm(a,a:t()))
end
H:mul(2/nSamples)

local butterflyRnd = torch.randn(n, N/2)
local Q = bf2nn(butterflyRnd)
local Hnet = nn.Sequential()
Hnet:add(Q)
local lambda = nn.CMul(N)
lambda.weight:fill(1)
Hnet:add(lambda)
--Hnet:add(lambda:clone('weight', 'gradWeight'))
Hnet:add(bfTranspose(Q, true))
local Hcrit = nn.MSECriterion()
local Hconf = {learningRate = 5, weightDecay = 0, momentum = 0,
	       learningRateDecay = 0}
local Hparameters, HgradParameters = Hnet:getParameters()
local prevx = nil
local prevg = nil

for iEpoch = 1,nEpochs do
   local perm = torch.randperm(nSamples)
   --local perm = torch.linspace(1,nSamples, nSamples)
   local meanErr = 0
   local HmeanErr = 0
   for iSample = 1,nSamples,nMinibatch do
      local updateErr = true
      local feval = function(x)
		       if x ~= parameters then
			  parameters:copy(x)
		       end
		       gradParameters:zero()
		       local err = 0
		       for iMinibatch = 0,nMinibatch-1 do
			  local input  = samples[perm[iSample+iMinibatch] ]
			  local target = targets[perm[iSample+iMinibatch] ]
			  local output = net:forward(input)
			  local lerr = criterion:forward(output, target)
			  err = err + lerr
			  local df_do = criterion:backward(output, target)
			  net:backward(input, df_do)
		       end
		       gradParameters:div(nMinibatch)
		       err = err/nMinibatch
		       if updateErr then
			  meanErr = meanErr + err
		       end
		       return err, gradParameters
		    end
      optim.sgd(feval, parameters, config)
      --[[
      local fx, dfdx = feval(parameters)
      config.evalCounter = config.evalCounter or 0
      local clr = config.learningRate/(1 + config.evalCounter*config.learningRateDecay)
      config.evalCounter = config.evalCounter + 1
      --parameters:add(-clr,dfdx)
      --local step = torch.mm(torch.inverse(H), dfdx:reshape(dfdx:size(1),1))
      local p = parameters:clone()
      for i = 1,N do
	 H
      local step = Hnet.modules[1]:forward(dfdx)
      step = Hnet.modules[2]
      parameters:add(-0.2, step)
      --]]

      if prevx ~= nil then
      
	 updateErr = false
	 --local Hinput = parameters - prevx
	 local Hinput = torch.randn(N)
	 local Hinput_norm = Hinput:norm()
	 --local Hinput_norm = 1
	 if Hinput_norm < 1e-30 then
	    print("step too small")
	 else
	    Hinput:mul(1/Hinput_norm)
	    local p = parameters:clone()
	    prevx = torch.randn(N)
	    local _, gxp = feval(prevx)
	    gxp = gxp:clone()
	    local _, gx  = feval(prevx+Hinput)
	    parameters:copy(p)
	    local Htarget = gx - gxp
	    --local Htarget = torch.mm(H, Hinput:reshape(N, 1))
	    --local Htarget = gradParameters - prevg
	    --Htarget:mul(1/Hinput_norm)
	    local Hfeval = function(x)
			      if x ~= Hparameters then
				 Hparameters:copy(x)
			      end
			      HgradParameters:zero()
			      local Houtput = Hnet:forward(Hinput)
			      local Herr = Hcrit:forward(Houtput, Htarget)
			      HmeanErr = HmeanErr + Herr
			      local Hdf_do = Hcrit:backward(Houtput, Htarget)
			      Hnet:backward(Hinput, Hdf_do)
			      --bfDistanceToNormalizeAccGrad(Hnet.modules[1], 0.1*iEpoch)
			      return Herr, HgradParameters
			   end
	    optim.sgd(Hfeval, Hparameters, Hconf)
	 end
      end
      prevx = parameters:clone()
      prevg = gradParameters:clone()
      --bfHalfNormalize(Hnet.modules[1])
      bfNormalize(Hnet.modules[1])
   end
   meanErr = meanErr/(nSamples/nMinibatch)
   HmeanErr = HmeanErr/(nSamples/nMinibatch)
   print("meanErr="..meanErr..'  HmeanErr='..HmeanErr)
end

local todisp = torch.Tensor(M:size(1),2)
todisp[{{},1}]:copy(M)
todisp[{{},2}]:copy(net.modules[1].weight)
print(net.modules[1].bias)
print(todisp)

local AH = torch.Tensor(N,N)
for i = 1,N do
   local a = torch.zeros(N)
   a[i] = 1
   AH[{{},i}]:copy(Hnet:forward(a))
end

--print(H)
--print(AH)
print(torch.eig(H))
print(torch.eig(AH))

print((H-AH):abs():max())