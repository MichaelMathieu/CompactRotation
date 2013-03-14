require 'dataset-mnist'
require 'nn'
require 'optim'
require 'gnuplot'
require 'hessian-optim'

torch.manualSeed(42)

local nEpochs = 1000
local nSamples = 10000
local nTestSamples = 1000

local geo = {32, 32}
local nhidden = 100

local train_set = mnist.loadTrainSet(nSamples, geo)
local test_set  = mnist.loadTestSet (nTestSamples, geo)

local net = nn.Sequential()
net:add(nn.SpatialContrastiveNormalization())
net:add(nn.Reshape(geo[1]*geo[2]))
net:add(nn.Linear(geo[1]*geo[2], nhidden))
net:add(nn.Tanh())
net:add(nn.Linear(nhidden, 10))
net:add(nn.LogSoftMax())
local parameters, gradParameters = net:getParameters()

local config = {learningRate = 1e-2, learningRateDecay = 1e-7}
--local crit = nn.MSECriterion() -- TODO: bad criterion for MNIST
local crit = nn.ClassNLLCriterion()
local confusion = optim.ConfusionMatrix(10)

ho = HessianOptimizer(parameters:size(1), 0.1)

local trainErrs = {}
local testErrs  = {}

for iEpoch = 1,nEpochs do
   local perm = torch.randperm(nSamples)
   local meanErr = 0
   for iSample = 1,nSamples do
      local input  = train_set[perm[iSample] ][1]
      local target = train_set[perm[iSample] ][2]
      local _, itarget = torch.max(target, 1)
      local feval = function(x)
		       local copyBack = nil
		       if parameters ~= x then
			  copyBack = parameters:clone()
			  parameters:copy(x)
		       end
		       gradParameters:zero()
		       local output = net:forward(input)
		       local err = crit:forward(output, itarget)
		       meanErr = meanErr + err
		       local dfdo = crit:backward(output, itarget)
		       net:backward(input, dfdo)
		       if copyBack ~= nil then
			  parameters:copy(copyBack)
		       end
		       return err, gradParameters
		    end
      --[[
      local _, gx = feval(parameters)
      gx = gx:clone()
      local dw = torch.randn(parameters:size(1))
      dw = dw / dw:norm()
      local _, gxp = feval(parameters + dw)
      ho:newPoint(dw, gxp-gx)
      local p = ho:invHessian(gx)
      parameters:add(gx:mul(-0.01))
      --]]
      optim.sgd(feval, parameters, config)
   end
   meanErr = meanErr / nSamples
   trainErrs[iEpoch] = meanErr
   print("Training error="..meanErr)
   
   confusion:zero()
   meanErr = 0
   for iTestSample = 1,nTestSamples do
      local input  = test_set[iTestSample][1]
      local target = test_set[iTestSample][2]
      local _, itarget = torch.max(target, 1)
      local output = net:forward(input)
      meanErr = meanErr + crit:forward(output, itarget)
      confusion:add(output, target)
   end
   meanErr = meanErr / nTestSamples
   testErrs[iEpoch] = meanErr
   print("Testing error="..meanErr)
   print(confusion)
   torch.save("mnist-no-hessian.th", {["test"]=testErrs,["train"]=trainErrs})
end

gnuplot.plot(torch.Tensor(testErrs), "~")