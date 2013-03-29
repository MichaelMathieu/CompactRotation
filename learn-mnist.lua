require 'dataset-mnist'
require 'nn'
require 'optim'
require 'gnuplot'
require 'hessian-optim'
require 'xlua'
require 'timer'

torch.manualSeed(42)

local nEpochs = 1000
local nSamples = 10000
local nTestSamples = 2000
local nMinibatch = 50

local geo = {32, 32}
local nhidden = 50

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
print(parameters:size(1) .. " parameters")

local config = {learningRate = 1e-2, learningRateDecay = 1e-7}
--local crit = nn.MSECriterion() -- TODO: bad criterion for MNIST
local crit = nn.ClassNLLCriterion()
local confusion = optim.ConfusionMatrix(10)

ho = HessianOptimizer(parameters:size(1), 0.01, 1e-5)

local trainErrs = {}
local testErrs  = {}
local confusions = {}

local tfb1 = newTimer("f/b 1")
local tclone = newTimer("clone")
local tfb1b = newTimer("f/b 1b")
local tfb2 = newTimer("f/b 2")
local tinv = newTimer("inv H")

local dw = torch.Tensor(parameters:size(1))

for iEpoch = 1,nEpochs do
   local perm = torch.randperm(nSamples)
   local meanErr = 0
   for iSample = 1,nSamples,nMinibatch do
      --xlua.progress(iSample, nSamples)
      local feval = function(x)
		       --local copyBack = nil
		       if parameters ~= x then
			  --copyBack = parameters:clone()
			  parameters:copy(x)
		       end
		       gradParameters:zero()
		       local err = 0
		       for iMinibatch = 0,nMinibatch-1 do
			  local input  = train_set[perm[iSample+iMinibatch] ][1]
			  local target = train_set[perm[iSample+iMinibatch] ][2]
			  local _, itarget = torch.max(target, 1)
			  local output = net:forward(input)
			  local lerr = crit:forward(output, itarget)
			  err = err + lerr
			  local dfdo = crit:backward(output, itarget)
			  net:backward(input, dfdo)
		       end
		       gradParameters:div(nMinibatch)
		       err = err / nMinibatch
		       meanErr = meanErr + err
		       --if copyBack ~= nil then
		       --parameters:copy(copyBack)
		       --end
		       return err, gradParameters
		    end


      local lr = 1e-5
      local _, gp = feval(parameters)
      gx = gp:clone()
      dw:normal() --why does that slow the NEXT feval ????
      dw:div(dw:norm())
      parameters:add(lr, dw)
      local _, gxp = feval(parameters)
      parameters:add(-lr, dw)
      ho:newPoint(gxp-gx, dw)
      local p = ho:invHessian(gx):clone()
      print(p:mean())
      parameters:add(p:mul(-lr))

      --optim.sgd(feval, parameters, config)
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
   table.insert(confusions, confusion)
   torch.save("mnist-hessian.th", {["test"]=testErrs,["train"]=trainErrs,["confusion"]=confusions})
end

gnuplot.plot(torch.Tensor(testErrs), "~")