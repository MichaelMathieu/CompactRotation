--torch.setdefaulttensortype("torch.FloatTensor")

require 'dataset-mnist'
require 'nn'
require 'optim'
require 'gnuplot'
--require 'cuda-hessian-optim'
require 'hessian-optim'
require 'xlua'
require 'timer'
require 'image'
require 'sys'

local function isnan(a)
   return a ~= a
end

torch.manualSeed(42)

local nEpochs = 1000
local nSamples = 60000
local nTestSamples = 10000
local nMinibatch = 50

local geo = {28, 28}
local nhidden = 500

local train_set = mnist.loadTrainSet(nSamples, geo)
local test_set  = mnist.loadTestSet (nTestSamples, geo)

local net = nn.Sequential()
net:add(nn.SpatialContrastiveNormalization())
net:add(nn.Reshape(geo[1]*geo[2]))
net:add(nn.Linear(geo[1]*geo[2], nhidden))
net:add(nn.Tanh())
net:add(nn.Linear(nhidden, nhidden))
net:add(nn.Tanh())
net:add(nn.Linear(nhidden, nhidden))
net:add(nn.Tanh())
net:add(nn.Linear(nhidden, 10))
net:add(nn.LogSoftMax())
local parameters, gradParameters = net:getParameters()
print(parameters:size(1) .. " parameters")

local config = {learningRate = 1, learningRateDecay = 1e-4}
local crit = nn.ClassNLLCriterion()
confusion = optim.ConfusionMatrix(10)
confusionControl = optim.ConfusionMatrix(10)

local ho = HessianOptimizer(parameters:size(1), 0.01, 1e5)

local controlParameters = parameters:clone()
local backupParameters = parameters:clone()

local trainErrs = {}
local testErrs  = {}
local confusions = {}

local gx = torch.Tensor(parameters:size(1))
local dw = torch.Tensor(parameters:size(1))

print("starting")
for iEpoch = 1,nEpochs do
   print(iEpoch)
   local perm = torch.randperm(nSamples)
   local meanErr = 0
   for iSample = 1,nSamples,nMinibatch do
      print("--")
      sys:tic()
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


      local lr = 0.5
      local _, gp = feval(parameters)
      sys:toc()
      gx:copy(gp)
      dw:copy(gx)
      --dw:normal() --why does that slow the NEXT feval ????
      parameters:add(lr, dw)
      local _, gxp = feval(parameters)
      parameters:add(-lr, dw)
      sys:toc()
      --sys:tic()
      --gxp:add(-1, gx) --TODO: why doesn't this work? likely a bug
      gxp = gxp - gx
      ho:newPoint(gxp, dw)
      sys:toc()
      local p = ho:invHessian(dw)
      if isnan(p:mean()) then
	 print "NaN..."
	 exit(0)
      end
      parameters:add(-lr, p)
      
      sys:toc()

      backupParameters:copy(parameters)
      optim.sgd(feval, controlParameters, config)
      parameters:copy(backupParameters)
   end
   meanErr = meanErr / nSamples
   trainErrs[iEpoch] = meanErr
   print("Training error="..meanErr)
   
   confusion:zero()
   confusionControl:zero()
   meanErr = 0
   for iTestSample = 1,nTestSamples do
      local input  = test_set[iTestSample][1]
      local target = test_set[iTestSample][2]
      local _, itarget = torch.max(target, 1)
      local output = net:forward(input)
      meanErr = meanErr + crit:forward(output, itarget)
      confusion:add(output, target)
   end

   --print(controlParameters:mean())
   backupParameters:copy(parameters)
   parameters:copy(controlParameters)
   for iTestSample = 1,nTestSamples do
      local input  = test_set[iTestSample][1]
      local target = test_set[iTestSample][2]
      local _, itarget = torch.max(target, 1)
      local output = net:forward(input)
      confusionControl:add(output, target)
   end   
   parameters:copy(backupParameters)

   meanErr = meanErr / nTestSamples
   testErrs[iEpoch] = meanErr
   print("Testing error="..meanErr)
   confusion:updateValids()
   confusionControl:updateValids()
   print(confusion.totalValid)
   print(confusionControl.totalValid)
   print("")
   table.insert(confusions, confusion)
   torch.save("mnist-hessian.th", {["test"]=testErrs,["train"]=trainErrs,["confusion"]=confusions})
end

--gnuplot.plot(torch.Tensor(testErrs), "~")