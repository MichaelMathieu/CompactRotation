torch.setdefaulttensortype("torch.FloatTensor")

require 'dataset-mnist'
require 'nn'
require 'optim'
require 'gnuplot'
require 'cuda-hessian-optim'
--require 'hessian-optim'
require 'xlua'
require 'timer'
require 'image'
require 'sys'

local function isnan(a)
   return a ~= a
end

torch.manualSeed(42)

local nEpochs = 1000
local nSamples = 6000
local nTestSamples = 1000
local nMinibatch = 50

local geo = {28, 28}
local nhidden = 50

local train_set = mnist.loadTrainSet(nSamples, geo)
local test_set  = mnist.loadTestSet (nTestSamples, geo)

local train_set_d = torch.Tensor(nSamples, 1, geo[1], geo[2])
local train_set_t = torch.Tensor(nSamples, 1)
local test_set_d = torch.Tensor(nTestSamples, 1, geo[1], geo[2])
local test_set_t = torch.Tensor(nTestSamples, 1)

for i = 1,nSamples do
   train_set_d[i][1]:copy(train_set[i][1])
   local _, itarget = torch.max(train_set[i][2], 1)
   train_set_t[i][1] = itarget
end
for i = 1,nTestSamples do
   test_set_d[i][1]:copy(test_set[i][1])
   local _, itarget = torch.max(test_set[i][2], 1)
   test_set_t[i][1] = itarget
end
print("shipping data")
train_set_d = train_set_d:cuda()
train_set_t = train_set_t:cuda()
test_set_d = test_set_d:cuda()
test_set_t = test_set_t:cuda()
print("done")
--test_set = test_set:cuda()

local net = nn.Sequential()
net:add(nn.SpatialContrastiveNormalization():cuda())
net:add(nn.Reshape(geo[1]*geo[2]))
net:add(nn.Linear(geo[1]*geo[2], nhidden))
net:add(nn.Tanh())
net:add(nn.Linear(nhidden, nhidden))
net:add(nn.Tanh())
net:add(nn.Linear(nhidden, nhidden))
net:add(nn.Tanh())
net:add(nn.Linear(nhidden, 10))
net:add(nn.LogSoftMax())
net = net:cuda()
local parameters, gradParameters = net:getParameters()
print(parameters:size(1) .. " parameters")

local config = {learningRate = 1, learningRateDecay = 1e-4}
local crit = nn.ClassNLLCriterion():cuda()
confusion = optim.ConfusionMatrix(10)
confusionControl = optim.ConfusionMatrix(10)

local ho = HessianOptimizer(parameters:size(1), 0.01, 1e5)

local controlParameters = parameters:clone()
local backupParameters = parameters:clone()

local trainErrs = {}
local testErrs  = {}
local confusions = {}

local dw = torch.CudaTensor(parameters:size(1))

print("starting")
for iEpoch = 1,nEpochs do
   print(iEpoch)
   local perm = torch.randperm(nSamples)
   local meanErr = 0
   for iSample = 1,nSamples,nMinibatch do
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
			  local input  = train_set_d[perm[iSample+iMinibatch] ]
			  local itarget = train_set_t[perm[iSample+iMinibatch] ]
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
      gx = gp:clone()
      dw:copy(gx)
      --dw:normal() --why does that slow the NEXT feval ????
      parameters:add(lr, dw)
      local _, gxp = feval(parameters)
      parameters:add(-lr, dw)
      --sys:tic()
      ho:newPoint(gxp:add(-1, gx), dw) -- gxp is not to be used anymore
      --sys:toc()
      local p = ho:invHessian(gx):clone()
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
      local input  = test_set_d[iTestSample]
      local target = test_set_t[iTestSample]
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