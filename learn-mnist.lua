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
require 'cut'

local function isnan(a)
   return a ~= a
end

torch.manualSeed(42)

dofile('config-conv.lua')

train_set = mnist.loadTrainSet(nSamples, geo)
test_set  = mnist.loadTestSet (nTestSamples, geo)

local controlParameters = parameters:clone()
local backupParameters = parameters:clone()

local trainErrs = {}
local testErrs  = {}
local confusions = {}

local gx = torch.Tensor(parameters:size(1))
local dw = torch.Tensor(parameters:size(1))
local nevals = 0
local lrdw = 0

print("starting")
for iEpoch = 1,nEpochs do
   print(iEpoch)
   local perm = torch.randperm(nSamples)
   local meanErr = 0
   for iSample = 1,nSamples,nMinibatch do
      print ("  " .. (iSample-1)/nMinibatch .. "/" .. nSamples/nMinibatch)
      io.flush()
      --print("--")
      --sys:tic()
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
			  itarget = itarget[1]
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


      local lrd = learningRateDecayHO
      local lr = learningRateHO
      local _, gp = feval(parameters)
      --sys:toc()
      gx:copy(gp)
      dw:copy(gx)
      --dw:normal() --why does that slow the NEXT feval ????
      parameters:add(lrdw, dw)
      local _, gxp = feval(parameters)
      parameters:add(-lrdw, dw)
      --sys:toc()
      --sys:tic()
      --gxp:add(-1, gx) --TODO: why doesn't this work? likely a bug
      gxp = gxp - gx
      ho:newPoint(gxp, dw)
      --sys:toc()
      local p = ho:invHessian(dw)
      if isnan(p:mean()) then
	 print "NaN..."
	 exit(0)
      end

      if nevals % 25 == 0 then
	 meanErrCpy = meanErr
	 local lambda = lr / (1+lrd*nevals)
	 local energy = cutEnergy(feval, parameters, p, -lambda)
	 local energyControl = cutEnergy(feval, parameters, gx, -lambda)
	 local todisp = torch.Tensor(energy:size(1), 2)
	 todisp[{{},1}]:copy(energy)
	 todisp[{{},2}]:copy(energyControl)
	 print(todisp)
	 meanErr = meanErrCpy
      end

      parameters:add(-lr/(1+lrd*nevals), p)
      lrdw = lr/(1+lrd*nevals)*p:norm()
      nevals = nevals + 1

      --sys:toc()
      
      local meanErrCpy = meanErr
      backupParameters:copy(parameters)
      optim.sgd(feval, controlParameters, config)
      parameters:copy(backupParameters)
      meanErr = meanErrCpy

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
      itarget = itarget[1]
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
      itarget = itarget[1]
      local output = net:forward(input)
      confusionControl:add(output, target)
   end   
   parameters:copy(backupParameters)

   meanErr = meanErr / nTestSamples
   testErrs[iEpoch] = meanErr
   print("Testing error="..meanErr)
   confusion:updateValids()
   confusionControl:updateValids()
   print("Accuracy= " .. confusion.totalValid)
   print("Control= " .. confusionControl.totalValid)
   print("")
   io.flush()
   table.insert(confusions, confusion)
   torch.save("mnist-hessian.th", {["test"]=testErrs,["train"]=trainErrs,["confusion"]=confusions})
end

--gnuplot.plot(torch.Tensor(testErrs), "~")