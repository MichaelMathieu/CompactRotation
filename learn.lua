require 'butterfly'
require 'nn'
require 'optim'

local n = 6
local N = 2^n
local butterfly = torch.randn(n,N/2)
local targetMat = bf2mat(butterfly)

local butterfly2 = torch.randn(n,N/2)
local net = bf2nn(butterfly2)

local nEpochs = 100
local nSamples = 100
local criterion = nn.MSECriterion()
local config = {learningRage = 1, weightDecay = 0,
		momentum = 0, learningRageDecay = 1e-5}
local parameters, gradParameters = net:getParameters()
local samples = torch.randn(nSamples, N)

for iEpoch = 1,nEpochs do
   local meanErr = 0
   for iSample = 1,nSamples do
      local input = samples[iSample]
      local target = torch.mm(targetMat, input:reshape(input:size(1),1))
      local feval = function(x)
		       if x ~= parameters then
			  parameters:copy(x)
		       end
		       gradParameters:zero()
		       local output = net:forward(input)
		       local err = criterion:forward(output, target)
		       local df_do = criterion:backward(output, target)
		       net:backward(input, df_do)
		       meanErr = meanErr + err
		       return err, gradParameters
		    end
      optim.sgd(feval, parameters, config)
      bfNormalize(net)
   end
   meanErr = meanErr/nSamples
   print("meanErr=",meanErr)
end