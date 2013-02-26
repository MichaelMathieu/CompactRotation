require 'butterfly'
require 'nn'
require 'optim'
require 'randomRotation'
require 'xlua'

op = xlua.OptionParser('%prog [options]')
op:option{'-n', '--n', action='store', dest='n',
	  default = 6, help='n=lg(N) where N is the size of the matrix'}
op:option{'-e', '--num-epochs', action='store', dest='nEpochs',
	  default=1000, help='Number of epochs'}
op:option{'-s', '--num-samples', action='store', dest='nSamples',
	  default=500, help='Size of the training set'}
op:option{'-norm', '--k-normalize', action='store', dest='kNormalize',
	  default=10, help='Number of SGD iterations between two normalizations'}
op:option{'-b', '--butterfly-target', action='store_true', dest='butterfly_target',
	  default=false, help='The target rotation matrix is a butterfly rotation'}
op:option{'-lr', '--learning-rate', action='store', dest='learning_rate',
	  default = 1, help='Learning rate'}
op:option{'-lrd', '--learning-rate-decay', action='store',
	  dest='learning_rate_decay', default = 1e-3, help='Learning rate decay'}
opt = op:parse()

local n = tonumber(opt.n)
local N = 2^n
local targetMat
if opt.butterfly_target then
   local butterfly = torch.randn(n,N/2)
   targetMat = bf2mat(butterfly)
else
   targetMat = randomRotation(N)
end

local butterfly2 = torch.randn(n,N/2)
local net = bf2nn(butterfly2)

local nEpochs = tonumber(opt.nEpochs)
local nSamples = tonumber(opt.nSamples)
local kNormalize = tonumber(opt.kNormalize)
local criterion = nn.MSECriterion()
local config = {learningRate = tonumber(opt.learning_rate), weightDecay = 0,
		momentum = 0, learningRageDecay = tonumber(opt.learning_rate_decay)}
print(config)
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
      if iSample % kNormalize == 0 then
	 bfNormalize(net)
      end
   end
   
   --[[
   input = samples[1]
   target = torch.mm(targetMat, input:reshape(input:size(1),1))
   local output = net:forward(input)
   local toprint = torch.Tensor(output:size(1), 2)
   toprint[{{},1}]:copy(output)
   toprint[{{},2}]:copy(target)
   print(toprint)
   --]]
   
   meanErr = meanErr/nSamples
   print("meanErr="..meanErr.." (epoch "..iEpoch..")")
end