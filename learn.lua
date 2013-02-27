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
op:option{'-r', '--renew-dataset', action='store_true', dest='renew_dataset',
	  default=false, help='Generate new training set at each epoch'}
op:option{'-norm', '--k-normalize', action='store', dest='kNormalize',
	  default=10, help='Number of SGD iterations between two normalizations'}
op:option{'-t', '--target-matrix', action='store', dest='target_matrix',
	  default="random",
	  help='The class of target matrix (random|butterfly|hessian)'}
op:option{'-lr', '--learning-rate', action='store', dest='learning_rate',
	  default = 1, help='Learning rate'}
op:option{'-lrd', '--learning-rate-decay', action='store',
	  dest='learning_rate_decay', default = 1e-3, help='Learning rate decay'}
op:option{'-sampling', '--sampling', action='store', dest='sampling',
	  default='gaussian',
	  help='Training points sampling method (gaussian|uniform)'}
op:option{'-normalize', '--normalize-input', action='store_true',
	  dest='normalize_input', default=false,
	  help='Normalize each training vector'}
op:option{'-c', '--criterion', action='store', dest='criterion',
	  default='Abs', help='Loss function (MSE|Abs)'}
opt = op:parse()

local n = tonumber(opt.n)
local N = 2^n
local targetMat
if opt.target_matrix == 'butterfly' then
   local butterfly = torch.randn(n,N/2)
   targetMat = bf2mat(butterfly)
elseif opt.target_matrix == 'random' then
   targetMat = randomRotation(N)
else
   error("target matrix class "..opt.target_matrix..' not implemented')
end
local sampler
if opt.sampling == 'gaussian' then
   sampler = function(nSamples, N)
		local a = torch.randn(nSamples, N)
		if opt.normalize_input then
		   for i = 1,nSamples do
		      a[i]:copy(a[i]/a[i]:norm())
		   end
		end
		return a
	     end
elseif opt.sampling == 'uniform' then
   sampler = function(nSamples, N)
		local a = torch.rand(nSamples,N):mul(2):add(-1)
		if opt.normalize_input then
		   for i = 1,nSamples do
		      a[i]:copy(a[i]/a[i]:norm())
		   end
		end
		return a
	     end
end

local butterfly2 = torch.randn(n,N/2)
local net = bf2nn(butterfly2)

local nEpochs = tonumber(opt.nEpochs)
local nSamples = tonumber(opt.nSamples)
local kNormalize = tonumber(opt.kNormalize)
local criterion
if opt.criterion == 'MSE' then
   criterion = nn.MSECriterion()
elseif opt.criterion == 'Abs' then
   criterion = nn.AbsCriterion()
end
local config = {learningRate = tonumber(opt.learning_rate), weightDecay = 0,
		momentum = 0, learningRageDecay = tonumber(opt.learning_rate_decay)}
print(config)
local parameters, gradParameters = net:getParameters()
local samples = sampler(nSamples, N)

for iEpoch = 1,nEpochs do
   local perm = torch.randperm(nSamples)
   local meanErr = 0
   local meanAngle = 0
   for iSample = 1,nSamples do
      if opt.renew_dataset then
	 samples = sampler(nSamples, N)
      end
      local input = samples[perm[iSample]]
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
		       --meanAngle = meanAngle + output:dot(target)/(target:norm()*output:norm())
		       meanAngle = meanAngle + output:dot(target)
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
   meanAngle = math.acos(meanAngle/nSamples)/3.14159*180
   print("meanErr="..meanErr.."  meanAngle="..meanAngle.." (epoch "..iEpoch..")")
end