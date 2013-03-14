require 'butterfly'
require 'nn'
require 'optim'
require 'randomRotation'
require 'xlua'
require 'gnuplot'

torch.manualSeed(1)

op = xlua.OptionParser('%prog [options]')
op:option{'-n', '--n', action='store', dest='n',
	  default = 6, help='n=lg(N) where N is the size of the matrix'}
op:option{'-e', '--num-epochs', action='store', dest='nEpochs',
	  default=1000, help='Number of epochs'}
op:option{'-s', '--num-samples', action='store', dest='nSamples',
	  default=500, help='Size of the training set'}
op:option{'-r', '--renew-dataset', action='store_true', dest='renew_dataset',
	  default=false, help='Generate new training set at each epoch'}
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
op:option{'-l', '--lambda', action='store', dest='lambda',
	  default = 0.001, help='Normalizer lambda'}
opt = op:parse()

local n = tonumber(opt.n)
local N = 2^n
local lambda = tonumber(opt.lambda)

local nX = 10000
--local X = torch.randn(nX, N-1)
local X = torch.randn(nX, N)
local rndbut = torch.randn(n, N/2)
local rndrot = bf2mat(rndbut)
--local qX = rndrot[{{1,N-1},{1,N-1}}]
local qX = rndrot
--local eigX = torch.zeros(N-1)
local eigX = torch.zeros(N)
local kStepEig = 2
for i = 1,N-1,kStepEig do
   --eigX[i] = torch.normal(0,0.5)
   for j = 0,kStepEig-1 do
--      if i+j <= N-1 then
      if i+j <= N then
--	 eigX[i+j] = (i-1) / (N-1)
	 eigX[i+j] = (i-1) / (N)
      end
   end
end
--eigX[4] = torch.normal(1,0.5)
eigX = eigX:abs()
for i = 1,nX do
   X[i]:cmul(eigX)
--   X[i]:copy(torch.mm(qX:t(), X[i]:reshape(N-1,1)))
   X[i]:copy(torch.mm(qX:t(), X[i]:reshape(N,1)))
end
local targetMat = torch.zeros(N,N)
for i = 1,nX do
   local a = torch.Tensor(N, 1)
--   a[{{1,N-1},{}}]:copy(X[i])
   a:copy(X[i])
--   a[N] = 1
   targetMat:add(torch.mm(a,a:t()))
end
targetMat:mul(2/nX)
print(targetMat)

--[[
local targetMat
local H = randomRotation(N)
local eigv = torch.randn(N)*0.4
local p = torch.randperm(N)
for i = 1,5 do
   eigv[p[i] ] = torch.normal(1,0.4)
end
--eigv[p[1] ] = 1
eigv = eigv:abs()
targetMat = H * torch.diag(eigv) * H:t()
--]]

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

local net = nn.Sequential()
local butterflyRnd = torch.randn(n,N/2)
-- H = Q D Q:t()
local Q = bf2nn(butterflyRnd)
local butterflyNet = Q
net:add(Q)                    -- Q
net:add(nn.CMul(N))           -- D
--net:add(net.modules[2]:clone("weight", "gradWeight"))           -- D
net:add(bfTranspose(Q, true)) -- Q:t()

--net.modules[1] = nn.Identity()
--net.modules[3] = nn.Identity()

local nEpochs = tonumber(opt.nEpochs)
local nSamples = tonumber(opt.nSamples)
local kNormalize = tonumber(opt.kNormalize)
local criterion = nn.MSECriterion()
local config = {learningRate = tonumber(opt.learning_rate), weightDecay = 0,
		momentum = 0, learningRateDecay = tonumber(opt.learning_rate_decay)}
print(config)
local parameters, gradParameters = net:getParameters()
local samples = sampler(nSamples, N)

-- make targets
local bnet = nn.Linear(N-1, 1)
local bparameters, bgradParameters = bnet:getParameters()
local bcriterion = nn.MSECriterion()
local bfeval = function(x)
		  if x ~= bparameters then
		     bparameters:copy(x)
		  end
		  bgradParameters:zero()
		  local err = 0
		  for iX = 1,nX do
		     local input = X[iX]
		     local target = torch.Tensor(1):fill(input[1]) --TODO
		     local output = bnet:forward(input)
		     local lerr = bcriterion:forward(output, target)
		     err = err + lerr
		     local df_do = bcriterion:backward(output, target)
		     bnet:backward(input, df_do)
		  end
		  bgradParameters:div(nX)
		  err = err / nX
		  return err, bgradParameters
	       end

local targets = torch.Tensor(nSamples, N)
for i = 1,nSamples do
   --[[
   local _, gx = bfeval(samples[i])
   gx = gx:clone()
   local _, g0 = bfeval(torch.zeros(N))
   g0 = g0:clone()
   targets[i]:copy(gx-g0)
   --]]
   targets[i] = torch.mm(targetMat, samples[i]:reshape(N, 1))
end

local data = {}
data.opt = opt
data.errors = torch.Tensor(nEpochs)
data.angles = torch.Tensor(nEpochs)
for iEpoch = 1,nEpochs do
   local perm = torch.randperm(nSamples)
   local meanErr = 0
   local meanAngle = 0
   for iSample = 1,nSamples do
      if opt.renew_dataset then
	 samples = sampler(nSamples, N)
      end
      local input = samples[perm[iSample] ]
      --local target = torch.mm(targetMat, input:reshape(input:size(1),1))
      local target = targets[perm[iSample] ]
      local feval = function(x)
		       if x ~= parameters then
			  parameters:copy(x)
		       end
		       gradParameters:zero()
		       local output = net:forward(input)
		       local err = criterion:forward(output, target)
		       local df_do = criterion:backward(output, target)
		       net:backward(input, df_do)
		       bfDistanceToNormalizeAccGrad(butterflyNet, lambda*math.pow(iEpoch, 1.5))
		       meanErr = meanErr + err
		       meanAngle = meanAngle + output:dot(target)/(target:norm()*output:norm())
		       --meanAngle = meanAngle + output:dot(target)
		       return err, gradParameters
		    end
      optim.sgd(feval, parameters, config)
      bfHalfNormalize(butterflyNet)
      --bfNormalize(butterflyNet)
   end
   
   local distToNormalize
   distToNormalize = bfDistanceToNormalize(butterflyNet)
   meanErr = meanErr/nSamples
   meanAngle = math.acos(meanAngle/nSamples)/3.14159*180
   print("meanErr="..meanErr.."  meanAngle="..meanAngle.."  distToNormalize="..distToNormalize.." (epoch "..iEpoch..")")
   data.errors[iEpoch] = meanErr
   data.angles[iEpoch] = meanAngle
end
gnuplot.plot(data.angles, '~')
torch.save("data.out", data)

local AH = torch.Tensor(N,N)
for i = 1,N do
   local a = torch.zeros(N)
   a[i] = 1
   AH[{{},i}]:copy(net:forward(a))
end
--[[
print(targetMat)
print(AH)
--]]
print((AH-targetMat):abs():max())
--print(net.modules[2].weight)
print(torch.eig(AH)[{{},1}])
print(torch.eig(targetMat)[{{},1}])

results = torch.load("results.th")
results[kStepEig] = data
torch.save("results.th", results)