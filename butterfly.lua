require 'torch'
require 'math'
require 'spaghetti'
require 'libhessian'

-- butterfly are 2d tensors, lg(N)x(N/2). n = lg(N)

function bf2matElem(butterflyElem, stride)
   local N = butterflyElem:size(1)*2
   local out = torch.Tensor(N,N):zero();
   for i = 1,(N-stride),(2*stride) do
      for j = i,(i+stride-1) do
	 local alpha = butterflyElem[(i-1)/2+1+j-i]
	 local ca = math.cos(alpha)
	 local sa = math.sin(alpha)
	 out[j][j] = ca
	 out[j][j+stride] = sa
	 out[j+stride][j] = -sa
	 out[j+stride][j+stride] = ca
      end
   end
   return out
end

function bf2mat(butterfly)
   local n = butterfly:size(1)
   local N = butterfly:size(2)*2
   assert(2^n == N)
   local out = torch.eye(N)
   for i = 1,n do
      out = torch.mm(out, bf2matElem(butterfly[i], 2^(i-1)))
   end
   return out
end

function bf2nnElem(butterflyElem, stride)
   local N = butterflyElem:size(1)*2
   local conSrc = torch.LongTensor(N*2,1)
   local conDst = torch.LongTensor(N*2,1)
   local weights = torch.Tensor(N*2)
   local k = 1
   for i = 1,(N-stride),(2*stride) do
      for j = i,(i+stride-1) do
	 local alpha = butterflyElem[(i-1)/2+1+j-i]
	 local ca = math.cos(alpha)
	 local sa = math.sin(alpha)
	 conSrc[k] = j
	 conDst[k] = j
	 weights[k] = ca
	 k = k + 1
	 conSrc[k] = j+stride
	 conDst[k] = j
	 weights[k] = sa
	 k = k + 1
	 conSrc[k] = j
	 conDst[k] = j+stride
	 weights[k] = -sa
	 k = k + 1
	 conSrc[k] = j+stride
	 conDst[k] = j+stride
	 weights[k] = ca
	 k = k + 1
      end
   end
   local out = nn.Spaghetti(conSrc, conDst, torch.LongStorage{N})
   out.weight:copy(weights)
   return out
end

function bf2nn(butterfly)
   local n = butterfly:size(1)
   local N = butterfly:size(2)*2
   assert(2^n == N)
   local out = nn.Sequential()
   for i = n,1,-1 do
      out:add(bf2nnElem(butterfly[i], 2^(i-1)))
   end
   return out
end

function bfNormalize(net)
   for iModule = 1,#net.modules do
      local netElem = net.modules[iModule]
      libhessian.bfNormalize(netElem.weight)
      --[[
      for i = 1,netElem.weight:size(1),4 do
	 local c = 0.5 * (netElem.weight[i  ] + netElem.weight[i+3])
	 local s = 0.5 * (netElem.weight[i+1] - netElem.weight[i+2])
	 local normalizer = 1/math.sqrt(c*c+s*s)
	 netElem.weight[i  ] =  c * normalizer
	 netElem.weight[i+1] =  s * normalizer
	 netElem.weight[i+2] = -s * normalizer
	 netElem.weight[i+3] =  c * normalizer
      end
      --]]
   end
end

function bfHalfNormalize(net)
   for iModule = 1,#net.modules do
      local netElem = net.modules[iModule]
      libhessian.bfHalfNormalize(netElem.weight)
   end
end   

function bfTransposeElem(module, shareWeights)
   local out = nn.Spaghetti(module.conDst:clone(),
			    module.conSrc:clone(),
			    module.output:size())
   if shareWeights then
      --out.weight = module.weight
      out.weight:set(module.weight)
      out.gradWeight:set(module.gradWeight)
   else
      out.weight:copy(module.weight)
   end
   return out
end

function bfTranspose(net, shareWeights)
   local out = nn.Sequential()
   for i = #net.modules,1,-1 do
      out:add(bfTransposeElem(net.modules[i], shareWeights))
   end
   return out
end

function bfDistanceToNormalize(net)
   local function sq(a)
      return a*a
   end
   local out = 0
   local n = 0
   for i = 1,#net.modules do
      local mod = net.modules[i]
      for j = 1,mod.weight:size(1),2 do
	 out = out + sq(1 - sq(mod.weight[j]) - sq(mod.weight[j+1]))
      end
      n = n + mod.weight:size(1)
   end
   return out/n
end

function bfDistanceToNormalizeAccGrad(net, lambda)
   local function sq(a)
      return a*a
   end
   local n = 0
   for i = 1,#net.modules do
      n = n + net.modules[i].weight:size(1)
   end
   for i = 1,#net.modules do
      local mod = net.modules[i]
      for j = 1,mod.weight:size(1),2 do
	 local d = 4*lambda / n * (1 - sq(mod.weight[j]) - sq(mod.weight[j+1]))
	 mod.gradWeight[j  ] = mod.gradWeight[j  ] - d * mod.weight[j]
	 mod.gradWeight[j+1] = mod.gradWeight[j+1] - d * mod.weight[j+1]
      end
   end
end

function bf2matElem_testme()
   local butterflyElem = torch.randn(8)
   print(bf2matElem(butterflyElem, 1))
   print(bf2matElem(butterflyElem, 2))
   print(bf2matElem(butterflyElem, 4))
   print(bf2matElem(butterflyElem, 8))
end

function bf2mat_testme()
   local n = 4
   local butterfly = torch.randn(n,2^(n-1))
   print(bf2mat(butterfly))
   assert((bf2mat(butterfly)*bf2mat(butterfly):t()-torch.eye(2^n)):abs():max()<1e-8)
end

function bf2nnElem_testme()
   local N = 16
   local butterflyElem = torch.randn(N/2)
   for i = 0,3 do
      local nn1 = bf2nnElem(butterflyElem, 2^i)
      local m1 = bf2matElem(butterflyElem, 2^i)
      for j = 1,20 do
	 local v = torch.randn(N)
	 assert((nn1:forward(v)-torch.mm(m1,v:reshape(v:size(1),1))):abs():max()<1e-8)
      end
   end
end

function bf2nn_testme()
   local n = 4
   local butterfly = torch.randn(n,2^(n-1))
   local nn1 = bf2nn(butterfly)
   local m = bf2mat(butterfly)
   for j = 1,100 do
      local v = torch.randn(2^n)
      assert((nn1:forward(v)-torch.mm(m,v:reshape(v:size(1),1))):abs():max() < 1e-6)
   end
end


function bf2nnElem_grad_testme()
   local N = 16
   local butterflyElem = torch.randn(N/2)
   for i = 0,3 do
      local nn1 = bf2nnElem(butterflyElem, 2^i)
      for j = 1,100 do
	 local v = torch.randn(N)
	 local u = torch.randn(N)
	 local h = torch.randn(N):mul(1e-2)
	 local nnvh = nn1:forward(v+h):clone()
	 local nnv = nn1:forward(v):clone()
	 local nngv = nn1:updateGradInput(v, u):clone()
	 assert(math.abs(torch.dot(nnvh-nnv,u) - torch.dot(nngv,h))<1e-7)
      end
   end
end

function bfTranspose_testme()
   local n = 5
   local N = 2^n
   local butterfly = torch.randn(n, N/2)
   local nn1 = bf2nn(butterfly)
   local nn1t = bfTranspose(nn1, false)
   for j = 1,100 do
      local v = torch.randn(N)
      local nnv = nn1:forward(v)
      local nntnnv = nn1t:forward(nnv)
      assert((v-nntnnv):abs():max()<1e-5)
   end
end