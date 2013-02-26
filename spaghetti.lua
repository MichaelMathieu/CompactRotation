require 'nn'
require 'math'
require 'libhessian'

--TODO: this module is *NOT* optimized

local Spaghetti, parent = torch.class('nn.Spaghetti', 'nn.Module')

local function TH2table(t)
   local out = {}
   assert(t:nDimension() == 1)
   for i = 1,t:size(1) do
      out[i] = t[i]
   end
   return out
end

function Spaghetti:__init(conSrc, conDst, dimDst)
   parent.__init(self)
   self.nCon = conSrc:size(1)
   self.conSrc = conSrc
   self.conDst = conDst
   self.weight = torch.Tensor(self.nCon)
   self.gradWeight = torch.Tensor():resizeAs(self.weight)
   self.output = torch.Tensor(dimDst)
   self.gradInput = torch.Tensor()
   self:reset()
end

function Spaghetti:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)   --from SpatialConvolutionMap TODO: why??
      self.weight:apply(function() return torch.uniform(-stdv, stdv) end)
   else
      stdv = 1
      self.weight:apply(function() return torch.uniform(-stdv, stdv) end)
   end
end

function Spaghetti:updateOutput(input)
   libhessian.spaghetti_updateOutput(input, self.conSrc, self.conDst,
				     self.weight, self.output);
   return self.output
   --[[
   self.output:zero()
   for i = 1,self.nCon do
      self.output[TH2table(self.conDst[i])] = self.output[TH2table(self.conDst[i])] + self.weight[i] * input[TH2table(self.conSrc[i])]
   end
   return self.output
   --]]
end

function Spaghetti:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   libhessian.spaghetti_updateGradInput(input, self.conSrc, self.conDst,
					self.weight, gradOutput, self.gradInput)
   return self.gradInput
   --[[
   self.gradInput:resizeAs(input):zero()
   for i = 1,self.nCon do
      self.gradInput[TH2table(self.conSrc[i])] = self.gradInput[TH2table(self.conSrc[i])] + self.weight[i] * gradOutput[TH2table(self.conDst[i])]
   end
   return self.gradInput
   --]]
end

function Spaghetti:accGradParameters(input, gradOutput, scale)
   --TODO: unit test this function
   scale = scale or 1
   libhessian.spaghetti_accGradParameters(input, self.conSrc, self.conDst, self.weight,
					  gradOutput, scale, self.gradWeight)
   --[[
   for i = 1,self.nCon do
      self.gradWeight[i] = self.gradWeight[i] + scale*input[TH2table(self.conSrc[i])]*gradOutput[TH2table(self.conDst[i])]
   end
   --]]
end

function Spaghetti:decayParameters(decay)
   --from SpatialConvolutionMap, TODO: check what it is used for
   self.weight:add(-decay, self.weight)
end