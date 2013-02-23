require 'nn'
require 'math'

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
   --from SpatialConvolutionMap
   if stdv then
      stdv = stdv * math.sqrt(3)
      self.weight:apply(function() return torch.uniform(-stdv, stdv) end)
   else
      stdv = 1
      self.weight:apply(function() return torch.uniform(-stdv, stdv) end)
   end
end

function Spaghetti:updateOutput(input)
   self.output:zero()
   for i = 1,self.nCon do
      self.output[TH2table(self.conDst[i])] = self.output[TH2table(self.conDst[i])] + self.weight[i] * input[TH2table(self.conSrc[i])]
   end
   return self.output
end

function Spaghetti:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   for i = 1,self.nCon do
      self.gradInput[TH2table(self.conSrc[i])] = self.gradInput[TH2table(self.conSrc[i])] + self.weight[i] * gradOutput[TH2table(self.conDst[i])]
   end
   return self.gradInput
end

function Spaghetti:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   for i = 1,self.nCon do
      self.gradWeights[i] = self.gradWeights[i] + scale*input[TH2table(self.conSrc[i])]*gradOutput[TH2table(self.conDst[i])]
   end
end

function Spaghetti:decayParameters(decay)
   --from SpatialConvolutionMap
   self.weight:add(-decay, self.weight)
end