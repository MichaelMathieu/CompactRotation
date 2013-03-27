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

function Spaghetti:__init(conSrc, conDst, dimDst, conWeights)
   parent.__init(self)
   self.nCon = conSrc:size(1)
   self.conSrc = conSrc:long():add(-1) -- zero-based
   self.conDst = conDst:long():add(-1) -- zero-based
   self.conWeights = conWeights or torch.linspace(1, self.nCon, self.nCon)
   self.conWeights = self.conWeights:long()
   self.weight = torch.Tensor(self.conWeights:max())
   self.conWeights:add(-1) -- careful : zero-based
   self.gradWeight = torch.Tensor():resizeAs(self.weight)
   self.output = torch.Tensor(dimDst)
   self.gradInput = torch.Tensor()
   self:reorganize()
   self:reset()
end

function Spaghetti:reorganize()
   local dimDst = self.output:size()
   local function getOrdering(A)
      local conDstSingle = torch.LongTensor(A:size(1))
      if A:nDimension() > 1 then
	 conDstSingle:copy(A:select(2,1))
	 for dim = 2,dimDst:size() do
	    conDstSingle:mul(dimDst[dim])
	    conDstSingle:add(A:select(2, dim))
	 end
      else
	 conDstSingle:copy(A)
      end
      local ordered = torch.LongTensor():resizeAs(conDstSingle)
      local order = torch.LongTensor():resizeAs(conDstSingle)
      libhessian.sort(conDstSingle, ordered, order) --because torch sort suck
      local outOrder = order--:add(-1) -- seriously, screw 1-based arrays
      local outOrderChunks = {0}
      local lasti = 1
      for i = 1,ordered:size(1) do
	 if ordered[i] ~= ordered[lasti] then
	    table.insert(outOrderChunks, i-1)
	    lasti = i
	 end
      end
      table.insert(outOrderChunks, self.nCon)
      outOrderChunks = torch.Tensor(outOrderChunks):long()
      outOrder = outOrder:contiguous()
      outOrderChunks = outOrderChunks:contiguous()
      return outOrder, outOrderChunks
   end

   self.dstOrder, self.dstOrderChunks = getOrdering(self.conDst)
   self.srcOrder, self.srcOrderChunks = getOrdering(self.conSrc)
   self.weiOrder, self.weiOrderChunks = getOrdering(self.conWeights)
   
   self.conSrc = self.conSrc:contiguous()
   self.conDst = self.conDst:contiguous()
   self.conWeights = self.conWeights:contiguous()
   print("sorting done")
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
				     self.conWeights,
				     self.weight, self.output,
				     self.dstOrder, self.dstOrderChunks);
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
					self.conWeights,
					self.weight, gradOutput, self.gradInput,
					self.srcOrder, self.srcOrderChunks)
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
   libhessian.spaghetti_accGradParameters(input, self.conSrc, self.conDst,
					  self.conWeights, gradOutput, scale,
					  self.gradWeight,
					  self.weiOrder, self.weiOrderChunks)
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