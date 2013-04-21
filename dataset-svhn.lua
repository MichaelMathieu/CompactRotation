require 'torch'
require 'paths'

svhn = {}

svhn.path_dataset = '/media/data_fast/datasets/svhn'
svhn.path_trainset = paths.concat(svhn.path_dataset, 'train_32x32.th')
svhn.path_testset = paths.concat(svhn.path_dataset, 'test_32x32.th')

function svhn.loadTrainSet(maxLoad, geometry)
   return svhn.loadConvDataset(svhn.path_trainset, maxLoad, geometry)
end

function svhn.loadTestSet(maxLoad, geometry)
   return svhn.loadConvDataset(svhn.path_testset, maxLoad, geometry)
end

function svhn.loadFlatDataset(fileName, maxLoad)
   local f = torch.load(fileName)

   local nExample = f.y:size(2)
   local dims = torch.LongStorage{f.X:size(2),f.X:size(3),f.X:size(4)}
   local dim = dims[1]*dims[2]*dims[3]
   if maxLoad and maxLoad > 0 and maxLoad < nExample then
      nExample = maxLoad
      print('<svhn> loading only ' .. nExample .. ' examples')
   end
   print('<svhn> reading ' .. nExample .. ' examples with ' .. dim-1 .. '+1 dimensions...')
   local tensor = torch.Tensor(nExample, dim)
   for i = 1,nExample do
      tensor[i]:copy(f.X[i]:transpose(2,3):reshape(dim))
   end
   print('<svhn> done')

   local dataset = {}
   dataset.tensor = tensor
   dataset.y = f.y
   dataset.dims = dims

   function dataset:normalize(mean_, std_)
      local data = tensor
      local std = std_ or torch.std(data, 1, true)
      local mean = mean_ or torch.mean(data, 1)
      for i=1,dim do
         tensor:select(2, i):add(-mean[1][i])
         if std[1][i] > 0 then
            tensor:select(2, i):mul(1/std[1][i])
         end
      end
      return mean, std
   end

   function dataset:normalizeGlobal(mean_, std_)
      local data = tensor
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std
   end

   dataset.dim = dim

   function dataset:size()
      return nExample
   end

   local labelvector = torch.zeros(10)

   setmetatable(dataset, {__index = function(self, index)
                                       local input = tensor[index]
                                       local class = f.y[1][index]
                                       local label = labelvector:zero()
                                       label[class] = 1
                                       local example = {input, class, label}
                                       return example
                                    end})

   return dataset
end

function svhn.loadConvDataset(fileName, maxLoad, geometry)
   local dataset = svhn.loadFlatDataset(fileName, maxLoad)
   local cdataset = {}
   
   function cdataset:normalize(m,s)
      return dataset:normalize(m,s)
   end
   function cdataset:normalizeGlobal(m,s)
      return dataset:normalizeGlobal(m,s)
   end
   function cdataset:size()
      return dataset:size()
   end

   local ichan = geometry[1]
   assert(ichan == 3)
   local iheight = geometry[2]
   local iwidth = geometry[3]
   local inputpatch = torch.zeros(ichan, iheight, iwidth)

   setmetatable(cdataset, {__index = function(self,index)
                                       local ex = dataset[index]
                                       local input = ex[1]
				       local class = ex[2]
                                       local label = ex[3]
                                       local w = math.sqrt(input:nElement())
                                       local cinput = input:reshape(dataset.dims)
                                       local h = dataset.dims[2]
                                       local w = dataset.dims[3]
                                       local x = math.floor((iwidth-w)/2)+1
                                       local y = math.floor((iheight-h)/2)+1
                                       inputpatch:narrow(3,x,w):narrow(2,y,h):copy(cinput)
                                       local example = {inputpatch, class, label}
                                       return example
                                    end})
   return cdataset
end
