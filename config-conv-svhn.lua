nEpochs = 1000
nSamples = 73257
nTestSamples = 26032
nMinibatch = 50

geo = {3, 32, 32}
nhidden1 = 50
nhidden2 = 80

net = nn.Sequential()
--stage 1
net:add(nn.SpatialContrastiveNormalization(3))
net:add(nn.SpatialConvolutionMM(3, 64, 5, 5))
net:add(nn.Threshold())
net:add(nn.SpatialMaxPooling(2, 2, 1, 1))
--stage 2
net:add(nn.SpatialSubtractiveNormalization(64))
net:add(nn.SpatialConvolutionMM(64, 128, 7, 7))
net:add(nn.Threshold())
net:add(nn.SpatialMaxPooling(2, 2, 1, 1))
--stage 3
net:add(nn.SpatialSubtractiveNormalization(128))
net:add(nn.SpatialConvolutionMM(128, 128, 7, 7))
net:add(nn.Threshold())
net:add(nn.SpatialMaxPooling(2, 2, 1, 1))
--stage 4
net:add(nn.Reshape(128*4*4))
net:add(nn.Linear(128*4*4, nhidden1))
net:add(nn.Tanh())
--net:add(nn.Linear(nhidden1, nhidden2))
--net:add(nn.Tanh())
--net:add(nn.Linear(nhidden2, 10))
net:add(nn.Linear(nhidden1, 10))

net:add(nn.LogSoftMax())

parameters, gradParameters = net:getParameters()
print(parameters:size(1) .. " parameters")

config = {learningRate = 0.2, learningRateDecay = 1e-4}

crit = nn.ClassNLLCriterion()
confusion = optim.ConfusionMatrix(10)
confusionControl = optim.ConfusionMatrix(10)

ho = HessianOptimizer(parameters:size(1), 0.01, 1e5, 3)
learningRateHO = 0.2
learningRateDecayHO = 1e-4