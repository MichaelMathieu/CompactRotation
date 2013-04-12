nEpochs = 1000
nSamples = 60000
nTestSamples = 10000
nMinibatch = 50

geo = {32, 32}
nhidden1 = 120
nhidden2 = 80

train_set = mnist.loadTrainSet(nSamples, geo)
test_set  = mnist.loadTestSet (nTestSamples, geo)

net = nn.Sequential()
--stage 1
net:add(nn.SpatialContrastiveNormalization())
net:add(nn.SpatialConvolutionMM(1, 6, 5, 5))
net:add(nn.Tanh())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
--stage 2
net:add(nn.SpatialSubtractiveNormalization(6))
net:add(nn.SpatialConvolutionMM(6, 16, 5, 5))
net:add(nn.Tanh())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
--stage 3
net:add(nn.Reshape(16*5*5))
net:add(nn.Linear(16*5*5, nhidden1))
net:add(nn.Tanh())
net:add(nn.Linear(nhidden1, nhidden2))
net:add(nn.Tanh())
net:add(nn.Linear(nhidden2, 10))

net:add(nn.LogSoftMax())

parameters, gradParameters = net:getParameters()
print(parameters:size(1) .. " parameters")

config = {learningRate = 0.5, learningRateDecay = 1e-4}

crit = nn.ClassNLLCriterion()
confusion = optim.ConfusionMatrix(10)
confusionControl = optim.ConfusionMatrix(10)

ho = HessianOptimizer(parameters:size(1), 0.01, 1e5)

