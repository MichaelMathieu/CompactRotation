nEpochs = 1000
nSamples = 60000
nTestSamples = 10000
nMinibatch = 50

geo = {28, 28}
nhidden = 500

train_set = mnist.loadTrainSet(nSamples, geo)
test_set  = mnist.loadTestSet (nTestSamples, geo)

net = nn.Sequential()
net:add(nn.SpatialContrastiveNormalization())
net:add(nn.Reshape(geo[1]*geo[2]))
net:add(nn.Linear(geo[1]*geo[2], nhidden))
net:add(nn.Tanh())
net:add(nn.Linear(nhidden, nhidden))
net:add(nn.Tanh())
net:add(nn.Linear(nhidden, nhidden))
net:add(nn.Tanh())
net:add(nn.Linear(nhidden, 10))
net:add(nn.LogSoftMax())

parameters, gradParameters = net:getParameters()
print(parameters:size(1) .. " parameters")

config = {learningRate = 0.5, learningRateDecay = 1e-4}

crit = nn.ClassNLLCriterion()
confusion = optim.ConfusionMatrix(10)
confusionControl = optim.ConfusionMatrix(10)

ho = HessianOptimizer(parameters:size(1), 0.01, 1e5)

