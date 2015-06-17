-- os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
-- os.execute('unzip cifar10torchsmall.zip')
os.execute('wget -c http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz')
os.execute('tar -xvf cifar-100-binary.tar.gz')
require 'torch'
require 'image'
require 'nn'

trainset = torch.load('cifar100-train.t7')
testset = torch.load('cifar100-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
print(trainset)
print(#trainset.data)
--itorch.image(trainset.data[100]) -- display the 100-th image in dataset
print(classes[trainset.label[100]])
-- ignore setmetatable for now, it is a feature beyond the scope of this tutorial. It sets the index operator.
setmetatable(trainset,
    {__index = function(t, i)
                    return {t.data[i], t.label[i]}
                end}
);
trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size()
    return self.data:size(1)
end
print(trainset:size()) -- just to test
print(trainset[33]) -- load sample number 33.
--itorch.image(trainset[33][1])
redChannel = trainset.data[{ {}, {1}, {}, {}  }] -- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}
print(#redChannel)
mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction

    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

net = nn.Sequential()
net:add(nn.SpatialConvolutionMM(3, 128, 5, 5, 1, 1))
   net:add(nn.ReLU())
   net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

   net:add(nn.SpatialConvolutionMM(128, 256, 5, 5, 1, 1))
   net:add(nn.ReLU())
   net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

   net:add(nn.SpatialZeroPadding(1, 1, 1, 1))
   net:add(nn.SpatialConvolutionMM(256, 512, 4, 4, 1, 1))
   net:add(nn.ReLU())

   -- fully connected layers
   net:add(nn.SpatialConvolutionMM(512, 1024, 2, 2, 1, 1))
   net:add(nn.ReLU())
   net:add(nn.Dropout(0.5))
   net:add(nn.SpatialConvolutionMM(1024, 10, 1, 1, 1, 1))

   net:add(nn.Reshape(10))
   --net:add(nn.SoftMax())            -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems
criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.01
trainer.learningRateDecay=0.0024
trainer.maxIteration = 25 -- just do 5 epochs of training.
trainer:train(trainset)
print(classes[testset.label[100]])
--itorch.image(testset.data[100])
testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
for i=1,3 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end
-- for fun, print the mean and standard-deviation of example-100
horse = testset.data[100]
print(horse:mean(), horse:std())
print(classes[testset.label[100]])
--itorch.image(testset.data[100])
predicted = net:forward(testset.data[100])
-- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x
print(predicted:exp())
for i=1,predicted:size(1) do
    print(classes[i], predicted[i])
end
correct = 0
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end
print(correct, 100*correct/10000 .. ' % ')
class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end
for i=1,#classes do
    print(classes[i], 100*class_performance[i]/1000 .. ' %')
end