----------------------------------------------------------------------
-- This script loads the CIFAR10 dataset
-- training data, and pre-process it to facilitate learning.
-- Clement Farabet
----------------------------------------------------------------------

local tar = 'http://data.neuflow.org/data/cifar10.t7.tgz'
if not paths.dirp('cifar-10-batches-t7') then
   print '==> downloading dataset'
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

-- dataset size: -- will be resized below by opt.smalldata!!!!! be cautious!
local trsize = 50000
local tesize = 10000

trainData = {
   data = torch.Tensor(trsize, 3,32,32),
   labels = torch.Tensor(trsize),
   size = function() return trsize end
}

for i = 0,4 do
   local subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
   trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t():float()
   trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels:float()
end
trainData.labels = trainData.labels + 1

local subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')

testData = {
   data = subset.data:t():float(),
   labels = subset.labels[1]:float(),
   size = function() return tesize end
}
testData.labels = testData.labels + 1

trainData.data = trainData.data:reshape(trsize,3,32,32)
testData.data = testData.data:reshape(tesize,3,32,32)

print '==> global contrast normalization'
for i=1, trainData.data:size(1) do
   local mean = trainData.data[i]:mean()
   local std = trainData.data[i]:std()
   trainData.data[i]:add(-mean)
   trainData.data[i]:div(std)
end
for i=1, testData.data:size(1) do
   local mean = testData.data[i]:mean()
   local std = testData.data[i]:std()
   testData.data[i]:add(-mean)
   testData.data[i]:div(std)
end


----------------------------------------------------------------------
print '==> whiten data'
require 'unsup'
trainData.data, means, P, invP = unsup.zca_whiten(trainData.data, mean, P, invP, 0.1)
testData.data = unsup.zca_whiten(testData.data, means, P, invP, 0.1)

-- Exports
return {
   trainData = trainData,
   testData = testData,
}

