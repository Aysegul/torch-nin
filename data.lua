----------------------------------------------------------------------
-- This script loads the CIFAR10 dataset
-- training data, and pre-process it to facilitate learning.
-- Clement Farabet
----------------------------------------------------------------------

-- download dataset
if not paths.dirp('cifar-10-batches-t7') then
   local www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
   local tar = paths.basename(www)
   os.execute('wget ' .. www .. '; '.. 'tar xvf ' .. tar)
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

print '==> gcn data'
require 'preprocessing'
trainData.data = gcn(trainData.data)
testData.data = gcn(testData.data)

trainData.data = trainData.data:reshape(50000, 3, 32, 32)
testData.data = testData.data:reshape(10000, 3, 32, 32)

----------------------------------------------------------------------
print '==> whiten data'
local means, P = zca_whiten_fit(trainData.data)
trainData.data = zca_whiten_apply(testData.data, means, P)
testData.data  = zca_whiten_apply(testData.data, means, P)

-- Exports
return {
   trainData = trainData,
   testData = testData,
}

