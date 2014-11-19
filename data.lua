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

print '==> gcn data'
function gcn(x, scale, bias, epsilon)
   local scale = scale or 55
   local bias = bias or 0
   local epsilon = epsilon or 1e-8

   if x:dim() > 2 then
      local num_samples = x:size(1)
      local length = x:nElement()/num_samples
      x = x:reshape(num_samples, length)
   elseif x:dim() < 2 then
      assert(false)
   end

   -- subtract mean: x = x - mean(x)
   local m = torch.ger(x:mean(2):squeeze(), torch.ones(x:size(2)))
   local xm = torch.add(x, -1, m)

   -- calculate normalizer
   local x_std_v = torch.pow(xm, 2):sum(2):add(bias):sqrt():div(scale)
   x_std_v[torch.lt(x_std_v, epsilon)]:fill(1)

   -- divide by normalizer
   local x_std = torch.ger(x_std_v:mean(2):squeeze(), torch.ones(x:size(2)))
   local x_norm = torch.cdiv(xm, x_std)

   return x_norm
end

trainData.data = gcn(trainData.data)
testData.data = gcn(testData.data)

trainData.data = trainData.data:reshape(50000, 3, 32, 32)
testData.data = testData.data:reshape(10000, 3, 32, 32)

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

