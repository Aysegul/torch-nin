----------------------------------------------------------------------
-- Train a ConvNet on cifar
-- Original : Clement Farabet
----------------------------------------------------------------------

require 'pl'
require 'torch'
----------------------------------------------------------------------
print '==> processing options'

opt = lapp[[
   -r,--learningRate       (default 1e-3)        learning rate
   -d,--learningRateDecay  (default 0)           learning rate decay)
   -w,--weightDecay        (default 0)           L2 penalty on the weights
   -m,--momentum           (default 0.9)         momentum
   -d,--dropout            (default 0.5)         dropout amount
   -b,--batchSize          (default 64)          batch size
   -t,--threads            (default 8)           number of threads
   -i,--devid              (default 1)           device ID
   -o,--save               (default results)     save directory
]]

-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')


require 'cutorch'
require 'cunn'

cutorch.setDevice(opt.devid)
print('==> using GPU #' .. cutorch.getDevice())
print(cutorch.getDeviceProperties(opt.devid))

----------------------------------------------------------------------
print '==> load modules'

local data  = require 'data'
local train = require 'train'
local test  = require 'test'

----------------------------------------------------------------------
print '==> training!'

while true do
   train(data.trainData)
   test(data.testData)
end
