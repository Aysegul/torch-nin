----------------------------------------------------------------------
-- Train a ConvNet on SVHN.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'pl'
require 'torch'
----------------------------------------------------------------------
print '==> processing options'

opt = lapp[[
   -r,--learningRate       (default 0.01)       learning rate (0.1)
   -d,--learningRateDecay  (default 0)      learning rate decay (in # samples 1e-7) )
   -w,--weightDecay        (default 0)        L2 penalty on the weights
   -m,--momentum           (default 0.9)         momentum (0.5)
   -d,--dropout            (default 0.5)         dropout amount
   -b,--batchSize          (default 128)         batch size
   -t,--threads            (default 8)           number of threads
   -p,--type               (default cuda)       float or cuda
   -i,--devid              (default 1)           device ID (if using CUDA)
   -s,--size               (default extra)       dataset: small or full or extra
   -o,--save               (default results)     save directory
]]

-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- type:
if opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cutorch'
   require 'cunn'

   cutorch.setDevice(opt.devid)
   print('==> using GPU #' .. cutorch.getDevice())
   print(cutorch.getDeviceProperties(opt.devid))
end

----------------------------------------------------------------------
print '==> load modules'

local data  = require 'data'
local train = require 'train'
local test  = require 'test'


print '==> configuring optimizer'


----------------------------------------------------------------------
print '==> training!'

--[[for i=1, 8 do
   train(data.trainData, optimState)
   test(data.testData)
end



optimState = {
   learningRate = opt.learningRate*0.1,
   momentum = opt.momentum,
   weightDecay = opt.weightDecay,
   learningRateDecay = opt.learningRateDecay
}

]]
while true do
   train(data.trainData)
   test(data.testData)
end
