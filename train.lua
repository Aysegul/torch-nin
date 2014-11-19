----------------------------------------------------------------------
-- original taken from Clement
----------------------------------------------------------------------
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
-- Model + Loss:
local t = require 'model'
local model = t.model
local loss = t.loss
local dropout = t.dropout
local lrs = t.lrs
local wds = t.wds

----------------------------------------------------------------------
print '==> defining some tools'

-- classes
local classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)

-- Log results to files
local trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

----------------------------------------------------------------------
print '==> flattening model parameters'

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
local w,dE_dw = model:getParameters()

local optimState = {
   learningRate = opt.learningRate,
   momentum = opt.momentum,
   dampening = 0,
   weightDecay = opt.weightDecay,
   learningRateDecay = opt.learningRateDecay,
   learningRates = lrs,
   weightDecays = wds
}

print '==> allocating minibatch memory'

local x = torch.Tensor(opt.batchSize,3,32,32)
local yt = torch.Tensor(opt.batchSize)
local target = torch.Tensor(opt.batchSize, 10):fill(0)

if opt.type == 'cuda' then
   x = x:cuda()
   yt = yt:cuda()
   target = target:cuda()
end

----------------------------------------------------------------------
print '==> defining training procedure'

local epoch

local function train(trainData)

   -- epoch tracker
   epoch = epoch or 1
   local time = sys.clock()

   -- shuffle at each epoch
   local shuffle = torch.randperm(trainData:size())

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trainData:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())
      collectgarbage()

      -- batch fits?
      if (t + opt.batchSize - 1) > trainData:size() then
         break
      end

      -- create mini batch
      local idx = 1
      target:fill(0)
      for i = t,t+opt.batchSize-1 do
         x[idx] = trainData.data[shuffle[i]]
         yt[idx] = trainData.labels[shuffle[i]]
         target[idx][trainData.labels[shuffle[i]]] = 1
         idx = idx + 1
      end
      -- create closure to evaluate f(X) and df/dX
      local eval_E = function(w)
         -- reset gradients
         dE_dw:zero()
         -- evaluate function for complete mini batch
         local y = model:forward(x)
         -- estimate df/dW
         local dE_dy = loss:backward(y,target)
         model:backward(x,dE_dy)
         dE_dw:div(opt.batchSize)
         return 0,dE_dw
      end
     optim.sgd(eval_E, w, optimState)

      -- update confusion
      -- dropout off
      for _,d in ipairs(dropout) do
        d.train = false
      end

      local y = model:forward(x)
      for i = 1,opt.batchSize do
         confusion:add(y[i],yt[i])
      end
      -- dropout on
      for _,d in ipairs(dropout) do
        d.train = true
      end


   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(tostring(confusion))

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1


end

-- Export:
return train

