----------------------------------------------------------------------
-- create network in network model
----------------------------------------------------------------------

require 'cunn'
require 'ccn2'

local dropout0 = nn.Dropout(0.5)
local dropout1 = nn.Dropout(0.5)

local model = nn.Sequential()

model:add(nn.Transpose({1,4},{1,3},{1,2}))

model:add(ccn2.SpatialConvolution(3, 192, 5, 1, 2))
model:add(nn.ReLU())
model:add(ccn2.SpatialConvolution(192, 160, 1, 1))
model:add(nn.ReLU())
model:add(ccn2.SpatialConvolution(160, 96, 1, 1))
model:add(nn.ReLU())
model:add(ccn2.SpatialMaxPooling(3, 2))
model:add(dropout0)

model:add(ccn2.SpatialConvolution(96, 192, 5, 1, 2))
model:add(nn.ReLU())
model:add(ccn2.SpatialConvolution(192, 192, 1, 1))
model:add(nn.ReLU())
model:add(ccn2.SpatialConvolution(192, 192, 1, 1))
model:add(nn.ReLU())
model:add(ccn2.SpatialMaxPooling(3, 2))
model:add(dropout1)

model:add(ccn2.SpatialConvolution(192, 192, 3, 1, 1))
model:add(nn.ReLU())
model:add(ccn2.SpatialConvolution(192, 192, 1, 1))
model:add(nn.ReLU())

model:add(nn.SpatialConvolutionMM(192, 10, 1, 1, 1, 1))
model:add(nn.ReLU())
model:add(nn.Transpose({4,1},{4,2},{4,3}))

for i,layer in ipairs(model.modules) do
   if layer.bias then
      layer.bias:fill(0)
      layer.weight:normal(0, 0.05)
   end
end

-------------------------------------------------
--- hacked global average pooling
model:add(nn.SpatialSubSampling(10, 8, 8, 8, 8))
model.modules[#model.modules].weight:fill(1)
model.modules[#model.modules].bias:fill(0)
model.modules[#model.modules].accGradParameters = function () return nil end
-------------------------------------------------

model:add(nn.Reshape(10))
model:add(nn.LogSoftMax())


model:cuda()
loss = nn.ClassNLLCriterion()

----------------------------------------------------------------------
local w, dE = model:getParameters()
local wds = 1e-4
local learningRates = torch.Tensor(w:size(1)):fill(0)
local weightDecays = torch.Tensor(w:size(1)):fill(0)
local counter = 0
for i, layer in ipairs(model.modules) do
   if layer.__typename == 'ccn2.SpatialConvolution' then
      local weight_size = layer.weight:size(1)*layer.weight:size(2)
      learningRates[{{counter+1, counter+weight_size}}]:fill(1)
      weightDecays[{{counter+1, counter+weight_size}}]:fill(wds)
      counter = counter+weight_size
      local bias_size = layer.bias:size(1)
      learningRates[{{counter+1, counter+bias_size}}]:fill(2)
      weightDecays[{{counter+1, counter+bias_size}}]:fill(0)
      counter = counter+bias_size
   elseif layer.__typename == 'nn.SpatialConvolutionMM' then
      local weight_size = layer.weight:size(1)*layer.weight:size(2)
      learningRates[{{counter+1, counter+weight_size}}]:fill(0.1)
      weightDecays[{{counter+1, counter+weight_size}}]:fill(wds)
      counter = counter+weight_size
      local bias_size = layer.bias:size(1)
      learningRates[{{counter+1, counter+bias_size}}]:fill(0.2)
      weightDecays[{{counter+1, counter+bias_size}}]:fill(0)
      counter = counter+bias_size
  end
end
loss:cuda()


learningRates:cuda()
weightDecays:cuda()
-- return package:
return {
   model = model,
   loss = loss,
   dropout = {dropout0, dropout1},
   lrs = learningRates,
   wds = weightDecays
}

