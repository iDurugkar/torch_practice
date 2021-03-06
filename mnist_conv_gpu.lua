require 'nn';
require 'image';
require 'cunn';
require 'cutorch';

cutorch.setDevice(1);

train = torch.load('mnist.t7/train_32x32.t7', 'ascii')
test = torch.load('mnist.t7/test_32x32.t7', 'ascii')

train.data = train.data:transpose(3,4):double();
mean = train.data:mean();
std = train.data:std();
train.data:add(-mean);
train.data:div(std);

dataset = {};
function dataset:size() return train.data:size(1) end
for i=1,dataset:size() do
    local input = torch.DoubleTensor(1, 32, 32);
    input[{1,{},{}}] = train.data[{i, 1, {}, {}}]
    local output = train.labels[i];
    dataset[i] = {input:cuda(), output};
    end

--print(dataset)
------------------------------------

width = 32;
height = 32;
inputs = width*height;
outputs = 10;
HUs = 300;

------------------------------------

cout = 14;

------------------------------------

nconv = 40;
filtsize = 5;
poolsize = 2;
normkernel = image.gaussian1D(7);

------------------------------------

model = nn.Sequential();

model:add(nn.SpatialConvolutionMM(1, nconv, filtsize, filtsize));
model:add(nn.ReLU());
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))


model:add(nn.Reshape(nconv*cout*cout));
model:add(nn.Linear(nconv*cout*cout,HUs));
--model:add(nn.Tanh());
--model:add(nn.Linear(HUs,outputs));
model:add(nn.LogSoftMax())

timer = torch.Timer()

criterion = nn.ClassNLLCriterion();
trainer = nn.StochasticGradient(model, criterion);
trainer.learningRate = 0.01;
trainer.maxIteration = 15;

model = model:cuda();
criterion = criterion:cuda();
trainer = trainer;
--dataset.input = dataset.input:cuda();

trainer:train(dataset);

test.data = test.data:transpose(3,4):double();
test.data:add(-mean);
test.data:div(std);
print(#test.data)
--test_data = torch.randn(1, 32, 32):cuda();
--[[for i=1,test_data:size(1) do
	test_data[{i,1,{},{}}] = test.data[{i,1,{},{}}]
    end
--]]
--outputs = model:forward(test_data);
test.labels = test.labels:double();

error = 0;
for i=1,10000 do
    pred = 1;
    test_data = test.data[{i,{},{},{}}]:cuda()
    outputs = model:forward(test_data);
    val = outputs[1];
    for j=1,outputs:size(1) do
        if outputs[j] > val then
            pred = j;
            val = outputs[j];
            end
        end
    if pred ~= test.labels[i] then
        error =error + 1;
        end
    end
print('The time it took was  '.. timer:time().real .. 'seconds');

print('Error is: ' .. error*100/10000);