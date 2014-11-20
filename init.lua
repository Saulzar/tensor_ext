
require 'torch'
require 'sys'
require 'paths'
require 'dok'




-- load C lib
require 'cutorch'
require 'libtensor_ext'


local tensor_ext = {}


local types = {
  torch.CharTensor,
  torch.ByteTensor, 
  torch.ShortTensor, 
  torch.IntTensor, 
  torch.LongTensor,
  torch.FloatTensor,
  torch.DoubleTensor,
  torch.CudaTensor, 
}



for _, t in pairs(types) do
  
  local m = getmetatable(t)
  
  if(not m.indexSum) then
    m.indexSum = t.libtensor_ext.indexSum
  end
  
  if(not m.mod) then
    m.mod = t.libtensor_ext.mod
  end
  
  m.clamp = t.libtensor_ext.clamp
end
  
  


return tensor_ext