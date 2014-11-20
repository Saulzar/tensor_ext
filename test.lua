
require 'torch'
require 'cutorch'

local tensor_ext = require 'tensor_ext'



local allTypes = {
  torch.CharTensor,
  torch.ByteTensor, 
  torch.ShortTensor, 
  torch.IntTensor, 
  torch.LongTensor,
  torch.FloatTensor,
  torch.DoubleTensor,
  torch.CudaTensor, 
}

local floatTypes = {
  torch.FloatTensor,
  torch.DoubleTensor,
  torch.CudaTensor,   
}

function testIndexSum(n)

  
  for i = 1, n do
   
    local dims = torch.random(3)  
    local dim = torch.random(dims)    
    
    local n = torch.random(1)    

    local d = {}
    local reduced = {}
    
    for i = 1, dims do
      d[i] = torch.random(100)
      reduced[i] = d[i]
    end
    
    x = torch.FloatTensor (unpack(d))
    x:random(5)
    
    inds = torch.LongTensor (d[dim])
    
--     print {i, d, {dims = dims,  dim = dim, n = n}, x, inds, dim}
        
    inds:random(n)    
    reduced[dim] = n
    
    local last = nil
    
    for _, t in pairs(floatTypes) do
      
      local x1 = x:type(t.__typename)
      y = (t (unpack(reduced))):zero()
      
      y:indexSum(dim, inds, x1)
      local y1 = y:float()
           
      if(last) then	
	local eq = (last - y1):abs():max()
	
	assert(eq < 1e-5, "indexSum failed")	
      end
      
      last = y1
    end

  end
  
  print(string.format("indexSum passed %d tests", n))
end


function test() 
  
  local n = 1000
  testIndexSum(n)
  
  
end
  

test()