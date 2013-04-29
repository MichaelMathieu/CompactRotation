function cutEnergy(feval, params, dir, lambda)
   local n = 10
   local imax = 2
   local out = torch.Tensor(n)
   local x = params:clone()
   for i = 1,n do
      x:add(imax/n*lambda, dir)
      out[i] = feval(x)
   end
   return out
end