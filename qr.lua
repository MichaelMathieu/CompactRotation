require 'libhessian'

function QR(A)
   local m = A:size(1)
   local n = A:size(2)
   local B = A:clone()
   local Q = torch.Tensor(m,m)
   local R = torch.Tensor(m,n)
   libhessian.QR(B, Q, R)
   return Q, R
end