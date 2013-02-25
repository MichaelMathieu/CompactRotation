require 'qr'

function randomRotation(n)
   -- from http://www.mathworks.com/matlabcentral/newsreader/view_thread/298500
   -- TODO: check if it is really uniform  
   local out
   local det = -1
   while det < 0 do
      local A = torch.randn(n,n)
      local Q, R = QR(A)
      out = torch.mm(Q,torch.diag(R:diag():sign()))
      det = torch.prod(R:diag():sign(),1)[1]
   end
   return out
end