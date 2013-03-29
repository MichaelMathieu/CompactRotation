require 'torch'

function newTimer(name)
   local t = torch.Timer()
   local ret = {}
   local n = 0
   local time = 0
   function ret:tic()
		t:reset()
	     end
   function ret:toc()
		time = time + t:time()['real']
		n = n + 1
		print(name .. " " .. time/n)
	     end
   return ret
end