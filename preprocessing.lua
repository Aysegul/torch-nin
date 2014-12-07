-- https://github.com/lisa-lab/pylearn2/blob/14b2f8bebce7cc938cfa93e640008128e05945c1/pylearn2/datasets/preprocessing.py

function zca_whiten_fit(data, bias)
   local bias= bias or 1e-1
   local auxdata = data:clone()
   local dims = data:size()
   local nsamples = dims[1]
   local n_dimensions = data:nElement() / nsamples

   if data:dim() >= 3 then
      auxdata = auxdata:view(nsamples, n_dimensions)
   end

   -- Center data
   means = torch.mean(auxdata, 1):squeeze()
   auxdata = auxdata - torch.ger(torch.ones(nsamples),means)


   bias = torch.eye(n_dimensions)*bias
   c = torch.mm(auxdata:t(),auxdata)
   c:div(nsamples):add(bias)
   local ce,cv = torch.symeig(c,'V')

   ce:sqrt()

   local invce = ce:clone():pow(-1)
   local invdiag = torch.diag(invce)
   P = torch.mm(cv, invdiag)
   P = torch.mm(P, cv:t())


   --local diag = torch.diag(ce)
   --invP = torch.mm(cv, diag)
   --invP = torch.mm(invP, cv:t())

   return means, P  --, invP
end

function zca_whiten_apply(data, means, P)

   local auxdata = data:clone()
   local dims = data:size()
   local nsamples = dims[1]
   local n_dimensions = data:nElement() / nsamples

    if data:dim() >= 3 then
       auxdata = auxdata:view(nsamples, n_dimensions)
    end


   local xmeans = means:new():view(1,n_dimensions):expand(nsamples,n_dimensions)
   auxdata:add(-1, xmeans)
   auxdata = torch.mm(auxdata, P)

   auxdata:resizeAs(data)
   return auxdata

end