function demo5

  close all;
  clear all;
  
  fprintf('\n Shannon Entropy 1D');
  
  functionalParams = struct;
  
  params = struct;
  params.alpha = 0.05;
  params.doAsympAnalysis = false;
  
  numX = 5;
  LOO = zeros(numX,1);
  DS = zeros(numX,1);
  n = zeros(numX,1);
  for ind = 1:numX
      n(ind) = 50*(ind^2);
      X = rand(n(ind), 1);
      params.numPartitions = 'loo';
      LOO(ind) = shannonEntropy(X, functionalParams, params);
      params.numPartitions = 2;
      DS(ind) = shannonEntropy(X, functionalParams, params);
  end
  scatter(n, DS, [], 'r');
  hold on;
  ylabel('Entropy');
  xlabel('Size of Data Sample');
  title('Shannon Entropy 1D of uniform random variable');
  scatter(n, LOO, [], 'b');
  legend('Data Split', 'Leave-one-out');
  hold off;
end