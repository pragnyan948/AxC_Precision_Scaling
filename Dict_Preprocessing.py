
InputFeaturesDict({
'data type': Text(shape=(), dtype=string),
'dataset': Text(shape=(), dtype=string),
'algorithm': Text(shape=(), dtype=string),
'pe type': Text(shape=(), dtype=string),
'hardware': Text(shape=(), dtype=string),
'quantization method': Text(shape=(), dtype=string),
'optimization method': Text(shape=(), dtype=string),
})

MidFeaturesDict({
'parallel': bool,
'analog': bool,
'bit slice': Tensor(shape=(1,),dtype=float32),
'mean dataset': float32,
'sigma dataset': float32,
'variance dataset': float32,
'mean sparsity dataset': float32,
'uniform quant':Tensor(shape=(1,),dtype=bool),
'levels quant': Tensor(shape=(1,),dtype=float32),
'max error quant': Tensor(shape=(1,),dtype=float32),
'min error quant': Tensor(shape=(1,),dtype=float32),
'average error quant': Tensor(shape=(1,),dtype=float32),
'number conv layers': int64,
'number act layers': int64,
'act type': Text(shape=(), dtype=string),
'reconfigurable precision': bool,
'single precision': bool,
'hybrid data types': bool,
'train': bool,
'resilience neuron train': Tensor(shape=(1,),dtype=bool),
'resilience weight train': Tensor(shape=(1,),dtype=bool),
})

OutputFeaturesDict({
'iso accuracy':ClassLabel(shape=(),dtype=int64,num classes=2),
'diagnostics':Tensor(shape=(1,),dtype=float32),
'cost reduction' : float32,
})