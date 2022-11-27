import pickle as pkl
import metrics

winoground_path = '/saltpool0/data/layneberry/WinoGround/'

model = 'clip' # 'lxmert', 'uniter'

probs = pkl.load(open(winoground_path+model+'_scores.pkl','rb')).cpu()

print('-------Computing Metrics--------')
print('Text to Image')
metrics.compute_metrics(probs)
print('Image to Text')
metrics.compute_metrics(probs.T)
