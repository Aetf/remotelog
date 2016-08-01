fab run_exp:all,BatchDNNTopology,24,'num-workers\\=1','fetcher\\=image','max-spout-pending\\=100000','fat\\=2','use-caffe\\=1','use-gpu\\=2','auto-sleep\\=0','batch-size\\=20'
#mv archive/2 archive/gpu1batch5

#fab run_exp:all,DNNTopology,24,'num-workers\\=1','fetcher\\=image','max-spout-pending\\=100000','fat\\=2','use-caffe\\=1','use-gpu\\=-1','auto-sleep\\=0','batch-size\\=5'
#mv archive/3 archive/gpu2batch5
#
#fab run_exp:all,DNNTopology,24,'num-workers\\=1','fetcher\\=image','max-spout-pending\\=100000','fat\\=2','use-caffe\\=0','use-gpu\\=0','auto-sleep\\=0','batch-size\\=5'
#[lvnguyen@localhost] out: sudo password: 
#mv archive/4 archive/cpubatch5

