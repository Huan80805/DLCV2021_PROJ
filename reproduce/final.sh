echo 'image input file: $1'
echo 'image output file: $2'

wget https://www.dropbox.com/s/o1wliasjpi2guot/model_300.bin?dl=1 -O model_300.bin
wget https://www.dropbox.com/s/kpkks4fds9mtoig/model_600.bin?dl=1 -O model_600.bin
wget https://www.dropbox.com/s/4ej2lnf0zpdqgo8/model_900.bin?dl=1 -O model_900.bin
wget https://www.dropbox.com/s/ef879t6687uf6i1/model_1200.bin?dl=1 -O model_1200.bin
wget https://www.dropbox.com/s/3og20qkogqdjp7i/fastrcnn.bin?dl=1 -O fastrcnn.bin
wget https://www.dropbox.com/s/pkyj4qci7wdd47a/alex_acc0823.pth?dl=1 -O alex_acc0823.pth
wget https://www.dropbox.com/s/rseym7w1bohfc7l/alex_finetune.pth?dl=1 -O alex_finetune.pth

python3 ./final_output_voting.py -i $1

python3 ./fasterrcnn_reproduce.py $1 ./voting.csv $2
