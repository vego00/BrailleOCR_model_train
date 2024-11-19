with open('train_list.txt', 'w') as f:
    for i in range(1, 10):
        f.write('data/original/성북소식지' + str(i).zfill(2) + '.jpg\n')
        
with open('val_list.txt', 'w') as f:
    for i in range(1, 10):
        f.write('data/original/성북소식지' + str(i).zfill(2) + '.json\n')