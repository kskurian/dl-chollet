import os, shutil

base_dir = 'C:/Users/10126/PycharmProjects/kaggle'
os.makedirs(base_dir)

train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')

train_cats_dir = os.path.join(train_dir,'cats')
train_dogs_dir = os.path.join(train_dir,'dogs')

valid_cats_dir = os.path.join(validation_dir,'cats')
valid_dogs_dir = os.path.join(validation_dir,'dogs')

test_cats_dir = os.path.join(test_dir,'cats')
test_dogs_dir = os.path.join(test_dir,'dogs')

os.makedirs(train_dir)
os.makedirs(validation_dir)
os.makedirs(test_dir)
os.makedirs(test_cats_dir)
os.makedirs(test_dogs_dir)
os.makedirs(train_cats_dir)
os.makedirs(train_dogs_dir)
os.makedirs(valid_cats_dir)
os.makedirs(valid_dogs_dir)

dataset_dir = 'C:/Users/10126/PycharmProjects/all/train'

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(dataset_dir, fname)
    dst = os.path.join(valid_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(dataset_dir, fname)
    dst = os.path.join(valid_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total training dogs images:', len(os.listdir(train_dogs_dir)))
print('total test dogs images:', len(os.listdir(test_dogs_dir)))
print('total valid cat images:', len(os.listdir(valid_cats_dir)))
print('total valid dogs images:', len(os.listdir(valid_dogs_dir)))