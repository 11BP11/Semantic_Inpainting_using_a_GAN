import sys
from shutil import copytree, copyfile, rmtree

#print(sys.argv[0])
print(sys.argv[1])

copytree('samples',''+sys.argv[1]+'\samples')
copytree('checkpoint',''+sys.argv[1]+'\checkpoint')
copytree('samples_progress',''+sys.argv[1]+'\samples_progress')
copyfile('main.py',''+sys.argv[1]+'\main.py')
copyfile('model.py',''+sys.argv[1]+'\model.py')
copyfile('ops.py',''+sys.argv[1]+'\ops.py')
copyfile('utils.py',''+sys.argv[1]+'\\utils.py')
#copyfile('utils.py',''+sys.argv[1]+'\utils.py')
copyfile('settings.txt',''+sys.argv[1]+'\settings.txt')

rmtree('samples')
rmtree('checkpoint')
rmtree('samples_progress')