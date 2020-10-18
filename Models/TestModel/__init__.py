# import os
# import pkgutil
#
# pkgpath = os.path.dirname(__file__)
# pkgname = os.path.basename(pkgpath)
# for _, file, _ in pkgutil.iter_modules([pkgpath]):
#     print(pkgname,file)
#     #from Models.TestModel.+'file'
#     __import__('Models.'+pkgname+'.'+file)