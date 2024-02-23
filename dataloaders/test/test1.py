import os
import sys

print(os.getcwd())
#print(os.sys[0])
print(sys.path)
print(os.path.realpath(__file__), os.path.dirname(os.path.realpath("__file__")))
sys.path.insert(0, os.path.realpath(__file__))
print(os.path.abspath(".."))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
#from dataloaders import ECGDatase