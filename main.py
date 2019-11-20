from testing import Testing
from training import Training
from processing import PimaProcessing

def main():
    process = PimaProcessing()
    process.open_file_and_store()
    process.list_to_array()
    print()
    def 