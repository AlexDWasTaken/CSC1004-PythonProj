from multiprocessing import Process, Array, Manager

    
if __name__ == '__main__':
    arr = Manager().list()
    arr.append([[1, 2], [3, 4]])
    arr.append([[1, 2], [3, 4]])
    arr = list(arr)
    print(arr)
        