# Searching

def binary_search(arr, target):
    start = 0
    end = len(arr)-1
  
    while start <= end:
        mid = (start + end)//2
        num = arr[ mid]
        if num == target:
            return True
        elif num < target:
            start = start + 1
        else: 
            end = end -1
    return False
  
def Sequential_Search(arr,target):
    n = len(arr)
    for i in range(n):
        if target == arr[i]:
            return True
    return False

# Sorting
    
def bubble_sort(arr):
    n = len(arr)
    count = 0
    for i in range(n-1):
        already_sorted = True
        for j in range(n-1-i):
            count += 1
            if arr[j] > arr[j+1]:
                arr[j],arr[j+1] =  arr[j+1],arr[j]
                already_sorted = False
        if already_sorted == True:
            break
    return arr,count

def selection_sort(arr):
    n = len(arr)
    count = 0
    for i in range(n-1):
        min_idx=i
        for j in range(i+1,n):
            count += 1
            if arr[min_idx]>arr[j]:
                min_idx = j
        arr[i],arr[min_idx] =  arr[min_idx], arr[i]
    return arr, count


def insertion_sort(arr):
    n = len(arr)
    count = 0
    for i in range(1,n):
        key = arr[i]
        j = i - 1
        while j >= 0 and key<arr[j]:
            count += 1
            arr[j+1]=arr[j]
            j = j - 1
        arr[j+1] = key
    return arr,count
    

# DSA

  # Searching

print("Searching Algorithms:-")
print(binary_search([1,2,3,5,8], 1))
print(Sequential_Search([11,23,58,31,56,77,43,12,65,19],31))

  # Sorting
print("\nSorting Algorithms:-")
print(bubble_sort([14,46,43,27,57,41,45,21,70]))
print(selection_sort([14,46,43,27,57,41,45,21,70]))
print(insertion_sort([14,46,43,27,57,41,45,21,70]))

arr = list(range(1000,1,-2))

print("\nSorting Worst Case Algorithms:-")
print(bubble_sort(arr)[1])
print(selection_sort(arr)[1])
print(insertion_sort(arr)[1])


print("\nSorting Best Case Algorithms:-")
print(bubble_sort(arr)[1])
print(selection_sort(arr)[1])
print(insertion_sort(arr)[1])