extern "C" void bubbleSort(int *arr, int n)
{
    int i;
    int isSorted = 0;
    int count = 0;

    while (isSorted == 0)
    {
        isSorted = 1;
        for (i = 0; i < n - 1 - count; i++)
        {
            if (arr[i] > arr[i + 1])
            {
                int temp = arr[i];  // arr[i], arr[i + 1] = arr[i + 1], arr[i]
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                isSorted = 0;
            }
        }
        count++;
    }
}

// gcc -shared -o libbubble.so -fPIC py_cpp_test.cpp
